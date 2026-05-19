#include "llama_bridge.h"
#include <atomic>
#include <llama.h>
#include <minja.hpp> // imports nhlohmann json into global namespace??
#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

using namespace std;

const char *llama_bridge_get_devices(const char *backend_path)
{
    thread_local string devbuf;
    if (backend_path && backend_path[0])
        ggml_backend_load_all_from_path(backend_path);
    else
        ggml_backend_load_all();

    size_t n_devs = ggml_backend_dev_count();
    json ret_json = json::array();
    for (size_t dev_i = 0; dev_i < n_devs; dev_i++)
    {
        ggml_backend_device *dev = ggml_backend_dev_get(dev_i);
        json j;
        j["backend"] = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
        j["desc"] = ggml_backend_dev_description(dev);
        j["name"] = ggml_backend_dev_name(dev);
        size_t free_mem = 0;
        size_t total_mem = 0;
        ggml_backend_dev_memory(dev, &free_mem, &total_mem);
        j["free_mem"] = free_mem;
        j["total_mem"] = total_mem;
        ret_json.push_back(j);
    }
    if (ret_json.empty())
    {
        ret_json = json::object();
        ret_json["err"] = "No devices found";
    }
    devbuf = ret_json.dump();
    return devbuf.c_str();
}

struct llama_bridge_obj
{
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    const llama_vocab *vocab = nullptr;
    llama_sampler *smpl = nullptr;
    ggml_backend_dev_t *devices = nullptr;
    shared_ptr<minja::TemplateNode> tmpl;

    int n_ctx;
    int n_pos = 0;

    std::atomic<bool> halt_flag{false};

    int32_t top_k = 20;
    double top_p = 0.95;
    double min_p = -0.8;
    double temp = 0.7;
    double presence_penalty = 1.5;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    bool thinking = false;
    int think_budget = -1; // -1 means use default (n_ctx / 8)
    llama_token end_think_token = -1;

    vector<llama_token> cached_tokens; // Tracks exact tokens in the KV cache
    json chat_history = json::array();

    llama_bridge_obj(const char *model_path, const char *device_name)
    {
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = -1;

        if (!device_name || !device_name[0])
        {
            throw runtime_error("device_name must be provided");
        }

        if (ggml_backend_dev_t dev = ggml_backend_dev_by_name(device_name))
        {
            devices = new ggml_backend_dev_t[2]{dev, nullptr};
            model_params.devices = devices;
        }
        else
        {
            throw runtime_error(string("Device not found: ") + device_name);
        }

        model = llama_model_load_from_file(model_path, model_params);
        if (!model)
        {
            delete[] devices;
            throw runtime_error("Unable to load model");
        }

        vocab = llama_model_get_vocab(model);
        const char *raw_tmpl = llama_model_chat_template(model, nullptr);
        std::string new_tmpl = regex_replace(raw_tmpl, regex{"multi_step_tool=true"}, "multi_step_tool=false");
        tmpl = minja::Parser::parse(new_tmpl, {});

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 48 * 1024;
        ctx_params.n_batch = 512;
        ctx_params.no_perf = true;
        ctx_params.flash_attn_type = llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_AUTO;
        ctx_params.type_k = GGML_TYPE_Q8_0;
        ctx_params.type_v = GGML_TYPE_Q8_0;

        ctx_params.offload_kqv = determine_offload_kqv(ctx_params);

        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx)
        {
            llama_model_free(model);
            delete[] devices;
            throw runtime_error("Unable to create context");
        }

        n_ctx = llama_n_ctx(ctx);
        think_budget = n_ctx / 8;
        set_sampler_params(temp, top_k, min_p, top_p, presence_penalty, seed);

        // Resolve the </think> token ID from the vocab
        const char *end_think_str = "</think>";
        llama_token tok_buf[8];
        int n = llama_tokenize(vocab, end_think_str, (int)strlen(end_think_str), tok_buf, 8, false, true);
        if (n == 1)
        {
            end_think_token = tok_buf[0];
        }
        else
        {
            throw runtime_error("Could not resolve </think> token");
        }
    }

    bool determine_offload_kqv(const llama_context_params &ctx_params)
    {
        ggml_backend_dev_t active_dev = devices[0];

        // CPU device doesn't have VRAM constraints, return default offload status (true)
        if (ggml_backend_dev_type(active_dev) == GGML_BACKEND_DEVICE_TYPE_CPU)
        {
            return true;
        }

        size_t free_mem = 0;
        size_t total_mem = 0;
        ggml_backend_dev_memory(active_dev, &free_mem, &total_mem);

        // Estimate KV Cache Size
        int32_t n_layer = llama_model_n_layer(model);
        int32_t n_head = llama_model_n_head(model);
        int32_t n_head_kv = llama_model_n_head_kv(model);
        int32_t n_embd = llama_model_n_embd(model);
        int32_t head_dim = n_head > 0 ? (n_embd / n_head) : 0;

        int32_t n_embd_k_gqa = n_head_kv * head_dim;
        int32_t n_embd_v_gqa = n_head_kv * head_dim;

        size_t row_size_k = ggml_row_size(ctx_params.type_k, n_embd_k_gqa);
        size_t row_size_v = ggml_row_size(ctx_params.type_v, n_embd_v_gqa);

        size_t layer_kv_size = row_size_k * ctx_params.n_ctx + row_size_v * ctx_params.n_ctx;
        size_t total_kv_cache_size = layer_kv_size * n_layer;

        // 1 GB Safety Margin
        size_t safety_margin = 1ULL << 30;

        // Retrieve active llama logging callback and log details
        ggml_log_callback log_callback = nullptr;
        void *log_user_data = nullptr;
        llama_log_get(&log_callback, &log_user_data);

        if (log_callback)
        {
            char log_buf[256];
            snprintf(log_buf, sizeof(log_buf),
                     "[llama-bridge] Active Device: %s, Free VRAM: %.2f MB, Estimated KV Cache Size: %.2f MB\n",
                     ggml_backend_dev_name(active_dev), (double)free_mem / (1024.0 * 1024.0),
                     (double)total_kv_cache_size / (1024.0 * 1024.0));
            log_callback(GGML_LOG_LEVEL_INFO, log_buf, log_user_data);
        }

        if (free_mem < total_kv_cache_size + safety_margin)
        {
            if (log_callback)
            {
                log_callback(GGML_LOG_LEVEL_WARN,
                             "[llama-bridge] Insufficient VRAM to offload KV cache with a 1GB safety margin. Disabling "
                             "offload_kqv.\n",
                             log_user_data);
            }
            return false;
        }

        if (log_callback)
        {
            log_callback(GGML_LOG_LEVEL_INFO, "[llama-bridge] Sufficient VRAM detected. Enabling offload_kqv.\n",
                         log_user_data);
        }
        return true;
    }

    ~llama_bridge_obj()
    {
        if (devices)
            delete[] devices;
        if (smpl)
            llama_sampler_free(smpl);
        if (ctx)
            llama_free(ctx);
        if (model)
            llama_model_free(model);
    }

    std::string apply_chat_template(const std::vector<llama_chat_message> &msgs, bool thinking_flag, bool add_ass)
    {
        json minja_ctx = json::object();
        json minja_ctx_msgs = json::array();
        for (const auto &msg : msgs)
        {
            std::string role(msg.role);
            if (role != "user" && role != "assistant" && role != "system")
            {
                throw std::runtime_error(std::string("Invalid role: ") + std::string(role));
            }
            minja_ctx_msgs.push_back(json{{"role", msg.role}, {"content", msg.content}});
        }
        minja_ctx["messages"] = minja_ctx_msgs;
        minja_ctx["enable_thinking"] = thinking_flag;
        minja_ctx["add_generation_prompt"] = add_ass;
        minja_ctx["multi_step_tool"] = false;
        return tmpl->render(minja::Context::make(minja::Value(minja_ctx)));
    }

    void set_sampler_params(double temp_in, int32_t top_k_in, double min_p_in, double top_p_in, double presence_penalty_in, int64_t seed_in)
    {
        temp = temp_in;
        top_k = top_k_in;
        min_p = min_p_in;
        top_p = top_p_in;
        presence_penalty = presence_penalty_in;
        seed = (uint32_t)seed_in;
        if (smpl)
            llama_sampler_free(smpl);
        auto sparams = llama_sampler_chain_default_params();
        sparams.no_perf = true;
        smpl = llama_sampler_chain_init(sparams);
        if (top_k >= 0)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        if (top_p >= 0)
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p((float)top_p, 1));
        if (min_p >= 0)
            llama_sampler_chain_add(smpl, llama_sampler_init_min_p((float)min_p, 1));
        if (presence_penalty != 0.0)
            llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, 1.0f, 0.0f, (float)presence_penalty));
        if (temp > 0.0)
        {
            llama_sampler_chain_add(smpl, llama_sampler_init_temp((float)temp));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
        }
        else
        {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        }
    }
    std::vector<llama_token> tokenize(const std::string &text, bool add_special)
    {
        int n_tokens = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, true);
        if (n_tokens <= 0)
            return {};
        std::vector<llama_token> res(n_tokens);
        if (llama_tokenize(vocab, text.c_str(), text.size(), res.data(), res.size(), add_special, true) < 0)
        {
            throw runtime_error("Failed to tokenize");
        }
        return res;
    }

    void decode_tokens(const llama_token *tokens, size_t num_tokens)
    {
        if (n_pos + num_tokens > (size_t)n_ctx)
            throw runtime_error("Out of context");

        const int n_batch = llama_n_batch(ctx);
        for (size_t i = 0; i < num_tokens; i += n_batch)
        {
            const int chunk = std::min(n_batch, static_cast<int>(num_tokens - i));
            llama_batch batch = llama_batch_get_one(const_cast<llama_token *>(tokens) + i, chunk);
            if (llama_decode(ctx, batch))
                throw runtime_error("Failed to decode tokens");
            n_pos += chunk;
        }
    }

    void reset_ctx(const json &initial_prompts, bool thinking_in, int think_budget_in)
    {
        halt_flag.store(false);
        thinking = thinking_in;
        think_budget = think_budget_in < 0 ? (n_ctx / 8) : think_budget_in;
        chat_history = json::array();
        cached_tokens.clear();
        n_pos = 0;

        llama_memory_clear(llama_get_memory(ctx), true);

        vector<llama_chat_message> chat_msgs;
        bool inject_system = true;

        if (initial_prompts.is_array() && !initial_prompts.empty())
        {
            for (const auto &msg : initial_prompts)
            {
                const string &role = msg[0].get_ref<const string &>();
                const string &content = msg[1].get_ref<const string &>();
                if (role == "system")
                    inject_system = false;
                chat_history.push_back(json{role, content});
                chat_msgs.push_back({role.c_str(), content.c_str()});
            }
        }

        if (inject_system)
        {
            chat_history.push_back({"system", ""});
            chat_msgs.insert(chat_msgs.begin(), {"system", ""});
        }

        string formatted = apply_chat_template(chat_msgs, thinking, false);

        // Tokenize and decode initial state
        cached_tokens = tokenize(formatted, true);
        decode_tokens(cached_tokens.data(), cached_tokens.size());
    }
    vector<char> promptbuf;
    const char *prompt(const string &text)
    {
        // warning: returns pointers into promptbuf, so will be reset on next prompt() call.
        promptbuf.clear();

        // 1. Build the full message history to pass to minja
        chat_history.push_back(json{"user", text});
        vector<llama_chat_message> chat_msgs;
        for (const auto &msg : chat_history)
        {
            chat_msgs.push_back({msg[0].get_ref<const string &>().c_str(), msg[1].get_ref<const string &>().c_str()});
        }

        // 2. Format the entire sequence and let minja trim/strip whatever it wants
        string formatted = apply_chat_template(chat_msgs, thinking, true);

        // 3. Tokenize the entire template (must use add_special=true to match reset_ctx)
        vector<llama_token> new_tokens = tokenize(formatted, true);

        // 4. Token Diffing: Find the exact divergence point
        size_t min_len = std::min(cached_tokens.size(), new_tokens.size());
        auto mismatch_it = std::mismatch(cached_tokens.begin(), cached_tokens.begin() + min_len, new_tokens.begin());
        size_t match_len = std::distance(cached_tokens.begin(), mismatch_it.first);

        // 5. Truncate KV Cache if tags were stripped or altered
        if (match_len < cached_tokens.size())
        {
            // Sequence ID 0, slice from match_len to the end (-1)
            llama_memory_seq_rm(llama_get_memory(ctx), 0, match_len, -1);
            cached_tokens.resize(match_len);
            n_pos = match_len;
        }

        // 6. Decode only the new suffix
        if (match_len < new_tokens.size())
        {
            size_t delta_len = new_tokens.size() - match_len;
            decode_tokens(new_tokens.data() + match_len, delta_len);
            // Append evaluated tokens to cache
            cached_tokens.insert(cached_tokens.end(), new_tokens.begin() + match_len, new_tokens.end());
        }

        // 7. Clear halt flag before generation
        halt_flag.store(false);

        // --- Generation Loop ---
        bool eog_reached = false;
        bool in_thinking = thinking; // if thinking mode is on, we start in the thinking phase
        bool budget_injected = false;
        int think_tokens_generated = 0;
        char buf[128];
        while (n_pos < n_ctx)
        {
            if (halt_flag.load())
            {
                break;
            }
            llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id))
            {
                decode_tokens(&new_token_id, 1);
                cached_tokens.push_back(new_token_id); // Sync EOG to cache
                eog_reached = true;
                break;
            }

            // Track thinking phase: detect </think> token
            if (in_thinking && end_think_token >= 0 && new_token_id == end_think_token)
            {
                in_thinking = false;
            }

            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                size_t old_size = promptbuf.size();
                promptbuf.resize(old_size - n);
                llama_token_to_piece(vocab, new_token_id, promptbuf.data() + old_size, -n, 0, true);
            }
            else
            {
                promptbuf.insert(promptbuf.end(), buf, buf + n);
            }

            decode_tokens(&new_token_id, 1);
            cached_tokens.push_back(new_token_id); // Sync generated token to cache

            // Think budget enforcement
            if (in_thinking)
            {
                think_tokens_generated++;
                if (!budget_injected && think_budget > 0 && think_tokens_generated >= think_budget)
                {
                    budget_injected = true;
                    // Inject a soft nudge to wrap up thinking
                    const string nudge = "\n[Reasoning budget exceeded, let's wrap this up and produce the answer.]\n";
                    auto nudge_tokens = tokenize(nudge, false);
                    if (!nudge_tokens.empty())
                    {
                        decode_tokens(nudge_tokens.data(), nudge_tokens.size());
                        cached_tokens.insert(cached_tokens.end(), nudge_tokens.begin(), nudge_tokens.end());
                        // Also append to promptbuf so it appears in output
                        promptbuf.insert(promptbuf.end(), nudge.begin(), nudge.end());
                    }
                }
            }
        }

        if (!eog_reached)
        {
            throw runtime_error("Out of context");
        }

        promptbuf.push_back('\0');
        chat_history.push_back(json{"assistant", promptbuf.data()});

        return promptbuf.data();
    }

    json get_status() const
    {
        json j;
        j["history"] = chat_history;
        j["top_k"] = top_k;
        j["min_p"] = min_p;
        j["top_p"] = top_p;
        j["temp"] = temp;
        j["presence_penalty"] = presence_penalty;
        j["seed"] = seed;
        j["thinking"] = thinking;
        j["think_budget"] = think_budget;
        return j;
    }

    int token_count() const { return n_pos; }
};

llama_bridge_obj *llama_bridge_create(const char *model_path, const char *device_name)
{
    try
    {
        return new llama_bridge_obj(model_path, device_name);
    }
    catch (...)
    {
        return nullptr;
    }
}

const char *wrap_exception(const std::exception &e)
{
    thread_local string errbuf;
    errbuf = (json{{"err", e.what()}}).dump();
    return errbuf.c_str();
}

const char *llama_bridge_invoke(llama_bridge_obj *obj, const char *cmd)
{
    thread_local string invokebuf;
    if (!obj)
        return wrap_exception(runtime_error("Null object"));

    try
    {
        auto j = json::parse(cmd);
        if (!j.is_array() || j.size() != 2)
            throw runtime_error("Invalid cmd format");

        string method = j[0].get<string>();
        auto params = j[1];

        if (method == "prompt")
        {
            return obj->prompt(params.at("text").get<string>());
        }
        else if (method == "reset_ctx")
        {
            json initial;
            if (params.contains("initial_prompts") && !params["initial_prompts"].is_null())
            {
                initial = params["initial_prompts"];
            }
            obj->reset_ctx(initial, params.value("thinking", false), params.value("think_budget", -1));
            return nullptr;
        }
        else if (method == "set_sampler")
        {
            obj->set_sampler_params(params.value("temp", obj->temp), params.value("top_k", obj->top_k),
                                    params.value("min_p", obj->min_p), params.value("top_p", obj->top_p),
                                    params.value("presence_penalty", obj->presence_penalty),
                                    params.value("seed", obj->seed));
            return nullptr;
        }
        else if (method == "get_status")
        {
            invokebuf = obj->get_status().dump();
            return invokebuf.c_str();
        }
        else if (method == "token_count")
        {
            invokebuf = to_string(obj->token_count());
            return invokebuf.c_str();
        }
        else if (method == "halt_prompt")
        {
            obj->halt_flag.store(true);
            return nullptr;
        }

        return wrap_exception(runtime_error("Unknown method"));
    }
    catch (const std::exception &e)
    {
        return wrap_exception(e);
    }
}

void llama_bridge_destroy(llama_bridge_obj *obj)
{
    delete obj;
}

void llama_bridge_set_log_callback(llama_bridge_log_callback_fn cb, void *user_data)
{
    llama_log_set(reinterpret_cast<ggml_log_callback>(cb), user_data);
}