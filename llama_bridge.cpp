#include "llama_bridge.h"
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
    thread_local string strbuf;
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
    strbuf = ret_json.dump();
    return strbuf.c_str();
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

    int32_t top_k = 64;
    double top_p = 0.95;
    double min_p = -1;
    double temp = 1.0;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    bool thinking = false;

    vector<char> strbuf;
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
        ctx_params.type_k = GGML_TYPE_Q4_0;
        ctx_params.type_v = GGML_TYPE_Q4_0;

        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx)
        {
            llama_model_free(model);
            delete[] devices;
            throw runtime_error("Unable to create context");
        }

        n_ctx = llama_n_ctx(ctx);
        set_sampler_params(temp, top_k, min_p, top_p, seed);
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

    void set_sampler_params(double temp_in, int32_t top_k_in, double min_p_in, double top_p_in, int64_t seed_in)
    {
        temp = temp_in;
        top_k = top_k_in;
        min_p = min_p_in;
        top_p = top_p_in;
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

    void reset_ctx(const json &initial_prompts, bool thinking_in)
    {
        thinking = thinking_in;
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
            chat_msgs.insert(chat_msgs.begin(), {"system", ""});
        }

        string formatted = apply_chat_template(chat_msgs, thinking, false);

        // Tokenize and decode initial state
        cached_tokens = tokenize(formatted, true);
        decode_tokens(cached_tokens.data(), cached_tokens.size());
    }

    const char *prompt(const string &text)
    {
        strbuf.clear();

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

        // --- Generation Loop ---
        bool eog_reached = false;
        char buf[128];
        while (n_pos < n_ctx)
        {
            llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);

            if (llama_vocab_is_eog(vocab, new_token_id))
            {
                decode_tokens(&new_token_id, 1);
                cached_tokens.push_back(new_token_id); // Sync EOG to cache
                eog_reached = true;
                break;
            }

            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0)
            {
                size_t old_size = strbuf.size();
                strbuf.resize(old_size - n);
                llama_token_to_piece(vocab, new_token_id, strbuf.data() + old_size, -n, 0, true);
            }
            else
            {
                strbuf.insert(strbuf.end(), buf, buf + n);
            }

            decode_tokens(&new_token_id, 1);
            cached_tokens.push_back(new_token_id); // Sync generated token to cache
        }

        if (!eog_reached)
        {
            throw runtime_error("Out of context");
        }

        strbuf.push_back('\0');
        chat_history.push_back(json{"assistant", strbuf.data()});

        return strbuf.data();
    }

    const char *get_status()
    {
        json j;
        j["history"] = chat_history;
        j["top_k"] = top_k;
        j["min_p"] = min_p;
        j["top_p"] = top_p;
        j["temp"] = temp;
        j["seed"] = seed;
        j["thinking"] = thinking;
        string status = j.dump();
        strbuf.assign(status.begin(), status.end());
        strbuf.push_back('\0');
        return strbuf.data();
    }
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
            obj->reset_ctx(initial, params.value("thinking", false));
            return nullptr;
        }
        else if (method == "set_sampler")
        {
            obj->set_sampler_params(params.value("temp", obj->temp), params.value("top_k", obj->top_k),
                                    params.value("min_p", obj->min_p), params.value("top_p", obj->top_p),
                                    params.value("seed", obj->seed));
            return nullptr;
        }
        else if (method == "get_status")
        {
            return obj->get_status();
        }

        return wrap_exception(runtime_error("Unknown method"));
    }
    catch (const std::exception &e)
    {
        return wrap_exception(e);
    }
}

void llama_bridge_destroy(llama_bridge_obj *obj)
{ delete obj; }

void llama_bridge_set_log_callback(llama_bridge_log_callback_fn cb, void *user_data)
{ llama_log_set(reinterpret_cast<ggml_log_callback>(cb), user_data); }