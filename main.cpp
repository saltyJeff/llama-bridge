#include <nlohmann/json.hpp>
#include "llama_bridge.h"
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#pragma execution_character_set("utf-8")

#ifdef _WIN32
#include <windows.h>
#endif

#ifndef MODEL_LOCATION
#define MODEL_LOCATION "Qwen3.5-4B-UD-Q4_K_XL.gguf"
#endif

using namespace nlohmann;
using namespace std;

void print_separator(const string &title)
{
    cout << "\n" << string(60, '=') << "\n";
    cout << " TEST: " << title << "\n";
    cout << string(60, '=') << endl;
}

void print_json(const char *json_str)
{
    if (!json_str)
    {
        cout << "(null)" << endl;
        return;
    }
    try
    {
        auto j = json::parse(json_str);
        cout << j.dump(4) << endl;
    }
    catch (...)
    {
        cout << "RAW: " << json_str << endl;
    }
}

void log_cb(int level, const char *msg, void *user_data)
{
    (void)level;
    (void)user_data;
    if (level < 3)
        return;
    cout << "[LLAMA_BRIDGE LOG] " << msg << endl;
}

void run_test(llama_bridge_obj *obj, const string &test_name, const json &command)
{
    print_separator(test_name);
    cout << "Command: " << command.dump() << endl;
    const char *resp = llama_bridge_invoke(obj, command.dump().c_str());
    cout << "Result: " << endl;
    print_json(resp);
}

int main()
{
#ifdef _WIN32
    SetConsoleOutputCP(65001); // CP_UTF8
#endif

    llama_bridge_set_log_callback(log_cb, nullptr);

    print_separator("Device Discovery");
    const char *devices_json = llama_bridge_get_devices(nullptr);
    print_json(devices_json);

    auto devices = json::parse(devices_json);
    string device_name = "CPU"; // Default
    if (devices.is_array() && !devices.empty())
    {
        // Prefer CUDA if available
        for (const auto &dev : devices)
        {
            if (dev["name"].get<string>().find("CUDA") != string::npos)
            {
                device_name = dev["name"];
                break;
            }
        }
        if (device_name == "CPU")
        {
            device_name = devices[0]["name"];
        }
    }
    cout << "Selected device: " << device_name << endl;

    print_separator("Initialization");
    const char *model_path = MODEL_LOCATION;
    cout << "Loading model: " << model_path << " on " << device_name << "..." << endl;
    auto obj = llama_bridge_create(model_path, device_name.c_str());
    if (!obj)
    {
        cerr << "Failed to create llama_bridge_obj!" << endl;
        return 1;
    }
    cout << "Success!" << endl;

    // Run the automated test suite
    run_test(obj, "Initial Status", {"get_status", json::object()});

    run_test(obj, "Sampler Configuration", {"set_sampler", {{"temp", 0.7}, {"top_k", 40}, {"seed", 42}}});

    run_test(obj, "Updated Status", {"get_status", json::object()});

    run_test(obj, "Prompt (No Thinking)", {"prompt", {{"text", "Who are you?"}}});

    run_test(obj, "Context Reset with Thinking Mode ENABLED", {"reset_ctx", {{"thinking", true}}});

    run_test(obj, "Prompt (With Thinking)", {"prompt", {{"text", "Explain quantum entanglement in one sentence."}}});

    run_test(obj, "Context Reset with Initial Prompts",
             {"reset_ctx",
              {{"initial_prompts",
                json::array({
                    json::array({"system", "You are a helpful assistant that speaks like a pirate."}),
                    json::array({"user", "Ahoy!"}),
                    json::array({"assistant", "<think>\nThinking hard about a pirate greeting.\n</think>\nAhoy there, matey! What can I do for ye today?"})
                })
              }}});

    run_test(obj, "Status after reset (verify thoughts)", {"get_status", json::object()});

    run_test(obj, "Prompt (After reset)", {"prompt", {{"text", "What is the secret to a happy life?"}}});

    run_test(obj, "Error Handling", {"invalid_method", json::object()});

    run_test(obj, "Token Count", {"token_count", json::object()});

    // Start Interactive REPL Loop
    cout << "\n" << string(60, '=') << "\n";
    cout << " INTERACTIVE REPL MODE\n";
    cout << string(60, '=') << endl;
    cout << "Enter commands to interact with the model:\n";
    cout << "  - Type a message to prompt the model.\n";
    cout << "  - Type '/reset_ctx <json_payload>' to reset context.\n";
    cout << "    (e.g., /reset_ctx {\"thinking\": true})\n";
    cout << "  - Type '/get_status' to query the current session status.\n";
    cout << "  - Type '/exit' or '/quit' to exit the driver.\n";
    cout << "------------------------------------------------------------\n";

    // Spawn background thread to print token count every 10 seconds
    bool running = true;
    json token_cmd = json::array({"token_count", json::object()});
    thread monitor_thread([&]() {
        while (running) {
            this_thread::sleep_for(chrono::seconds(5));
            if (!running)
                break;
            const char *count = llama_bridge_invoke(obj, token_cmd.dump().c_str());
            if (count) {
                cerr << "\n[Token Count: " << count << "]" << endl << "> " << flush;
            }
        }
    });

    string line;
    while (true)
    {
        cout << "\n> " << flush;
        if (!getline(cin, line))
        {
            break;
        }

        if (line.empty())
        {
            continue;
        }

        if (line == "/quit" || line == "/exit")
        {
            break;
        }
        else if (line == "/get_status")
        {
            json cmd = {"get_status", json::object()};
            const char *resp = llama_bridge_invoke(obj, cmd.dump().c_str());
            print_json(resp);
        }
        else if (line.rfind("/reset_ctx", 0) == 0)
        {
            string payload = line.substr(10);
            // Trim leading/trailing spaces
            size_t first = payload.find_first_not_of(" \t");
            if (first == string::npos)
            {
                payload = "{}";
            }
            else
            {
                payload = payload.substr(first);
            }

            try
            {
                json args = json::parse(payload);
                json cmd = {"reset_ctx", args};
                cout << "Resetting context with args: " << args.dump() << "..." << endl;
                const char *resp = llama_bridge_invoke(obj, cmd.dump().c_str());
                print_json(resp);
            }
            catch (const exception &e)
            {
                cerr << "Error parsing JSON payload: " << e.what() << endl;
            }
        }
        else if (line[0] == '/')
        {
            cout << "Unknown command. Available commands: /get_status, /reset_ctx <json>, /quit, /exit" << endl;
        }
        else
        {
            json cmd = {"prompt", {{"text", line}}};
            const char *resp = llama_bridge_invoke(obj, cmd.dump().c_str());
            if (resp)
            {
                cout << resp << endl;
            }
            else
            {
                cout << "(null response)" << endl;
            }
        }
    }

    running = false;
    if (monitor_thread.joinable())
        monitor_thread.join();

    print_separator("Cleanup");
    cout << "Destroying bridge object..." << endl;
    llama_bridge_destroy(obj);
    cout << "Done." << endl;

    return 0;
}