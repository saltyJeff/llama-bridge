/**
 * @file API file for llama bridge.
 * Should be invoked by a single thread only.
 *
 * STRING LIFETIME NOTE:
 *  each returned string (const char *)'s lifetime is
 *  only guaranteed until the next library call.
 */
#pragma once

#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef LLAMA_BRIDGE_EXPORTS
#define LLAMA_BRIDGE_API __declspec(dllexport)
#else
#define LLAMA_BRIDGE_API __declspec(dllimport)
#endif
#else
#if __GNUC__ >= 4
#define LLAMA_BRIDGE_API __attribute__((visibility("default")))
#else
#define LLAMA_BRIDGE_API
#endif
#endif

#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct llama_bridge_obj llama_bridge_obj;

    /**
     * Gets the available devices
     * @param backend_path A path to where the backend DLLs are, or null to search the default path
     * @return A JSON array describing the available devices.
     * Each element in the array is:
     * {
     *  "backend": string,
     *  "desc": string,
     *  "name": string,
     *  "total_mem": number,
     *  "free_mem": number
     * }
     * If an error, the return will be a JSON string of {"err": string}
     * See STRING RETURN NOTE
     */
    LLAMA_BRIDGE_API const char *llama_bridge_get_devices(const char *backend_path);

    /**
     * Creates a new llama bridge instance on a given device.
     * NOTE: The llama bridge loads as many layers as possible onto the specified device.
     * @param model_path The path to the model file
     * @param device The name of the device to use
     *  Should be one of the names of the devices returned by llama_bridge_get_devices
     * @return A pointer to the Llama session. The pointer must be freed using
     *      llama_bridge_destroy(). Returns nullptr if the session cannot be created.
     */
    LLAMA_BRIDGE_API llama_bridge_obj *llama_bridge_create(const char *model_path, const char *device_name);

    /**
     * Invokes a command on the llama bridge instance.
     * @param obj the llama bridge instance
     * @param cmd a 2-element array of command, args, which is one of the following:
     *   ["prompt", {"text": string}]; returns string on success
     *      - Each prompt() call retains multi-turn context from prior prompt() calls
     *        within the same session. Call reset_ctx to start fresh.
     *   ["reset_ctx", {"initial_prompts": msg[]?, "thinking": bool?, "think_budget": int?}]; returns nullptr on success
     *      - Clears context and chat history. Optionally seeds with initial_prompts.
     *      - Optionally set the model into thinking mode
     *      - think_budget: max tokens allowed for the thinking phase before a soft
     *        "wrap it up" nudge is injected. Defaults to n_ctx/8. Ignored if thinking is false.
     *      - Model messages in initial_prompts have thinking blocks stripped.
     *   ["set_sampler", {"temp": double?, "top_k": int?, "min_p": double?, "top_p": double?, "presence_penalty": double?, "seed": int?}]; returns
     * nullptr on success
     *   ["get_status", {}]; returns {
     *      "history": msg[],
     *      "top_k": number,
     *      "min_p": number,
     *      "top_p": number,
     *      "temp": number,
     *      "presence_penalty": number,
     *      "seed": number,
     *      "thinking": bool,
     *      "think_budget": number
     *   } on success
     *   ["token_count", {}]; returns number on success.
     *     - This method is thread safe and can be called from another thread.
     *     - Use to monitor generation progress during a long generation session (it should keep increasing).
     *   ["halt_prompt", {}]; returns nullptr on success.
     *     - Halts an ongoing prompt() call. Thread safe.
     *
     *   msg is a 2-element array of ["role" (user|assistant|system), string]
     *   question mark means optional field
     */
    LLAMA_BRIDGE_API const char *llama_bridge_invoke(llama_bridge_obj *obj, const char *cmd);

    /**
     * Destroys a llama bridge object
     * @param obj A pointer to the object
     */
    LLAMA_BRIDGE_API void llama_bridge_destroy(llama_bridge_obj *obj);

    /**
     * Overrides the llama log callback
     * @param cb the callback function.
     * @param user_data a user data pointer that will be passed to the callback. Can be null.
     *
     * As of this moment, these are the level values:
     * enum ggml_log_level {
        GGML_LOG_LEVEL_NONE  = 0,
        GGML_LOG_LEVEL_DEBUG = 1,
        GGML_LOG_LEVEL_INFO  = 2,
        GGML_LOG_LEVEL_WARN  = 3,
        GGML_LOG_LEVEL_ERROR = 4,
        GGML_LOG_LEVEL_CONT  = 5, // continue previous log
    };
     */
    typedef void (*llama_bridge_log_callback_fn)(int level, const char *text, void *user_data);
    LLAMA_BRIDGE_API void llama_bridge_set_log_callback(llama_bridge_log_callback_fn cb, void *user_data);
#ifdef __cplusplus
}
#endif