# llama-bridge

This is a C wrapper around llama.cpp that turns it into a shared library with a
simple C API. It is designed for Qwen 3.5 4B models (tested with
`Qwen3.5-4B-UD-Q4_K_XL.gguf`). The library is meant to be consumed from a Dart
isolate via `dart:ffi`.

## File structure

- [`llama_bridge.h`](llama_bridge.h) - the public C API. read this to see all
  functions and their documentation.
- [`llama_bridge.cpp`](llama_bridge.cpp) - implementation of the library.
- [`main.cpp`](main.cpp) - a test driver that exercises all API functions and
  provides an interactive REPL.
- [`minja.hpp`](minja.hpp) - a Jinja-compatible template engine (vendored from
  Google/minja) used for chat templates.
- [`nlohmann/`](nlohmann/) - vendored `nlohmann/json.hpp`.
- [`CMakeLists.txt`](CMakeLists.txt) - build script.
- [`copy_dist.cmake`](copy_dist.cmake) - helper to copy DLLs into the dist
  folder after build.
- [`.clang-format`](.clang-format) - code formatting rules.
- [`.clangd`](.clangd) - LSP configuration for clangd.

## Requirements

- CMake 3.21+
- A C++17 compiler (MSVC, GCC, Clang)
- `llama.cpp` is fetched automatically by CMake from GitHub tag `b9209`

### Windows

- Visual Studio 2022 with "Desktop development with C++" workload
- For CUDA: NVIDIA CUDA Toolkit
- For Vulkan: Vulkan SDK (set `VULKAN_SDK` environment variable)

### macOS

- Xcode Command Line Tools
- For Metal: macOS 12+ (automatic with Apple Clang)

### Linux

- `g++` or `clang++`, `cmake`, `make`
- For CUDA: NVIDIA CUDA Toolkit
- For Vulkan: `vulkan-headers`, `vulkan-loader`

## Build instructions

### Windows (Visual Studio)

Open the folder in Visual Studio and select `x64-Debug` or `x64-Release` from
the configuration dropdown. Build the `llama_bridge_driver` target. Output goes
to `out/build/x64-Debug/dist/` (or `x64-Release`).

### Windows (command line with Ninja)

```
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama_bridge_driver
```

### macOS/Linux

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama_bridge_driver
```

## Output

After a successful build the `dist` folder contains:

- `llama_bridge.dll` / `llama_bridge.so` / `llama_bridge.dylib` (the shared
  library you link against)
- `llama_bridge_driver` (the test driver executable)
- `llama_bridge.h` (copy of the header for your FFI bindings)
- backend DLLs/SOs for CUDA, Vulkan, CPU

## LSP setup

To get clangd LSP working, first generate `compile_commands.json`:

```
cmake -B build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

Then copy or symlink `build/compile_commands.json` to the project root.

## Model file

The model file (`Qwen3.5-4B-UD-Q4_K_XL.gguf`) must be in the project root or
you can change `MODEL_LOCATION` in [`CMakeLists.txt`](CMakeLists.txt) to point
elsewhere.

## API reference

Read [`llama_bridge.h`](llama_bridge.h) for the full API reference. All
functions are `extern "C"` and return JSON strings or null. Each returned
string is valid only until the next library call (thread-local storage).