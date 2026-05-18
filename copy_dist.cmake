# Get all ggml backend DLLs/SOs
file(GLOB BACKEND_FILES "${BIN_DIR}/ggml-*.dll" "${BIN_DIR}/libggml-*.so" "${BIN_DIR}/ggml-*.so" "${BIN_DIR}/llama.dll" "${BIN_DIR}/libllama.so" "${BIN_DIR}/ggml.dll" "${BIN_DIR}/libggml.so")
if(BACKEND_FILES)
    file(COPY ${BACKEND_FILES} DESTINATION "${DIST_DIR}")
endif()

# If CUDA_BIN_DIR is provided, copy CUDA DLLs
if(CUDA_BIN_DIR)
    file(GLOB CUDA_DLLS 
        "${CUDA_BIN_DIR}/cudart64_*.dll"
        "${CUDA_BIN_DIR}/cublas64_*.dll"
        "${CUDA_BIN_DIR}/cublasLt64_*.dll"
        "${CUDA_BIN_DIR}/curand64_*.dll"
        "${CUDA_BIN_DIR}/cusolver64_*.dll"
        "${CUDA_BIN_DIR}/cufft64_*.dll"
        # Newer CUDA toolkits put DLLs in bin/x64
        "${CUDA_BIN_DIR}/x64/cudart64_*.dll"
        "${CUDA_BIN_DIR}/x64/cublas64_*.dll"
        "${CUDA_BIN_DIR}/x64/cublasLt64_*.dll"
        "${CUDA_BIN_DIR}/x64/curand64_*.dll"
        "${CUDA_BIN_DIR}/x64/cusolver64_*.dll"
        "${CUDA_BIN_DIR}/x64/cufft64_*.dll"
    )
    if(CUDA_DLLS)
        file(COPY ${CUDA_DLLS} DESTINATION "${DIST_DIR}")
    endif()
endif()
