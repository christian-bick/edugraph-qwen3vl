FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ADD https://huggingface.co/christian-bick/Qwen3-VL-4B-EduGraph-Q4_K_M-GGUF/resolve/main/qwen3-vl-4b-edugraph-q4_k_m.gguf /model.gguf
ADD https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF/resolve/main/mmproj-Qwen3VL-4B-Instruct-F16.gguf /mmproj.gguf

# 3. Set Environment Variables so we don't need complex CLI flags
ENV LLAMA_ARG_MODEL=/model.gguf
ENV LLAMA_ARG_MMPROJ=/mmproj.gguf
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080
ENV LLAMA_ARG_N_GPU_LAYERS=999
ENV LLAMA_ARG_CTX_SIZE=16384
ENV LLAMA_ARG_FLASH_ATTN=on

# Optimized parameters for deterministic classification
ENV LLAMA_ARG_TEMP=0.0
ENV LLAMA_ARG_TOP_K=1
ENV LLAMA_ARG_TOP_P=1.0
ENV LLAMA_ARG_MIN_P=0.0
ENV LLAMA_ARG_N_PREDICT=1024
