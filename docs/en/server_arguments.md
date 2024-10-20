# Server Arguments

All the arguments can be found in the [`server_args.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py).

## Model and Tokenizer Configuration

### `model_path`: str
- **Description**: The path to the model weights. It's required.
- **Usage**: Can be a local folder or a Hugging Face repo ID.

### `tokenizer_path`: Optional[str]
- **Description**: The path to the tokenizer.
- **Note**: It defaults to the same value as `model_path` if not provided.

### `tokenizer_mode`: str
- **Description**: Specifies [the tokenizer mode](https://huggingface.co/learn/nlp-course/chapter6/3).
- **Default**: "auto"
- **Options**: 
  - "auto": Uses the fast tokenizer if available
  - "slow": Always use the slow tokenizer

### `skip_tokenizer_init`: bool
- **Description**: If set, skips tokenizer initialization.
- **Default**: False
- **Usage**: When True, `input_ids` should be passed directly when sending requests. Otherwise, pass in the full prompt directly.

### `load_format`: str

- **Description**: The format of the model weights to load.
- **Default**: "auto"
- **Options**:
  - "auto": Tries safetensors, falls back to pytorch bin
  - "pt": PyTorch bin format
  - "safetensors": SafeTensors format
  - "npcache": PyTorch format with numpy cache for faster loading
  - "dummy": Initializes weights with random values (for profiling)

### `dtype`: str
- **Description**: Data type for model weights and activations.
- **Default**: "auto"
- **Options**:
  - "auto": FP16 for FP32/FP16 models, BF16 for BF16 models
  - "half"/"float16": FP16 precision (recommended for AWQ quantization)
  - "bfloat16": Balanced precision and range
  - "float"/"float32": FP32 precision
- **Note**: [TODO]

### `kv_cache_dtype`: str

- **Description**: Data type for KV cache storage.
- **Default**: "auto"
- **Options**:
  - "auto": Uses model data type
  - "fp8_e5m2": Supported for CUDA 11.8+

### `trust_remote_code`: bool

- **Description**: Allows custom tokenizer defined in our own path. In the original HuggingFace implementation, if you clone a model to your local device, take [openbmb/MiniCPM3-4B](https://huggingface.co/openbmb/MiniCPM3-4B) as an example, it usually contains the [modeling file](https://huggingface.co/openbmb/MiniCPM3-4B/blob/main/modeling_minicpm.py) and [tokenization file](https://huggingface.co/openbmb/MiniCPM3-4B/blob/main/tokenization_minicpm.py). If you want to adjust the local model's behavior. You should change these files and set  `trust_remote_code` to True. The "remote" here means that, for the HuggingFace hub, your local files are the "remote". However, in SGLang, we already pre-defined the modeling files in the [models' file](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models), so if you want to adjust local model behavior, you should change the model class in the [models' file](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/models). In other words, for SGLang the `trust_remote_code=True` is useful only when you change the tokenization in your local path.
- **Default**: True
- **Note**: Set to False for "Alibaba-NLP/gte-Qwen2-1.5B-instruct" model due to tokenizer issues.

### `context_length`: Optional[int]
- **Description**: The model's maximum context length.
- **Default**: None (uses the value from the model's `config.json`)
- **Note**: If you pass in an extremely long prompt to the engine, the prompt won't be truncated, instead, this request will fail and return an error code of 400, indicating that the context is out of length. Also, do not set the `context_length` larger than the default configuration, since if you do not do the [ROPE extension](https://blog.eleuther.ai/yarn/), the model will respond with meaningless contents after the input prompt passes the length of the default configuration.

### `quantization`: Optional[str]
- **Description**: The quantization method to use.
- **Default**: None
- **Options**: "awq", "fp8", "gptq", "marlin", "gptq_marlin", "awq_marlin", "squeezellm", "bitsandbytes"

### `served_model_name`: Optional[str]
- **Description**: Overrides the model name returned by the v1/models endpoint in OpenAI API server.
- **Default**: None (uses `model_path` if not specified)
- **Note**: For the `model` arguments while sending requests to the server, if you do not set `served_model_name` when you launch the server, it should be the same as the `model_path`. Otherwise, it's the `served_model_name`.

### `chat_template`: Optional[str]
- **Description**: The built-in chat template name or path to a chat template file.
- **Default**: None
- **Usage**: Used only for OpenAI-compatible API servers.
- **Note**: Follow [this document](https://sglang.readthedocs.io/en/latest/custom_chat_template.html) to set your Customized Chat Template.

### `is_embedding`: bool

- **Description**: Whether to use the model as an embedding model.
- **Default**: False
- **Note**: For the "Alibaba-NLP/gte-Qwen2-1.5B-instruct" model, it can be used both for casual completions and to generate embeddings. So setting this parameter is required when you want to use your model as an embedding model.

## Port Configuration

### `host`: str
- **Description**: The host address on which the server will run.
- **Default**: "127.0.0.1"
- **Usage**: Specifies the IP address or hostname for the server to bind to.
- **Note**: Using host `0.0.0.0` to enable external access, i.e., allowing other clients from the same subnet to send requests to your server’s IP address. Otherwise, the client and server should be on the same device and send requests only to the local host / 127.0.0.1.

### `port`: int

- **Description**: The primary port number on which the server will listen.
- **Default**: 30000
- **Usage**: Defines the main port for incoming connections to the server.

### `additional_ports`: Optional[Union[List[int], int]]

- **Description**: Additional port(s) that can be used by the server.
- **Default**: None
- **Usage**: Can be either a single integer or a list of integers representing extra ports.
- **Note**: If a single integer is provided, it will be converted to a list containing that integer. If None is provided, it will be treated as an empty list.

## Memory Management

# Server Configuration Documentation

## Port Configuration

### `host`: str
- **Description**: The host address on which the server will run.
- **Default**: "127.0.0.1"
- **Usage**: Specifies the IP address or hostname for the server to bind to.
- **Note**: Use host `0.0.0.0` to enable external access, allowing other clients from the same subnet to send requests to your server's IP address. Otherwise, the client and server should be on the same device and send requests only to the local host / 127.0.0.1.

### `port`: int
- **Description**: The primary port number on which the server will listen.
- **Default**: 30000
- **Usage**: Defines the main port for incoming connections to the server.

### `additional_ports`: Optional[Union[List[int], int]]
- **Description**: Additional port(s) that can be used by the server.
- **Default**: None
- **Usage**: Can be either a single integer or a list of integers representing extra ports.
- **Note**: If a single integer is provided, it will be converted to a list containing that integer. If None is provided, it will be treated as an empty list.

## Memory Management

### `mem_fraction_static`: Optional[float]

- **Description**: The fraction of memory used for static allocation, i.e., model weights, cached tokens, and running requests.
- **Default**: None (automatically set based on tensor parallelism size)
- **Usage**: Determines the proportion of GPU memory allocated for static elements.
- **Note**: 
  1. If not set, it's automatically determined based on the tensor parallelism size:
     - 0.79 for tp_size >= 16
     - 0.83 for tp_size >= 8
     - 0.85 for tp_size >= 4
     - 0.87 for tp_size >= 2
     - 0.88 for tp_size < 2
  2. SGLang does not preallocate a fixed-size memory pool as a cache. Instead, cached tokens and currently running requests share the same memory pool. The system dynamically allocates memory for cache and running requests. When enough waiting requests run, the system will evict all cached tokens in favor of a larger batch size. Thus, increasing the cache hit rate (the length of the shared prefix) would result in better latency and serving batch size.

### `max_running_requests`: Optional[int]
- **Description**: The maximum number of requests that can be processed simultaneously.
- **Default**: None
- **Usage**: Limits the number of concurrent requests the server can handle.

### `max_num_reqs`: Optional[int]
- **Description**: The maximum number of requests to serve in the memory pool.
- **Default**: None
- **Usage**: Sets an upper limit on the number of requests that can be held in memory.
- **Note**: For models with large context lengths, you may need to decrease this value to avoid out-of-memory errors.

### `max_total_tokens`: Optional[int]
- **Description**: The maximum number of tokens allowed in the memory pool.
- **Default**: None
- **Usage**: Limits the total number of tokens across all requests in the memory pool.
- **Note**: If not specified, it will be automatically calculated based on the memory usage fraction. This option is typically used for development and debugging purposes.

## Prefill and Scheduling

### `chunked_prefill_size`: int
- **Description**: The maximum number of tokens in a chunk for chunked prefill.
- **Default**: 8192
- **Usage**: Determines the size of chunks when using chunked prefill for long contexts.
- **Note**:
  1. Setting this to -1 or a value <= 0 disables chunked prefill.
  2. The chunk size should be a moderate value since short chunks allow to piggyback more decode requests but decrease the GPU utilization for the prefill requests. You can check the [original paper](https://arxiv.org/abs/2308.16369) on chunked prefill for more insights.

### `max_prefill_tokens`: int
- **Description**: The maximum number of tokens in a prefill batch.
- **Default**: 16384
- **Usage**: Sets an upper limit on the number of tokens processed in a single prefill operation.
- **Note**: The actual limit will be the maximum of this value and the model's maximum context length.

### `schedule_policy`: str
- **Description**: The scheduling policy for request processing.
- **Default**: "lpm"
- **Options**: "lpm", "random", "fcfs", "dfs-weight"
- **Usage**: Determines how requests are prioritized and scheduled for processing. "lpm" stands for "longest-shared-prefix-first", "fcfs" stands for "first-come-first-serve", and "dfs-weight" stands for "deep-first-search".

### `schedule_conservativeness`: float
- **Description**: How conservative the scheduling policy should be.
- **Default**: 1.0
- **Usage**: A larger value means more conservative scheduling.
- **Note**: Use a larger value if you see requests being retracted frequently.

## Other Runtime Options

### `tp_size`: int

- **Description**: The tensor parallelism size.
- **Default**: 1
- **Usage**: Specifies the number of GPUs to use for tensor parallelism.
- **Note**: Increasing this value can help with larger models that don't fit on a single GPU. Note that tensor parallelism requires extensive communication between GPUs, so if the model can fit on a single GPU and your server does not have a quick GPU connection like NVLink, please keep `tp_size` to 1.

### `stream_interval`: int

- **Description**: The interval (or buffer size) for streaming in terms of the token length.
- **Default**: 1
- **Usage**: Controls the frequency of streaming outputs.
- **Note**: A smaller value makes streaming smoother, while a larger value increases throughput.

### `random_seed`: Optional[int]

- **Description**: The random seed for reproducibility.
- **Default**: None
- **Usage**: Sets a fixed random seed for deterministic behavior.
- **Note**: If not set, a random seed will be generated.

## Logging Options

### `log_level`: str

- **Description**: The logging level for all loggers.
- **Default**: "info"
- **Usage**: Controls the verbosity of log messages.
- **Options**: Typically includes "debug", "info", "warning", "error", and "critical".

### `log_level_http`: Optional[str]

- **Description**: The logging level, specifically for the HTTP server.
- **Default**: None
- **Usage**: If set, overrides the general log_level for HTTP-related logs.
- **Note**: If not set, it will use the value from `log_level`.

### `log_requests`: bool

- **Description**: Whether to log the inputs and outputs of all requests.
- **Default**: False
- **Usage**: When True, logs detailed information about each request and response.
- **Note**: Enabling this can be useful for debugging but may impact performance and generate large log files.

### `show_time_cost`: bool

- **Description**: Whether to show the time cost of custom marks.
- **Default**: False
- **Usage**: When True, displays timing information for various operations.
- **Note**: Useful for performance analysis and optimization.

## Other Configuration Options

### `api_key`: Optional[str]

- **Description**: API key for server authentication.
- **Default**: None
- **Usage**: If set, clients must provide this key to access the server. Note that even if you do not set the `api_key` when launching the server, you still need to provide an arbitrary API key when sending requests.

### `file_storage_pth`: str

- **Description**: Path for file storage used by the server.
- **Default**: "SGLang_storage"
- **Usage**: Specifies the directory where the server stores files.

## Data Parallelism

### `dp_size`: int

- **Description**: The data parallelism size.
- **Default**: 1
- **Usage**: Specifies the number of data parallel workers.

### `load_balance_method`: str

- **Description**: Method for load balancing in data parallel setup.
- **Default**: "round_robin"
- **Usage**: Determines how requests are distributed among data parallel workers.

## Distributed Arguments

### `nccl_init_addr`: Optional[str]

- **Description**: NCCL initialization address for multi-node setups.
- **Default**: None
- **Usage**: Specifies the address for NCCL initialization in distributed environments.

### `nnodes`: int

- **Description**: Number of nodes in a distributed setup.
- **Default**: 1
- **Usage**: Specifies the total number of nodes in a multi-node configuration.

### `node_rank`: Optional[int]

- **Description**: Rank of the current node in a distributed setup.
- **Default**: None
- **Usage**: Identifies the current node's position in a multi-node configuration.

## Model Override Arguments

### `json_model_override_args`: str

- **Description**: JSON-formatted string for overriding model arguments.
- **Default**: "{}"
- **Usage**: Allows dynamic overriding of model parameters at runtime.

## Optimization and Debug Options

### `disable_flashinfer`: bool

- **Description**: Disables FlashInfer attention kernels.
- **Default**: False
- **Usage**: Set to True to use standard attention mechanisms instead of FlashInfer.

### `disable_flashinfer_sampling`: bool

- **Description**: Disables FlashInfer sampling kernels.
- **Default**: False
- **Usage**: Set to True to use standard sampling methods instead of FlashInfer.

### `disable_radix_cache`: bool

- **Description**: Disables RadixAttention for prefix caching.
- **Default**: False
- **Usage**: Set to True to disable RadixAttention optimization.

### `disable_regex_jump_forward`: bool

- **Description**: Disables regex-based jump-forward optimization.
- **Default**: False
- **Usage**: Set to True to disable regex optimizations in text processing.

### `disable_cuda_graph`: bool

- **Description**: Disables CUDA graph optimization.
- **Default**: False
- **Usage**: Set to True to disable CUDA graph usage for potential debugging.

### `disable_cuda_graph_padding`: bool

- **Description**: Disables CUDA graph padding.
- **Default**: False
- **Usage**: Set to True to disable padding in CUDA graphs.

### `disable_disk_cache`: bool

- **Description**: Disables disk caching.
- **Default**: False
- **Usage**: Set to True to prevent disk caching, which may help with filesystem-related issues.

### `disable_custom_all_reduce`: bool

- **Description**: Disables custom all-reduce operations.
- **Default**: False
- **Usage**: Set to True to use standard all-reduce instead of custom implementations.

### `enable_mixed_chunk`: bool

- **Description**: Enables mixed chunk processing.
- **Default**: False
- **Usage**: Set to True to allow mixing of prefilling and decode operations in a single batch.

### `enable_torch_compile`: bool

- **Description**: Enables PyTorch compilation optimization.
- **Default**: False
- **Usage**: Set to True to use torch.compile for potential performance improvements.

### `torchao_config`: str

- **Description**: Configuration string for TorchAO (Ahead-of-Time) compilation.
- **Default**: ""
- **Usage**: Specifies TorchAO compilation settings if enabled.

### `enable_p2p_check`: bool

- **Description**: Enables peer-to-peer (P2P) access checks for GPUs.
- **Default**: False
- **Usage**: Set to True to explicitly check and enable P2P access between GPUs.

### `enable_mla`: bool

- **Description**: Enables Multi-head Latent Attention (MLA).
- **Default**: False
- **Usage**: Set to True to use MLA, which can improve performance for certain models.

### `triton_attention_reduce_in_fp32`: bool

- **Description**: Forces attention reduction in FP32 for Triton attention kernels.
- **Default**: False
- **Usage**: Set to True to potentially improve stability at the cost of some performance.