SGLang Documentation
====================================

Welcome to SGLang!

SGLang is a fast serving framework for large language models and vision language models. It makes your interaction with models faster and more controllable by co-designing the backend runtime and frontend language.

The core features include:


* **Fast Backend Runtime**: Efficient serving with RadixAttention for prefix caching, jump-forward constrained decoding, continuous batching, token attention (paged attention), tensor parallelism, flashinfer kernels, and quantization (AWQ/FP8/GPTQ/Marlin).

* **Flexible Frontend Language**: Enables easy programming of LLM applications with chained generation calls, advanced prompting, control flow, multiple modalities, parallelism, and external interactions.

* **Extensive Model Support**: SGLang supports a wide range of generative models including the Llama series (up to Llama 3.1), Mistral, Gemma, Qwen, DeepSeek, LLaVA, Yi-VL, StableLM, Command-R, DBRX, Grok, ChatGLM, InternLM 2 and Exaone 3. It also supports embedding models such as e5-mistral and gte-Qwen2. Easily extensible to support new models.

* **Open Source Community**: SGLang is an open source project with a vibrant community of contributors. We welcome contributions from anyone interested in advancing the state of the art in LLM and VLM serving.

Documentation
-------------

**In this documentation, we'll dive into these following areas to help you get the most out of SGLang.**

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   install.md
   backend.md
   frontend.md

.. toctree::
   :maxdepth: 1
   :caption: References

   sampling_params.md
   hyperparameter_tuning.md
   model_support.md
   contributor_guide.md
   choices_methods.md
   benchmark_and_profiling.md
   troubleshooting.md