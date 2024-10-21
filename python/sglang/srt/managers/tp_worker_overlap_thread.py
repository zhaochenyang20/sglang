"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""A tensor parallel worker."""

import logging
import threading
import time
from queue import Queue
from typing import Optional

import torch

from sglang.srt.managers.io_struct import UpdateWeightReqInput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class TpModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
    ):
        # Load the model
        self.worker = TpModelWorker(server_args, gpu_id, tp_rank, dp_rank, nccl_port)
        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = torch.empty(
            (self.max_running_requests * 5,), dtype=torch.int32, device=self.device
        )

        # Launch a thread
        self.input_queue = Queue()
        self.output_queue = Queue()
        self.forward_stream = torch.cuda.Stream()
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
        )
        self.forward_thread.start()

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_tp_cpu_group(self):
        return self.worker.get_tp_cpu_group()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool,
        )

    def forward_thread_func(self):
        with torch.cuda.stream(self.forward_stream):
            self.forward_thread_func_()

    @torch.inference_mode()
    def forward_thread_func_(self):
        while True:
            tic1 = time.time()
            model_worker_batch, future_token_ids_ct = self.input_queue.get()

            # Resolve future tokens in the input
            tic2 = time.time()
            resolved_input_ids = model_worker_batch.input_ids
            future_mask = resolved_input_ids < 0
            resolved_input_ids[future_mask] = self.future_token_ids_map[
                -resolved_input_ids[future_mask]
            ]

            # Run forward
            logits_output, next_token_ids = self.worker.forward_batch_generation(
                model_worker_batch
            )

            # Update the future token ids map
            bs = len(model_worker_batch.seq_lens)
            future_next_token_ids = torch.arange(
                -(future_token_ids_ct + bs),
                -(future_token_ids_ct),
                dtype=torch.int32,
                device=self.device,
            )
            self.future_token_ids_map[-future_next_token_ids] = next_token_ids.to(
                torch.int32
            )

            # Set the result
            next_token_ids = next_token_ids.tolist()
            assert logits_output.next_token_logprobs is None, "Not supported"
            self.output_queue.put((None, next_token_ids))

            if False:
                tic3 = time.time()
                self.acc_time_with_waiting += tic3 - tic1
                self.acc_time_without_waiting += tic3 - tic2
                if self.forward_queue.qsize() == 0:
                    logger.info(
                        f"{self.acc_time_with_waiting=:.3f}, {self.acc_time_without_waiting=:.3f}, {self.forward_queue.qsize()=}"
                    )

    def resulve_batch_result(self, bid: int):
        logits_output, next_token_ids = self.output_queue.get()
        return logits_output, next_token_ids

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch):
        # Push a new batch to the queue
        self.input_queue.put((model_worker_batch.copy(), self.future_token_ids_ct))

        # Allocate output future objects
        bs = len(model_worker_batch.seq_lens)
        future_next_token_ids = torch.arange(
            -(self.future_token_ids_ct + bs),
            -(self.future_token_ids_ct),
            dtype=torch.int32,
            device=self.device,
        )
        self.future_token_ids_ct = (
            self.future_token_ids_ct + bs
        ) % self.future_token_ids_limit
        return None, future_next_token_ids

    def forward_batch_embedding(self, model_worker_batch: ModelWorkerBatch):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        embeddings = logits_output.embeddings
        return embeddings

    def update_weights(self, recv_req: UpdateWeightReqInput):
        success, message = self.model_runner.update_weights(
            recv_req.model_path, recv_req.load_format
        )
        return success, message
