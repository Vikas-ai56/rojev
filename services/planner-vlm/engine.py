"""Embodied-R1.5 (a Qwen3-VL-8B fine-tune) behind vLLM, asked image + text questions in batches.

Follows Embodied-R1.5's official offline engine (github.com/pickxiguapi/Embodied-R1.5 @ b38627f2,
`inference/vllm_offline_example.py`): `AutoProcessor` chat template + `qwen_vl_utils.process_vision_info
(image_patch_size=16)` build each vLLM input; `LLM(max_model_len=10240, gpu_memory_utilization=0.85, seed=3407)`;
its sampling defaults (temperature 1.0, top_p 1.0, repetition_penalty 1.05, 1024 new tokens). vLLM is the serving
path R1.5's model card documents, so its batch throughput and single-request latency are the ones a deployment sees.
Weights come from the revision pinned in `config/embodied_r1_weights.txt`, resolved to a local snapshot by
`scripts/hf_pins.py`. vLLM and transformers are imported in `load` (Linux only), so the bench's tests run without them.
"""
import os
import time
from dataclasses import dataclass

from PIL import Image

MAX_MODEL_LEN = 10240
GPU_MEMORY_UTILIZATION = 0.85
SEED = 3407
MAX_IMAGES = 8
SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "repetition_penalty": 1.05, "max_tokens": 1024}


@dataclass(frozen=True)
class Request:
    images: list[Image.Image]
    text: str


@dataclass(frozen=True)
class Reply:
    text: str
    new_tokens: int


class Engine:
    def __init__(self, llm, processor, process_vision_info, sampling):
        self._llm = llm
        self._processor = processor
        self._process_vision_info = process_vision_info
        self._sampling = sampling

    @classmethod
    def load(cls, snapshot: str) -> "Engine":
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        processor = AutoProcessor.from_pretrained(snapshot)
        llm = LLM(model=snapshot, max_model_len=MAX_MODEL_LEN, gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
                  seed=SEED, limit_mm_per_prompt={"image": MAX_IMAGES}, dtype="bfloat16")
        return cls(llm, processor, process_vision_info, SamplingParams(seed=SEED, **SAMPLING))

    def _input(self, request: Request) -> dict:
        content = [{"type": "image", "image": image} for image in request.images]
        messages = [{"role": "user", "content": content + [{"type": "text", "text": request.text}]}]
        prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, _, video_kwargs = self._process_vision_info(messages, image_patch_size=16, return_video_kwargs=True,
                                                            return_video_metadata=True)
        return {"prompt": prompt, "multi_modal_data": {"image": images}, "mm_processor_kwargs": video_kwargs}

    def generate(self, requests: list[Request]) -> list[Reply]:
        outputs = self._llm.generate([self._input(r) for r in requests], sampling_params=self._sampling)
        return [Reply(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outputs]

    def latency_ms(self, request: Request) -> float:
        start = time.perf_counter()
        self.generate([request])
        return (time.perf_counter() - start) * 1000
