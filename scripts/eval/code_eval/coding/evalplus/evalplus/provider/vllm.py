from typing import List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)


class VllmDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        force_base_prompt: bool = False,
        tensor_parallel_size: int = 1,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager: bool = False,
        max_model_len: int = 32768,
        gguf_file: str = None,
        **kwargs
    ) -> None:
        super().__init__(name, **kwargs)

        kwargs = {
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "enable_prefix_caching": enable_prefix_caching,
            "enable_chunked_prefill": enable_chunked_prefill,
            "enforce_eager": enforce_eager,
            "max_model_len": max_model_len,
        }

        self.force_base_prompt = force_base_prompt
        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        tokenizer_kwargs = {}
        if gguf_file is None:
            tokenizer_kwargs["use_fast"] = False
        else:
            tokenizer_kwargs["gguf_file"] = gguf_file
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, **tokenizer_kwargs)
        if self.is_direct_completion():
            self.eos += extra_eos_for_direct_completion(dataset)
        else:
            self.eos += ["\n```\n"]
        self.llm = LLM(model=name, **kwargs)

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    def _format_prompt(self, prompt: str) -> str:
        if self.is_direct_completion():
            return prompt
        return make_raw_chat_prompt(
            prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
        )

    def _sampling_params(self, do_sample: bool, num_samples: int) -> SamplingParams:
        temperature = self.temperature if do_sample else 0.0
        if do_sample:
            assert self.temperature > 0, "Temperature must be greater than 0!"
        return SamplingParams(
            n=num_samples,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            top_p=self.top_p,
            stop=self.eos,
        )

    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        batch_size = min(self.batch_size, num_samples)
        formatted = self._format_prompt(prompt)
        vllm_outputs = self.llm.generate(
            [formatted] * batch_size,
            self._sampling_params(do_sample, 1),
            use_tqdm=False,
        )
        return [x.outputs[0].text for x in vllm_outputs]

    def codegen_batch(
        self, prompts: List[str], do_sample: bool = True, num_samples: int = 1
    ) -> List[List[str]]:
        if not prompts:
            return []
        formatted = [self._format_prompt(prompt) for prompt in prompts]
        vllm_outputs = self.llm.generate(
            formatted,
            self._sampling_params(do_sample, num_samples),
            use_tqdm=False,
        )
        return [[o.text for o in out.outputs] for out in vllm_outputs]
