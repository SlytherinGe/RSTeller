from .base_annotator import BaseAnnotator

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class GemmaAnnotator(BaseAnnotator):

    _VALID_MODELS = ["google/gemma-2-9b-it"]

    def __init__(self, model_id, **kwargs):
        super().__init__(model_id=model_id, **kwargs)

        self.model_id = model_id
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
        # max_tokens is for the maximum length for generation.
        self.sampling_params = SamplingParams(
            temperature=0.8, max_tokens=512, top_p=0.95, top_k=1
        )

        # Input the model name or path. Can be GPTQ or AWQ models.
        self.llm = LLM(model=model_id)

    def inference(self, text):

        prompts = [text]

        outputs = self.llm.generate(prompts, self.sampling_params)

        return outputs[0].outputs[0].text
