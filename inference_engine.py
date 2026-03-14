import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import asyncio

class InferenceEngine:
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    async def infer(self, prompt, max_length):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        loop = asyncio.get_event_loop()
        outputs = await loop.run_in_executor(None, lambda: self.model.generate(**inputs, max_length=max_length))
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def stream_infer(self, prompt, max_length):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(inputs, streamer=streamer, max_length=max_length)
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        for new_text in streamer:
            yield new_text