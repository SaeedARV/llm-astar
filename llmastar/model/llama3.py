import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class Llama3:
    def __init__(self, hf_token=None):
        # Using Google's Gemma 2B, small but powerful for reasoning
        model_id = "google/gemma-2b"
        
        # Use provided token or try to get from environment
        token = hf_token or os.getenv("HF_TOKEN")
        
        # Set device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=token
        )
        
        # Load model with 4-bit quantization if on GPU
        if torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                token=token,
                low_cpu_mem_usage=True
            ).to(self.device)
        else:
            # Load without quantization on CPU
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=token,
                low_cpu_mem_usage=True
            ).to(self.device)
        
        # Set up terminators
        self.terminators = [
            self.tokenizer.eos_token_id
        ]
        
        # Set padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
    
    def ask(self, prompt):
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with standard generation - optimized for speed
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=self.terminators,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True  # Ensure caching is enabled for faster generation
            )
        
        # Decode and return
        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)