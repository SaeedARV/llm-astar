import transformers
import torch
from transformers import BitsAndBytesConfig
import os

class Llama3:
    def __init__(self, hf_token=None):
        # Using Mistral-7B-Instruct-v0.2 with quantization
        model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        
        # Use provided token or try to get from environment
        token = hf_token or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("Hugging Face token is required to access the gated model. Please provide it via constructor or set HF_TOKEN environment variable.")
        
        # Configure 4-bit quantization for CUDA
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer first
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=token
        )
        
        # Set up terminators and padding
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("</s>")
        ]
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Set device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load model with explicit device mapping
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map={"": self.device},  # Force all layers to same device
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=token
        )
    
    def ask(self, prompt):
        # Prepare inputs and move to correct device
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=8000,
                eos_token_id=self.terminators,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]