import torch
from unsloth import FastLanguageModel
import os

class Llama3:
    def __init__(self, hf_token=None):
        # Using Unsloth's pre-quantized Llama 3 model
        model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
        
        # Use provided token or try to get from environment
        token = hf_token or os.getenv("HF_TOKEN")
        
        # Set device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model with Unsloth optimization
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=2048,  # Unsloth's recommended length
            dtype=None,  # Auto detection
            load_in_4bit=True,  # Use 4bit quantization
            token=token
        )
        
        # Set up terminators
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("</s>")
        ]
        
        # Set padding
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
    
    def ask(self, prompt):
        # Clear CUDA cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate with Unsloth optimization
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
        
        # Clear CUDA cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Decode and return
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):]