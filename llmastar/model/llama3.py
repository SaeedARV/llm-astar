import torch
import sys
import os

# Check for specific PyTorch version required for compatibility
if not torch.__version__.startswith("2.0.1"):
    print(f"Warning: This code expects PyTorch 2.0.1, but found {torch.__version__}.")
    print("Please install the required version with: pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118")

# Check CUDA availability
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. This will run very slowly on CPU.")
    print("Please use a GPU-enabled environment for better performance.")

# Singleton pattern to ensure model is loaded only once
_MODEL_INSTANCE = None
_TOKENIZER_INSTANCE = None
_IS_UNSLOTH_MODEL = False

class Llama3:
    def __init__(self, hf_token=None):
        global _MODEL_INSTANCE, _TOKENIZER_INSTANCE, _IS_UNSLOTH_MODEL
        
        # If model is already loaded, reuse it
        if _MODEL_INSTANCE is not None and _TOKENIZER_INSTANCE is not None:
            self.model = _MODEL_INSTANCE
            self.tokenizer = _TOKENIZER_INSTANCE
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            # Set terminators and padding once
            self.terminators = [self.tokenizer.eos_token_id]
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.is_unsloth_model = _IS_UNSLOTH_MODEL
            return
        
        # Try to import unsloth for 4-bit optimization
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            print("Error: unsloth package not found. Please install it with: pip install unsloth==2024.3.19")
            print("This is required for 4-bit quantization of Llama 3 8B.")
            sys.exit(1)
            
        # Using Unsloth's pre-quantized Llama 3.1 8B model
        model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
        
        # Use provided token or try to get from environment
        token = hf_token or os.getenv("HF_TOKEN")
        
        # Set device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_unsloth_model = False
        
        try:
            # Load model and tokenizer with pre-quantized 4-bit version from Unsloth
            from unsloth import FastLanguageModel
            max_seq_length = 2048  # Auto support for RoPE Scaling
            dtype = None  # Auto detection (Float16 for Tesla T4, V100, Bfloat16 for Ampere+)
            load_in_4bit = True  # Use 4-bit quantization to reduce memory usage
            
            print(f"Loading Llama 3.1 model with Unsloth {model_id}...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                token=token
            )
            print("Llama 3.1 model loaded successfully with Unsloth!")
            self.is_unsloth_model = True
            _IS_UNSLOTH_MODEL = True
            
            # Note: We do NOT need to move the model to device as it's already on the correct device
            # The error occurs because .to() is not supported for 4-bit/8-bit bitsandbytes models
                
        except Exception as e:
            print(f"Error loading Llama 3.1 model with unsloth: {e}")
            print("Falling back to standard Transformers loading...")
            
            # Standard loading with transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            # Try using "meta-llama/Meta-Llama-3-8B" if pre-quantized version fails
            fallback_model_id = "meta-llama/Meta-Llama-3-8B"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                fallback_model_id,
                token=token
            )
            
            # Load model with 4-bit quantization if on GPU
            if torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model_id,
                    quantization_config=quantization_config,
                    token=token,
                    low_cpu_mem_usage=True
                )  # Note: No .to(device) for BitsAndBytes models
            else:
                # Load without quantization on CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback_model_id,
                    token=token,
                    low_cpu_mem_usage=True
                )  # Will be on CPU by default
        
        # Set up terminators
        self.terminators = [
            self.tokenizer.eos_token_id
        ]
        
        # Set padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Save instances for reuse
        _MODEL_INSTANCE = self.model
        _TOKENIZER_INSTANCE = self.tokenizer
    
    def ask(self, prompt):
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            # Generate with standard generation - optimized for speed
            with torch.no_grad():
                if self.is_unsloth_model:
                    # Try using Unsloth's specialized generate method
                    try:
                        from unsloth import FastLanguageModel
                        # Use a special method for Unsloth models that doesn't use paged attention
                        outputs = FastLanguageModel.generate(
                            model=self.model,
                            tokenizer=self.tokenizer,
                            prompts=[prompt],
                            max_new_tokens=1024,
                            temperature=0.0,  # Deterministic output
                            top_p=1.0,
                            use_cache=True
                        )
                        return outputs[0]  # Return the first (and only) output
                        
                    except (ImportError, AttributeError) as e:
                        print(f"Warning: Unable to use Unsloth's FastLanguageModel.generate: {e}")
                        print("Falling back to standard tokenizer-based approach...")
                        # Continue with standard generation below
                
                # Standard generation approach
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    eos_token_id=self.terminators,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    use_cache=True
                )
                return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            # If we get an error about paged_attention or other issues, try an even simpler approach
            print(f"Error during generation: {e}")
            print("Trying another approach with simple generation...")
            
            try:
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                # Use a very simple greedy generation approach
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        num_beams=1,  # Simple greedy search
                        do_sample=False,
                        use_cache=True
                    )
                    
                # Decode and return only the new tokens
                return self.tokenizer.decode(generated_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            except Exception as nested_e:
                print(f"Error with simplified generation: {nested_e}")
                # Last resort - return a message about the error
                return f"Error generating response: {e}. The model may need to be updated or reinstalled."