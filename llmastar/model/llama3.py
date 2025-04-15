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
        
        # Set parameters directly as shown in example
        max_seq_length = 2048  # Auto support for RoPE Scaling
        dtype = None  # Auto detection
        load_in_4bit = True  # Use 4-bit quantization
        
        print(f"Loading Llama 3.1 model with Unsloth from {model_id}...")
        
        # Load directly with Unsloth
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=token
        )
        
        # Apply LoRA configuration
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407
        )
        
        # Enable faster inference
        FastLanguageModel.for_inference(self.model)
        print("Llama 3.1 model loaded successfully!")
        
        # Set padding token for generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Save instances for reuse
        _MODEL_INSTANCE = self.model
        _TOKENIZER_INSTANCE = self.tokenizer
    
    def ask(self, prompt):
        """Generate a response to the prompt using the loaded model."""
        try:
            # Prepare inputs
            inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
            
            # Use model's generate method
            from unsloth import FastLanguageModel
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,  # Deterministic output
                use_cache=True
            )
            
            # Decode the response (only the generated part)
            input_length = inputs.input_ids.shape[1]
            return self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return f"Error generating response: {str(e)}"