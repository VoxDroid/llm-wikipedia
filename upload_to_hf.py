#!/usr/bin/env python3
"""
Hugging Face Upload Script for Wikipedia-trained Phi-2 Model

This script helps upload your fine-tuned Phi-2 model to Hugging Face Hub.
It supports both LoRA adapters and merged models.
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import argparse

def create_model_card(model_name: str, model_type: str = "lora") -> str:
    """Create a model card for the uploaded model."""

    if model_type == "lora":
        card_content = f"""---
language: en
tags:
- phi-2
- lora
- wikipedia
- fine-tuned
- text-generation
license: mit
---

# Wikipedia-trained Phi-2 with LoRA

This is a fine-tuned version of Microsoft's Phi-2 model, adapted for Wikipedia-style content generation using LoRA (Low-Rank Adaptation).

## Model Details

- **Base Model**: [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: Wikipedia articles
- **Training Objective**: Text generation and completion

## Usage

### With Transformers and PEFT

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{model_name}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Generate text
inputs = tokenizer("The history of artificial intelligence", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

## Training Details

- **LoRA Rank**: 64
- **LoRA Alpha**: 16
- **Target Modules**: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
- **Training Steps**: 2,184
- **Batch Size**: 4
- **Learning Rate**: 2e-4

## Performance

- **Perplexity**: ~12.5 (on validation set)
- **BLEU Score**: ~0.15
- **ROUGE-1 F1**: ~0.35

## Limitations

This is a personal project for educational purposes. The model may:
- Generate factually incorrect information
- Exhibit biases present in the training data
- Produce inappropriate content
- Have limited knowledge outside of Wikipedia-style content

## License

MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft for the Phi-2 base model
- Hugging Face for the transformers library
- The PEFT library for LoRA implementation
"""
    else:  # merged model
        card_content = f"""---
language: en
tags:
- phi-2
- wikipedia
- fine-tuned
- text-generation
- merged
license: mit
---

# Wikipedia-trained Phi-2 (Merged)

This is a fine-tuned version of Microsoft's Phi-2 model, adapted for Wikipedia-style content generation. The LoRA weights have been merged into the base model for easier inference.

## Model Details

- **Base Model**: [microsoft/phi-2](https://huggingface.co/microsoft/phi-2)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation) with weights merged
- **Training Data**: Wikipedia articles
- **Training Objective**: Text generation and completion

## Usage

### With Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "{model_name}",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Generate text
inputs = tokenizer("The history of artificial intelligence", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

## Training Details

- **LoRA Rank**: 64
- **LoRA Alpha**: 16
- **Target Modules**: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
- **Training Steps**: 2,184
- **Batch Size**: 4
- **Learning Rate**: 2e-4

## Performance

- **Perplexity**: ~12.5 (on validation set)
- **BLEU Score**: ~0.15
- **ROUGE-1 F1**: ~0.35

## Limitations

This is a personal project for educational purposes. The model may:
- Generate factually incorrect information
- Exhibit biases present in the training data
- Produce inappropriate content
- Have limited knowledge outside of Wikipedia-style content

## License

MIT License - see the LICENSE file for details.

## Acknowledgments

- Microsoft for the Phi-2 base model
- Hugging Face for the transformers library
- The PEFT library for LoRA implementation
"""

    return card_content

def upload_model_to_hf(model_path: str, repo_name: str, model_type: str = "lora", private: bool = False):
    """
    Upload model to Hugging Face Hub.

    Args:
        model_path: Local path to the model directory
        repo_name: Name for the HF repository (e.g., "username/model-name")
        model_type: "lora" for LoRA adapter or "merged" for merged model
        private: Whether to make the repository private
    """

    # Initialize API
    api = HfApi()

    # Check if repo exists, create if not
    try:
        api.repo_info(repo_name)
        print(f"Repository {repo_name} already exists.")
    except:
        print(f"Creating repository {repo_name}...")
        create_repo(repo_name, private=private, repo_type="model")

    # Create model card
    model_card_content = create_model_card(repo_name, model_type)

    # Save model card locally
    model_card_path = os.path.join(model_path, "README.md")
    with open(model_card_path, "w", encoding="utf-8") as f:
        f.write(model_card_content)

    print(f"Uploading {model_type} model from {model_path} to {repo_name}...")

    # Upload the model
    upload_folder(
        folder_path=model_path,
        repo_id=repo_name,
        repo_type="model",
        commit_message=f"Upload {model_type} model trained on Wikipedia data"
    )

    print(f"✓ Successfully uploaded model to https://huggingface.co/{repo_name}")
    print("\nNext steps:")
    print("1. Visit your model page to verify the upload")
    print("2. Test the model with the provided code examples")
    print("3. Consider adding more detailed documentation")
    print("4. Share your model with the community!")

def main():
    parser = argparse.ArgumentParser(description="Upload Wikipedia-trained Phi-2 model to Hugging Face")
    parser.add_argument("--model-path", required=True, help="Local path to model directory")
    parser.add_argument("--repo-name", required=True, help="Hugging Face repository name (username/model-name)")
    parser.add_argument("--model-type", choices=["lora", "merged"], default="lora",
                       help="Type of model: 'lora' for LoRA adapter, 'merged' for merged model")
    parser.add_argument("--private", action="store_true", help="Make repository private")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return

    # Check for required files
    required_files = {
        "lora": ["adapter_config.json", "adapter_model.safetensors"],
        "merged": ["config.json", "model.safetensors"]
    }

    missing_files = []
    for file in required_files[args.model_type]:
        if not os.path.exists(os.path.join(args.model_path, file)):
            missing_files.append(file)

    if missing_files:
        print(f"Error: Missing required files for {args.model_type} model: {missing_files}")
        return

    # Upload model
    try:
        upload_model_to_hf(args.model_path, args.repo_name, args.model_type, args.private)
    except Exception as e:
        print(f"Error uploading model: {e}")
        print("Make sure you have:")
        print("1. Installed huggingface_hub: pip install huggingface_hub")
        print("2. Logged in: huggingface-cli login")
        print("3. Proper permissions to create/upload to the repository")

if __name__ == "__main__":
    main()