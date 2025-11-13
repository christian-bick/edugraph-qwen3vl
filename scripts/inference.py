import os
import torch
import argparse
import json

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)
from peft import PeftModel

def main(args):
    # --- Configuration ---
    # Update to a specific Qwen3-VL model
    base_model_id = "Qwen/Qwen3-VL-4B-Instruct"
    adapter_path = "out/adapters/multimodal_adapter"

    print("--- Loading model and adapter for inference ---")

    # Load the processor
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)

    # Load the base model without quantization
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load the LoRA adapter and merge it into the base model
    print(f"Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    print("Adapter merged successfully.")

    # --- Run Inference using model.generate() ---
    print(f"\n--- Running inference on {args.image_path} ---")
    
    # Load the detailed prompt from the file
    with open("prompts/classification_v2.txt", "r") as f:
        prompt_text = f.read()

    # Create the conversational prompt
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]},
    ]
    
    # Apply the chat template and prepare inputs
    text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[args.image_path], return_tensors="pt").to(model.device)

    # Generate the token IDs
    input_ids_len = inputs['input_ids'].shape[1]
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    
    # Modernized response parsing: decode only the newly generated tokens
    new_tokens = generated_ids[0, input_ids_len:]
    assistant_response = processor.decode(new_tokens, skip_special_tokens=True)

    print("\n--- Generated Classification ---")
    print(assistant_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned EduGraph model.")
    parser.add_argument("image_path", type=str, help="Path to the image file to classify.")
    args = parser.parse_args()
    main(args)
