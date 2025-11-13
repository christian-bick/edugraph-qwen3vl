import os
import torch
from dotenv import load_dotenv

from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Qwen3VLForConditionalGeneration,
)
from peft import get_peft_model, PeftModel
from trl import SFTTrainer
from scripts.config import get_config

def main():
    # Load environment variables
    load_dotenv()

    # --- Configuration ---
    run_mode = os.environ.get("RUN_MODE", "train")
    model_size = os.environ.get("MODEL_SIZE", "4b")
    
    model_config = get_config(model_size)
    stage2_config = model_config.stage2
    base_model_id = f"Qwen/Qwen3-VL-{model_size.upper()}-Instruct"
    
    multimodal_dataset_path = "train_dataset.jsonl"
    knowledge_adapter_path = "out/adapters/knowledge_adapter"
    final_adapter_path = "out/adapters/multimodal_adapter"
    os.makedirs(os.path.dirname(final_adapter_path), exist_ok=True)

    # --- Mode-specific Adjustments ---
    if run_mode == "test":
        print("--- Running in TEST mode ---")
        num_train_epochs = 1
        max_train_samples = 30
    else:
        print("--- Running in TRAIN mode ---")
        num_train_epochs = stage2_config.num_train_epochs
        max_train_samples = None

    print("--- Starting Stage 2: Multimodal Task Tuning ---")

    # --- Model and Processor Loading ---
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # --- Adapter Merging and Configuration ---
    if os.path.exists(knowledge_adapter_path):
        print(f"Loading and merging knowledge adapter from {knowledge_adapter_path}...")
        model = PeftModel.from_pretrained(model, knowledge_adapter_path)
        model = model.merge_and_unload()
        print("Knowledge adapter loaded and merged successfully.")
    else:
        print(f"Knowledge adapter not found at {knowledge_adapter_path}, proceeding with base model.")
    
    model = get_peft_model(model, stage2_config.lora_config)
    print("Trainable parameters for Stage 2:")
    model.print_trainable_parameters()

    # --- Data Loading and Simplified Formatting ---
    dataset = load_dataset("json", data_files=multimodal_dataset_path, split="train")

    with open("prompts/classification_v2.txt", "r") as f:
        prompt_text = f.read()

    def format_chat_messages(examples):
        """
        Creates a 'messages' column with the chat dictionary,
        which SFTTrainer will use to apply the template.
        The 'image' column is passed through automatically.
        """
        messages = []
        for conversation in examples['conversations']:
            assistant_content = conversation[1]['value']
            messages.append([
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": [{"type": "image"}]},
                {"role": "assistant", "content": assistant_content}
            ])
        return {"messages": messages}

    # Get the columns to remove, but be sure to keep the 'image' column
    columns_to_remove = [col for col in dataset.column_names if col != 'image']
    
    # Apply the simple formatting
    processed_dataset = dataset.map(format_chat_messages, batched=True, remove_columns=columns_to_remove)

    if max_train_samples:
        processed_dataset = processed_dataset.select(range(max_train_samples))

    # --- Trainer Setup and Execution ---
    training_args = TrainingArguments(
        output_dir="out/results/multimodal_results",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=stage2_config.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        # Evaluation is removed as per the improvement suggestions
        bf16=True, 
        max_grad_norm=1.0,
    )

    # Use SFTTrainer for a simpler training loop
    trainer = SFTTrainer(
        model=model,
        processor=processor, # Pass the processor to handle multimodal inputs
        args=training_args,
        train_dataset=processed_dataset,
        dataset_text_field="messages", # Tell SFTTrainer which column has the chat messages
        max_seq_length=2048, # Recommended to set a max sequence length
    )

    print("Starting training with SFTTrainer...")
    trainer.train()
    print("Training finished.")

    print(f"Saving final adapter to {final_adapter_path}")
    model.save_pretrained(final_adapter_path)

if __name__ == "__main__":
    main()
