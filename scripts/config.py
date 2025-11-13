from peft import LoraConfig

# A comprehensive set of target modules for modern Qwen models.
# This targets all linear layers in the attention and feed-forward blocks.
COMMON_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

class Stage1Config:
    def __init__(self, r, lora_alpha, lora_dropout, learning_rate, num_train_epochs):
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=COMMON_TARGET_MODULES,
            task_type="CAUSAL_LM"
        )
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs

class Stage2Config:
    def __init__(self, r, lora_alpha, lora_dropout, learning_rate, num_train_epochs):
        self.lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=COMMON_TARGET_MODULES,
            task_type="CAUSAL_LM"
        )
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs

class ModelConfig:
    def __init__(self, stage1: Stage1Config, stage2: Stage2Config):
        self.stage1 = stage1
        self.stage2 = stage2

# --- Configurations for different model sizes ---

# Configurations for the 4B model
config_4b = ModelConfig(
    stage1=Stage1Config(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        learning_rate=2e-4,
        num_train_epochs=6
    ),
    stage2=Stage2Config(
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        learning_rate=1e-4,
        num_train_epochs=4
    )
)

# Configurations for the 8B model
config_8b = ModelConfig(
    stage1=Stage1Config(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        learning_rate=1e-4,
        num_train_epochs=8
    ),
    stage2=Stage2Config(
        r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        learning_rate=1e-4,
        num_train_epochs=4
    )
)

def get_config(model_size: str):
    """
    Returns the configuration for a given model size string.
    Defaults to the 4B configuration.
    """
    model_size_lower = model_size.lower()
    if "8b" in model_size_lower:
        print("--- Using configuration for 8B model. ---")
        return config_8b
    elif "4b" in model_size_lower:
        print("--- Using configuration for 4B model. ---")
        return config_4b
    else:
        print(f"--- No specific config for '{model_size}'. Using default 4B model configuration. ---")
        return config_4b