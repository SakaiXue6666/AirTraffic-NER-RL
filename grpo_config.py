from data import load_entity_types

MODEL_NAME = "Qwen/Qwen3-1.7B"

# 这里建议填 SFT 后的 LoRA 路径，GRPO 从它继续训
SFT_LORA_PATH = "./output/lora_sft"
GRPO_OUTPUT_DIR = "./output/lora_grpo"

TRAIN_FILE = "datasets/cluener/train.json"
ENT2ID_JSON = "datasets/cluener/ent2id.json"
ENTITY_TYPES = load_entity_types(ENT2ID_JSON)

# prompt / generation
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 256

# GRPO
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-6
NUM_TRAIN_EPOCHS = 1
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

NUM_GENERATIONS = 4
TEMPERATURE = 0.8
TOP_P = 0.9

# beta=0 时不加载 reference model，省显存
BETA = 0.0
EPSILON = 0.2

# LoRA
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 先不用 vLLM，单机先跑通
USE_VLLM = False

SEED = 42