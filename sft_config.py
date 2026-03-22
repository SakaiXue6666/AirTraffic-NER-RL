from data import load_entity_types

MODEL_NAME = "Qwen/Qwen3-1.7B"
TRAIN_FILE = "datasets/cluener/train.json"
ENT2ID_JSON = "datasets/cluener/ent2id.json"
OUTPUT_DIR = "output/lora_sft"

ENTITY_TYPES = load_entity_types(ENT2ID_JSON)

MAX_LENGTH = 512

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10
SAVE_STEPS = 100