from datasets import Dataset

from data import load_jsonl, build_prompt
from grpo_config import ENTITY_TYPES


def build_rl_dataset(train_file):
    raw_data = load_jsonl(train_file)

    rows = []
    for sample in raw_data:
        text = sample["text"]
        label = sample["label"]
        prompt = build_prompt(text, ENTITY_TYPES)

        rows.append(
            {
                "prompt": prompt,
                "text": text,
                "label": label,
            }
        )

    return Dataset.from_list(rows)
