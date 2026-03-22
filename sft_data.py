import json
from datasets import Dataset
from config import ENTITY_TYPES, MAX_LENGTH


def build_prompt(text, entity_types):
    entity_str = ", ".join(entity_types)
    return (
        "<|im_start|>system\n"
        f"你是一个飞行与管制文本命名实体识别助手。"
        f"请从输入文本中抽取实体，并严格输出 JSON。"
        f"实体类别包括：{entity_str}。"
        f"输出格式为：{{\"类别\": {{\"实体mention\": [[start, end]]}}}}。"
        f"如果某类实体不存在，则不要输出该类。"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"# 文本\n{text}\n\n"
        "请输出抽取结果：\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def build_dataset(file_path):
    raw_data = load_jsonl(file_path)
    return Dataset.from_list(raw_data)


def preprocess_function(example, tokenizer):
    text = example["text"]
    label = example["label"]

    prompt = build_prompt(text, ENTITY_TYPES)
    target = json.dumps(label, ensure_ascii=False)

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False
    )["input_ids"]

    target_ids = tokenizer(
        target,
        add_special_tokens=False
    )["input_ids"]

    input_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)

    labels = [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    else:
        pad_len = MAX_LENGTH - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attention_mask = attention_mask + [0] * pad_len
        labels = labels + [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        remove_columns=dataset.column_names
    )