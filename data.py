import json


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


def load_entity_types(ent2id_path):
    """从 ent2id.json（实体名 -> 整数 id）得到按 id 排序的实体类型列表。"""
    with open(ent2id_path, "r", encoding="utf-8") as f:
        ent2id = json.load(f)
    if not isinstance(ent2id, dict):
        raise ValueError("ent2id 文件应为 JSON 对象：实体名 -> id")
    return [name for name, _ in sorted(ent2id.items(), key=lambda kv: kv[1])]
