import argparse
import json
import os
import random
from typing import Dict, Iterable, List

import pandas as pd

from rl_prompt import build_messages
from reward_utils import infer_lang


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def to_verl_row(item: Dict, idx: int, split: str) -> Dict:
    lang = infer_lang(item)
    question = item.get("question", "")
    extra = {
        "id": item.get("id", ""),
        "index": idx,
        "split": split,
        "question": question,
        "target_lang": lang,
        "test_in": item.get("test_in", []),
        "test_out": item.get("test_out", []),
        "test_code": item.get("test_code", ""),
        "source": item.get("source", ""),
    }
    return {
        "data_source": "kodcode_rl",
        "prompt": build_messages(question, lang),
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": extra,
    }


def write_parquet_split(input_jsonl: str, out_dir: str, seed: int = 42, val_size: int = 512):
    rows = list(read_jsonl(input_jsonl))
    rng = random.Random(seed)
    rng.shuffle(rows)

    if len(rows) <= 1:
        train_rows, val_rows = rows, rows
    else:
        n_val = min(val_size, max(1, len(rows) // 20))
        val_rows = rows[:n_val]
        train_rows = rows[n_val:]
        if not train_rows:
            train_rows = val_rows

    os.makedirs(out_dir, exist_ok=True)
    train_data = [to_verl_row(x, i, "train") for i, x in enumerate(train_rows)]
    val_data = [to_verl_row(x, i, "val") for i, x in enumerate(val_rows)]

    train_path = os.path.join(out_dir, "train.parquet")
    val_path = os.path.join(out_dir, "val.parquet")
    pd.DataFrame(train_data).to_parquet(train_path, index=False)
    pd.DataFrame(val_data).to_parquet(val_path, index=False)
    return train_path, val_path, len(train_data), len(val_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=512)
    args = parser.parse_args()

    train_path, val_path, n_train, n_val = write_parquet_split(args.input, args.out_dir, args.seed, args.val_size)
    print(json.dumps({"train": train_path, "val": val_path, "n_train": n_train, "n_val": n_val}, ensure_ascii=False))


if __name__ == "__main__":
    main()
