import argparse
import concurrent.futures as futures
import json
import os
import random
import threading
import time
from typing import Dict, Iterable, List

import requests

from prepare_verl_data import write_parquet_split
from reward_utils import evaluate_response, infer_lang
from rl_prompt import build_messages


write_lock = threading.Lock()


def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_seen(*paths: str) -> set:
    seen = set()
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get("id"):
                        seen.add(obj["id"])
                except Exception:
                    pass
    return seen


def append_jsonl(path: str, obj: Dict):
    with write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def request_once(endpoint: str, model: str, item: Dict, args) -> str:
    payload = {
        "model": model,
        "messages": build_messages(item.get("question", ""), infer_lang(item)),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    for attempt in range(args.retry):
        try:
            resp = requests.post(url, json=payload, timeout=args.request_timeout)
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            time.sleep(1 + attempt)
        except Exception:
            time.sleep(1 + attempt)
    return ""


def process_item(item: Dict, endpoints: List[str], args) -> Dict:
    samples = []
    stats = []
    for k in range(args.sample_times):
        endpoint = endpoints[(hash(item.get("id", "")) + k) % len(endpoints)]
        text = request_once(endpoint, args.model, item, args)
        score = evaluate_response(text, item, timeout=args.exec_timeout)
        samples.append({"sample_id": k, "response": text, "score": score})
        stats.append(score)

    correct = sum(1 for s in stats if s.get("all_pass"))
    keep = correct < args.sample_times
    out = dict(item)
    out["filter_model"] = args.model_tag
    out["sample_times"] = args.sample_times
    out["correct_count"] = correct
    out["keep_for_rl"] = keep
    out["filter_scores"] = stats
    if args.save_samples:
        out["filter_samples"] = samples
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/data/250010072/zlh/king/code_tailored_dataset/data/kodcode_reasoning_sft.jsonl")
    parser.add_argument("--out_dir", default="/data/250010072/zlh/king/code_tailored_dataset/train/rl/data")
    parser.add_argument("--model", required=True, help="served model name used by vLLM")
    parser.add_argument("--model_tag", required=True, help="short tag used in output filenames")
    parser.add_argument("--endpoints", default="http://127.0.0.1:35000,http://127.0.0.1:35001,http://127.0.0.1:35002,http://127.0.0.1:35003")
    parser.add_argument("--sample_times", type=int, default=10)
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--request_timeout", type=int, default=180)
    parser.add_argument("--exec_timeout", type=int, default=20)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--save_samples", action="store_true")
    parser.add_argument("--no_parquet", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    keep_path = os.path.join(args.out_dir, f"{args.model_tag}_rl_keep.jsonl")
    easy_path = os.path.join(args.out_dir, f"{args.model_tag}_all_correct_removed.jsonl")
    endpoints = [x.strip() for x in args.endpoints.split(",") if x.strip()]

    items = list(read_jsonl(args.input))
    rng = random.Random(args.seed)
    rng.shuffle(items)
    if args.max_items > 0:
        items = items[: args.max_items]

    seen = load_seen(keep_path, easy_path)
    items = [x for x in items if x.get("id") not in seen]
    print(f"pending={len(items)} seen={len(seen)} out_dir={args.out_dir}", flush=True)

    kept = 0
    removed = 0
    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_item, item, endpoints, args) for item in items]
        for fut in futures.as_completed(futs):
            obj = fut.result()
            done += 1
            if obj["keep_for_rl"]:
                append_jsonl(keep_path, obj)
                kept += 1
            else:
                append_jsonl(easy_path, obj)
                removed += 1
            if done % 20 == 0:
                print(f"done={done} kept={kept} removed={removed}", flush=True)

    if not args.no_parquet:
        parquet_dir = os.path.join(args.out_dir, f"{args.model_tag}_parquet")
        train_path, val_path, n_train, n_val = write_parquet_split(keep_path, parquet_dir, args.seed)
        print(json.dumps({"keep_jsonl": keep_path, "easy_jsonl": easy_path, "train": train_path, "val": val_path, "n_train": n_train, "n_val": n_val}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
