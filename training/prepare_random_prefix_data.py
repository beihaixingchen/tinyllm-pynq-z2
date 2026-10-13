import argparse
import hashlib
import json
import random
import sys
from array import array
from collections import Counter
from pathlib import Path

from prepare_story_pairs import (
    ASCII_WORD_RE,
    CJK_RE,
    NEGATIVE_PROMPT_TERMS,
    normalize_story,
    split_sentences,
)
from tokenizer import Tokenizer


BUFFER_LIMIT = 1 << 20


def flush(buffer, stream):
    if buffer:
        buffer.tofile(stream)
        buffer = array(buffer.typecode)
    return buffer


def main():
    parser = argparse.ArgumentParser(description="Build deterministic random-prefix data with continuation-only targets")
    parser.add_argument("--input-dir", type=Path, default=Path("data/TinyStories_all_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/story_pairs_masked_v2"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tok4096.model"))
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--min-cjk-chars", type=int, default=80)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.65)
    parser.add_argument("--min-sentences", type=int, default=3)
    parser.add_argument("--min-continuation-chars", type=int, default=30)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--eval-prompt-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--source-filter", choices=("GPT-3.5", "GPT-4"))
    parser.add_argument("--max-prefix-sentences", type=int, help="Choose the prefix boundary only from the first N sentences")
    parser.add_argument("--limit", type=int, help="Process only the first N records for a smoke test")
    args = parser.parse_args()

    if sys.byteorder != "little":
        raise RuntimeError("The trainer expects little-endian uint16 token files")
    if not 0 < args.validation_percent < 50:
        parser.error("validation-percent must be between 1 and 49")
    if args.max_seq_len < 8 or args.eval_prompt_count < 1:
        parser.error("max-seq-len must be at least 8 and eval-prompt-count must be positive")
    if args.max_prefix_sentences is not None and args.max_prefix_sentences < 1:
        parser.error("max-prefix-sentences must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")

    shards = sorted(args.input_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No JSONL shards in {args.input_dir}")
    tokenizer = Tokenizer(str(args.tokenizer))
    if tokenizer.eos_id < 0 or tokenizer.bos_id < 0:
        raise RuntimeError("Tokenizer must define BOS and EOS IDs")

    token_dir = args.output_dir / "tok4096"
    mask_dir = args.output_dir / "tok4096_mask"
    token_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    token_paths = {
        "train": token_dir / "data01_train.bin",
        "validation": token_dir / "data00_val.bin",
    }
    mask_paths = {
        "train": mask_dir / "data01_train.bin",
        "validation": mask_dir / "data00_val.bin",
    }

    stats = Counter()
    source_counts = Counter()
    prefix_boundary_counts = Counter()
    seen_hashes = set()
    eval_prompts = []
    prompt_seen = set()
    token_lengths = 0
    supervised_lengths = 0
    max_token_length = 0
    token_buffers = {split: array("H") for split in token_paths}
    mask_buffers = {split: array("B") for split in mask_paths}

    with token_paths["validation"].open("wb") as val_tokens, \
        token_paths["train"].open("wb") as train_tokens, \
        mask_paths["validation"].open("wb") as val_masks, \
        mask_paths["train"].open("wb") as train_masks:
        token_streams = {"train": train_tokens, "validation": val_tokens}
        mask_streams = {"train": train_masks, "validation": val_masks}
        stop = False
        for shard in shards:
            with shard.open("r", encoding="utf-8") as source:
                for line in source:
                    if args.limit is not None and stats["seen"] >= args.limit:
                        stop = True
                        break
                    stats["seen"] += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        stats["reject_invalid_json"] += 1
                        continue
                    if not isinstance(record, dict):
                        stats["reject_invalid_record"] += 1
                        continue

                    source = str(record.get("source", "unknown"))
                    if args.source_filter and source.casefold() != args.source_filter.casefold():
                        stats["reject_source_filter"] += 1
                        continue

                    instruction = record.get("instruction", {})
                    instruction_prompt = instruction.get("prompt:", "") if isinstance(instruction, dict) else ""
                    instruction_prompt = str(instruction_prompt).casefold()
                    if any(term in instruction_prompt for term in NEGATIVE_PROMPT_TERMS):
                        stats["reject_requested_negative"] += 1
                        continue

                    text = normalize_story(record.get("story_zh"))
                    if not text:
                        stats["reject_empty"] += 1
                        continue
                    cjk_count = len(CJK_RE.findall(text))
                    ascii_count = len(ASCII_WORD_RE.findall(text))
                    if cjk_count < args.min_cjk_chars:
                        stats["reject_short"] += 1
                        continue
                    if cjk_count / max(1, cjk_count + ascii_count) < args.min_cjk_ratio:
                        stats["reject_language_ratio"] += 1
                        continue

                    sentences = split_sentences(text)
                    if len(sentences) < args.min_sentences:
                        stats["reject_sentence_count"] += 1
                        continue
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    if digest in seen_hashes:
                        stats["reject_duplicate"] += 1
                        continue
                    seen_hashes.add(digest)

                    candidates = []
                    boundary_limit = len(sentences)
                    if args.max_prefix_sentences is not None:
                        boundary_limit = min(boundary_limit, args.max_prefix_sentences + 1)
                    for boundary in range(1, boundary_limit):
                        opening = " ".join(sentences[:boundary]).strip()
                        continuation = " ".join(sentences[boundary:]).strip()
                        if len(continuation) >= args.min_continuation_chars:
                            candidates.append((boundary, opening, continuation))
                    if not candidates:
                        stats["reject_short_continuation"] += 1
                        continue

                    rng = random.Random(args.seed ^ int.from_bytes(digest[:8], "big"))
                    boundary, opening, continuation = rng.choice(candidates)
                    prompt = f"故事开头：{opening}\n续写："
                    prompt_tokens = tokenizer.encode(prompt, bos=False, eos=False)
                    continuation_tokens = tokenizer.encode(continuation, bos=False, eos=False)
                    token_count = 1 + len(prompt_tokens) + len(continuation_tokens) + 1
                    if token_count > args.max_seq_len:
                        stats["reject_over_context"] += 1
                        continue

                    tokens = [tokenizer.bos_id, *prompt_tokens, *continuation_tokens, tokenizer.eos_id]
                    mask_start = 1 + len(prompt_tokens)
                    loss_mask = [0] * mask_start + [1] * (len(continuation_tokens) + 1)
                    if len(tokens) != len(loss_mask):
                        raise AssertionError("token and loss-mask lengths differ")
                    if max(tokens) > 65535:
                        raise RuntimeError("Tokenizer produced an ID outside uint16 range")

                    split = "validation" if int.from_bytes(digest[:8], "big") % 100 < args.validation_percent else "train"
                    token_buffers[split].extend(tokens)
                    mask_buffers[split].extend(loss_mask)
                    stats[f"{split}_examples"] += 1
                    stats[f"{split}_tokens"] += len(tokens)
                    stats[f"{split}_supervised_tokens"] += sum(loss_mask)
                    stats["kept_examples"] += 1
                    source_counts[source] += 1
                    prefix_boundary_counts[boundary] += 1
                    token_lengths += len(tokens)
                    supervised_lengths += sum(loss_mask)
                    max_token_length = max(max_token_length, len(tokens))

                    if split == "validation" and prompt not in prompt_seen:
                        prompt_seen.add(prompt)
                        eval_prompts.append({
                            "sha256": digest.hex(),
                            "prefix_boundary": boundary,
                            "source": source,
                            "prompt": prompt,
                            "reference": continuation,
                        })
                        if len(eval_prompts) > args.eval_prompt_count:
                            eval_prompts.sort(key=lambda item: item["sha256"])
                            removed = eval_prompts.pop()
                            prompt_seen.remove(removed["prompt"])

                    if len(token_buffers[split]) >= BUFFER_LIMIT:
                        token_buffers[split] = flush(token_buffers[split], token_streams[split])
                        mask_buffers[split] = flush(mask_buffers[split], mask_streams[split])
                    if stats["seen"] % 100000 == 0:
                        print(
                            f"processed={stats['seen']:,} kept={stats['kept_examples']:,} "
                            f"duplicates={stats['reject_duplicate']:,}",
                            flush=True,
                        )

            if stop:
                break

        for split in token_paths:
            flush(token_buffers[split], token_streams[split])
            flush(mask_buffers[split], mask_streams[split])

    eval_prompts.sort(key=lambda item: item["sha256"])
    prompt_path = args.output_dir / "eval_prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8", newline="\n") as target:
        for item in eval_prompts:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "complete": args.limit is None,
        "source_shards": len(shards),
        "input_dir": str(args.input_dir),
        "tokenizer": str(args.tokenizer),
        "seed": args.seed,
        "source_filter": args.source_filter,
        "max_prefix_sentences": args.max_prefix_sentences,
        "prefix_policy": (
            "random eligible interior boundary"
            if args.max_prefix_sentences is None
            else f"random eligible boundary among first {args.max_prefix_sentences} sentences"
        ),
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "max_seq_len": args.max_seq_len,
        "target_policy": "continuation tokens plus EOS; prompt and BOS labels ignored",
        "filters": {
            "min_cjk_chars": args.min_cjk_chars,
            "min_cjk_ratio": args.min_cjk_ratio,
            "min_sentences": args.min_sentences,
            "min_continuation_chars": args.min_continuation_chars,
            "excluded_instruction_terms": list(NEGATIVE_PROMPT_TERMS),
            "validation_percent": args.validation_percent,
        },
        "mean_tokens_per_kept_example": token_lengths / max(1, stats["kept_examples"]),
        "mean_supervised_tokens_per_kept_example": supervised_lengths / max(1, stats["kept_examples"]),
        "max_tokens_per_example": max_token_length,
        "counts": dict(stats),
        "kept_source_counts": dict(source_counts),
        "prefix_boundary_counts": dict(sorted(prefix_boundary_counts.items())),
        "files": {
            "validation_bin": str(token_paths["validation"]),
            "train_bin": str(token_paths["train"]),
            "validation_mask": str(mask_paths["validation"]),
            "train_mask": str(mask_paths["train"]),
            "eval_prompts": str(prompt_path),
            "validation_bytes": token_paths["validation"].stat().st_size,
            "train_bytes": token_paths["train"].stat().st_size,
            "validation_mask_bytes": mask_paths["validation"].stat().st_size,
            "train_mask_bytes": mask_paths["train"].stat().st_size,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
