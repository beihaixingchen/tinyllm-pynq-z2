import argparse
import hashlib
import json
import re
import sys
import unicodedata
from array import array
from collections import Counter
from pathlib import Path

from tokenizer import Tokenizer


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
ASCII_WORD_RE = re.compile(r"[A-Za-z0-9]")
SENTENCE_END_RE = re.compile(r"[。！？!?]+[\"'”’」』】）)]*")
NEGATIVE_PROMPT_TERMS = ("bad ending", "sad ending", "unhappy ending", "tragic ending", "scary story", "horror")
BUFFER_LIMIT = 1 << 20


def normalize_story(value):
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def split_sentences(text):
    sentences = []
    start = 0
    for match in SENTENCE_END_RE.finditer(text):
        sentence = text[start : match.end()].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def flush(buffer, stream):
    if buffer:
        buffer.tofile(stream)
        buffer = array("H")
    return buffer


def main():
    parser = argparse.ArgumentParser(description="Filter, format, split, and tokenize Chinese story continuations")
    parser.add_argument("--input-dir", type=Path, default=Path("data/TinyStories_all_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/story_pairs_eos_v1"))
    parser.add_argument("--tokenizer", type=Path, default=Path("data/tok4096.model"))
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--min-cjk-chars", type=int, default=80)
    parser.add_argument("--min-cjk-ratio", type=float, default=0.65)
    parser.add_argument("--min-sentences", type=int, default=3)
    parser.add_argument("--min-continuation-chars", type=int, default=30)
    parser.add_argument("--validation-percent", type=int, default=5)
    parser.add_argument("--eval-prompt-count", type=int, default=32)
    parser.add_argument("--limit", type=int, help="Process only the first N records for a smoke test")
    args = parser.parse_args()

    if sys.byteorder != "little":
        raise RuntimeError("The trainer expects little-endian uint16 token files")
    if not 0 < args.validation_percent < 50:
        parser.error("validation-percent must be between 1 and 49")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")

    shards = sorted(args.input_dir.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(f"No JSONL shards in {args.input_dir}")
    tokenizer = Tokenizer(str(args.tokenizer))
    if tokenizer.eos_id < 0 or tokenizer.bos_id < 0:
        raise RuntimeError("Tokenizer must define BOS and EOS IDs")

    bin_dir = args.output_dir / "tok4096"
    bin_dir.mkdir(parents=True)
    val_path = bin_dir / "data00_val.bin"
    train_path = bin_dir / "data01_train.bin"
    stats = Counter()
    source_counts = Counter()
    seen_hashes = set()
    eval_prompts = []
    token_lengths = 0
    max_token_length = 0
    train_buffer = array("H")
    val_buffer = array("H")

    with val_path.open("wb") as val_file, train_path.open("wb") as train_file:
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
                    opening = sentences[0]
                    continuation = " ".join(sentences[1:]).strip()
                    if len(continuation) < args.min_continuation_chars:
                        stats["reject_short_continuation"] += 1
                        continue

                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    if digest in seen_hashes:
                        stats["reject_duplicate"] += 1
                        continue
                    seen_hashes.add(digest)

                    prompt = f"故事开头：{opening}\n续写："
                    formatted = prompt + continuation
                    tokens = tokenizer.encode(formatted, bos=True, eos=True)
                    if len(tokens) > args.max_seq_len:
                        stats["reject_over_context"] += 1
                        continue
                    if max(tokens) > 65535:
                        raise RuntimeError("Tokenizer produced an ID outside uint16 range")

                    is_validation = int.from_bytes(digest[:8], "big") % 100 < args.validation_percent
                    if is_validation:
                        val_buffer.extend(tokens)
                        stats["validation_examples"] += 1
                        stats["validation_tokens"] += len(tokens)
                        if len(eval_prompts) < args.eval_prompt_count:
                            eval_prompts.append({
                                "sha256": digest.hex(),
                                "prompt": prompt,
                                "reference": continuation,
                            })
                        if len(val_buffer) >= BUFFER_LIMIT:
                            val_buffer = flush(val_buffer, val_file)
                    else:
                        train_buffer.extend(tokens)
                        stats["train_examples"] += 1
                        stats["train_tokens"] += len(tokens)
                        if len(train_buffer) >= BUFFER_LIMIT:
                            train_buffer = flush(train_buffer, train_file)

                    stats["kept_examples"] += 1
                    source_counts[str(record.get("source", "unknown"))] += 1
                    token_lengths += len(tokens)
                    max_token_length = max(max_token_length, len(tokens))
                    if stats["seen"] % 100000 == 0:
                        print(f"processed={stats['seen']:,} kept={stats['kept_examples']:,} duplicates={stats['reject_duplicate']:,}", flush=True)

            if stop:
                break

        flush(val_buffer, val_file)
        flush(train_buffer, train_file)

    prompt_path = args.output_dir / "eval_prompts.jsonl"
    with prompt_path.open("w", encoding="utf-8", newline="\n") as target:
        for item in eval_prompts:
            target.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "complete": args.limit is None,
        "source_shards": len(shards),
        "input_dir": str(args.input_dir),
        "tokenizer": str(args.tokenizer),
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
        "max_seq_len": args.max_seq_len,
        "filters": {
            "min_cjk_chars": args.min_cjk_chars,
            "min_cjk_ratio": args.min_cjk_ratio,
            "min_sentences": args.min_sentences,
            "min_continuation_chars": args.min_continuation_chars,
            "excluded_instruction_terms": list(NEGATIVE_PROMPT_TERMS),
            "validation_percent": args.validation_percent,
        },
        "mean_tokens_per_kept_example": token_lengths / max(1, stats["kept_examples"]),
        "max_tokens_per_example": max_token_length,
        "counts": dict(stats),
        "kept_source_counts": dict(source_counts),
        "files": {
            "validation_bin": str(val_path),
            "train_bin": str(train_path),
            "eval_prompts": str(prompt_path),
            "validation_bytes": val_path.stat().st_size,
            "train_bytes": train_path.stat().st_size,
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
