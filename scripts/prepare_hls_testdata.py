import argparse
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prepare the autonomous-HLS C-simulation inputs")
    parser.add_argument("--package-dir", type=Path, required=True)
    args = parser.parse_args()
    package = args.package_dir.resolve()
    request = json.loads((package / "request.json").read_text(encoding="utf-8"))
    prompt_ids = [int(value) for value in request["prompt_ids"]]
    stop_ids = [int(value) for value in request["stop_token_ids"]]
    if len(stop_ids) != 2:
        raise ValueError("The autonomous prototype expects two stop token IDs")
    with (package / "hls_prompt.bin").open("wb") as output:
        output.write(struct.pack(
            "<5i", len(prompt_ids), int(request["max_new_tokens"]),
            int(request["min_new_tokens"]), stop_ids[0], stop_ids[1],
        ))
        output.write(struct.pack("<{}i".format(len(prompt_ids)), *prompt_ids))
    print(json.dumps({
        "package": str(package),
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": request["max_new_tokens"],
        "reference_steps": request["reference_step_count"],
        "hls_prompt_bytes": (package / "hls_prompt.bin").stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
