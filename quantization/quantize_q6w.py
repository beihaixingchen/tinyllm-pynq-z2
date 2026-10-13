import argparse
import json
from pathlib import Path

from q6w_format import export_q6w


def main():
    parser = argparse.ArgumentParser(description="Export TinyLLMStory-zh as row-wise packed signed-int6 weights")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_q6w(args.checkpoint, args.output, args.manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(args.manifest),
                "bytes": manifest["binary_bytes"],
                "sha256": manifest["binary_sha256"],
                "source_step": manifest["source_step"],
                "q6_values": manifest["q6_value_count"],
                "fp32_norm_values": manifest["fp32_norm_value_count"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
