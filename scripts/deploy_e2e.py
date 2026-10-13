import argparse
import getpass
import json
import os
import posixpath
import shlex
import subprocess
import sys
from pathlib import Path

import paramiko


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Upload and run the prepared Q6 TinyLLM experiment on PYNQ-Z2")
    parser.add_argument("--package-dir", type=Path, default=project_dir / "build" / "e2e")
    parser.add_argument("--model-repo", type=Path,
                        default=Path(os.environ.get("TINYLLM_MODEL_REPO", project_dir.parent.parent / "TinyLLMStory-zh")))
    parser.add_argument("--host", default=os.environ.get("PYNQ_HOST"),
                        help="board address; defaults to PYNQ_HOST")
    parser.add_argument("--user", default=os.environ.get("PYNQ_USER", "xilinx"))
    parser.add_argument("--remote-dir", default="/home/xilinx/q6w_gemv/e2e")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    vocab_group = parser.add_mutually_exclusive_group()
    vocab_group.add_argument("--neon-vocab", action="store_true")
    vocab_group.add_argument("--neon-vocab-pairrows", action="store_true")
    vocab_group.add_argument("--neon-vocab-fourrows", action="store_true")
    vocab_group.add_argument("--neon-vocab-q6-int8", action="store_true")
    parser.add_argument("--neon-attention", action="store_true")
    parser.add_argument("--neon-rmsnorm", action="store_true")
    parser.add_argument("--fuse-projections", action="store_true")
    args = parser.parse_args()
    if not args.host:
        parser.error("set PYNQ_HOST or pass --host")

    if not args.remote_dir.startswith("/home/xilinx/") or any(char in args.remote_dir for char in " ';&|\n\r"):
        raise ValueError("remote-dir must be a simple path beneath /home/xilinx")
    package_dir = args.package_dir.resolve()
    request = json.loads((package_dir / "request.json").read_text(encoding="utf-8"))
    neon_sources = []
    if args.neon_vocab or args.neon_vocab_pairrows or args.neon_vocab_fourrows:
        neon_sources.append(project_dir / "scripts" / "neon_sgemv.c")
    if args.neon_vocab_q6_int8:
        neon_sources.append(project_dir / "scripts" / "neon_q6_vocab.c")
    if args.neon_attention:
        neon_sources.append(project_dir / "scripts" / "neon_gqa_attention.c")
    if args.neon_rmsnorm:
        neon_sources.append(project_dir / "scripts" / "neon_rmsnorm.c")
    for source in neon_sources:
        if not source.is_file():
            raise FileNotFoundError("NEON source is missing: {}".format(source))
    required = (
        "request.json",
        "reference_logits.bin",
        "model_q6w.bin",
        "manifest.json",
    )
    if request.get("linear_mode") != "ps":
        required += ("q6w_gemv.bit", "q6w_gemv.hwh")
    for name in required:
        if not (package_dir / name).is_file():
            raise FileNotFoundError("Prepared package is missing {}".format(package_dir / name))

    password = os.environ.get("PYNQ_PASSWORD") or getpass.getpass("PYNQ password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=15)
    try:
        command = "mkdir -p {}".format(args.remote_dir)
        _stdin, stdout, stderr = client.exec_command(command)
        if stdout.channel.recv_exit_status() != 0:
            raise RuntimeError(stderr.read().decode("utf-8", errors="replace"))

        runtime = project_dir / "board_e2e.py"
        with client.open_sftp() as sftp:
            for name in required:
                sftp.put(str(package_dir / name), posixpath.join(args.remote_dir, name))
            sftp.put(str(runtime), posixpath.join(args.remote_dir, "board_e2e.py"))
            for source in neon_sources:
                sftp.put(str(source), posixpath.join(args.remote_dir, source.name))

        if neon_sources:
            remote_sources = [posixpath.join(args.remote_dir, source.name) for source in neon_sources]
            remote_library = posixpath.join(args.remote_dir, "neon_kernels.so")
            command = "gcc -shared -fPIC -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard {} -lm -o {}".format(
                " ".join(shlex.quote(source) for source in remote_sources),
                shlex.quote(remote_library),
            )
            _stdin, stdout, stderr = client.exec_command(command)
            compile_output = stdout.read().decode("utf-8", errors="replace")
            compile_errors = stderr.read().decode("utf-8", errors="replace")
            compile_status = stdout.channel.recv_exit_status()
            if compile_status != 0:
                raise RuntimeError("PYNQ NEON kernel compile failed:\n{}{}".format(
                    compile_output, compile_errors
                ))

        result_path = posixpath.join(args.remote_dir, "board_result.json")
        runtime_path = posixpath.join(args.remote_dir, "board_e2e.py")
        command = "sudo -S python3 {} --app-dir {} --runs {} --warmup {} --output {}".format(
            shlex.quote(runtime_path), shlex.quote(args.remote_dir), args.runs,
            args.warmup, shlex.quote(result_path)
        )
        if args.neon_vocab:
            command += " --neon-vocab"
        if args.neon_vocab_pairrows:
            command += " --neon-vocab-pairrows"
        if args.neon_vocab_fourrows:
            command += " --neon-vocab-fourrows"
        if args.neon_vocab_q6_int8:
            command += " --neon-vocab-q6-int8"
        if args.neon_attention:
            command += " --neon-attention"
        if args.neon_rmsnorm:
            command += " --neon-rmsnorm"
        if args.fuse_projections:
            command += " --fuse-projections"
        stdin, stdout, stderr = client.exec_command(command)
        stdin.write(password + "\n")
        stdin.flush()
        stdin.channel.shutdown_write()
        output = stdout.read().decode("utf-8", errors="replace")
        errors = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        safe_log = (output + errors).replace(password, "<redacted>")
        (package_dir / "board_stdout.log").write_text(safe_log, encoding="utf-8")
        if status != 0:
            raise RuntimeError("PYNQ run failed with exit status {}:\n{}{}".format(status, output, errors))

        with client.open_sftp() as sftp:
            sftp.get(result_path, str(package_dir / "board_result.json"))
            remote_logits = posixpath.join(args.remote_dir, "board_logits.bin")
            try:
                sftp.get(remote_logits, str(package_dir / "board_logits.bin"))
            except OSError:
                pass
    finally:
        client.close()

    report = json.loads((package_dir / "board_result.json").read_text(encoding="utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    generated = report.get("generated_ids", [])
    if generated:
        tokenizer = args.model_repo.resolve() / "tok4096.model"
        code = "import sys,json; from sentencepiece import SentencePieceProcessor; p=SentencePieceProcessor(model_file=sys.argv[1]); print(p.decode(json.loads(sys.argv[2])))"
        decoded = subprocess.run(
            [sys.executable, "-c", code, str(tokenizer), json.dumps(generated)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("Decoded continuation: {}".format(decoded.stdout.strip()))


if __name__ == "__main__":
    main()
