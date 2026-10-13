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
    parser = argparse.ArgumentParser(description="Deploy and run the autonomous PL model on PYNQ-Z2")
    parser.add_argument("--package-dir", type=Path, default=project_dir / "build" / "autonomous_hls" / "e2e32")
    parser.add_argument("--overlay-dir", type=Path, default=project_dir / "build" / "autonomous_overlay" / "output")
    parser.add_argument("--model-repo", type=Path,
                        default=Path(os.environ.get("TINYLLM_MODEL_REPO", project_dir)))
    parser.add_argument("--host", default=os.environ.get("PYNQ_HOST"),
                        help="board address; defaults to PYNQ_HOST")
    parser.add_argument("--user", default=os.environ.get("PYNQ_USER", "xilinx"))
    parser.add_argument("--remote-dir", default="/home/xilinx/q6w_gemv/autonomous_full_pl_50mhz")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--clock-mhz", type=float,
                         help="set PYNQ PS7 FCLK0 before loading/running the overlay")
    parser.add_argument("--bus-probe", action="store_true")
    parser.add_argument("--memory-probe", action="store_true")
    parser.add_argument("--step-trace", action="store_true")
    parser.add_argument("--smoke-one", action="store_true")
    parser.add_argument("--smoke-prefix", type=int)
    parser.add_argument("--dump-logits", action="store_true",
                         help="return full vocabulary logits for numerical validation")
    parser.add_argument("--performance-overlay", action="store_true",
                         help="mark an overlay built without trace-debug hardware")
    parser.add_argument("--legacy-kv-mirror", action="store_true",
                        help="use the legacy auxiliary-buffer KV mirror interface")
    parser.add_argument("--dual-ffn-w3", action="store_true",
                        help="supply the W3 workspace base to a separate gmem3 master")
    parser.add_argument("--precomputed-rope", action="store_true",
                        help="load the precomputed RoPE table from the package")
    parser.add_argument("--parallel-kv", action="store_true",
                        help="parallelize K/V projections using separate HP ports")
    args = parser.parse_args()
    if not args.host:
        parser.error("set PYNQ_HOST or pass --host")
    if args.parallel_kv:
        if not args.legacy_kv_mirror:
            parser.error("--parallel-kv requires --legacy-kv-mirror")
        args.dual_ffn_w3 = True
        args.precomputed_rope = True
    if args.dual_ffn_w3 and not args.legacy_kv_mirror:
        parser.error("--dual-ffn-w3 requires --legacy-kv-mirror")
    if not args.remote_dir.startswith("/home/xilinx/") or any(char in args.remote_dir for char in " ';&|\n\r"):
        raise ValueError("remote-dir must be a simple path beneath /home/xilinx")
    package_dir = args.package_dir.resolve()
    overlay_dir = args.overlay_dir.resolve()
    files = {
        "request.json": package_dir / "request.json",
        "reference_logits.bin": package_dir / "reference_logits.bin",
        "hls_prompt.bin": package_dir / "hls_prompt.bin",
        "autonomous_model.bin": package_dir / "autonomous_model.bin",
        "tinyllm_autonomous.bit": overlay_dir / "tinyllm_autonomous.bit",
        "tinyllm_autonomous.hwh": overlay_dir / "tinyllm_autonomous.hwh",
        "autonomous_board.py": project_dir / "autonomous_board.py",
    }
    if args.precomputed_rope:
        files["rope_table.bin"] = package_dir / "rope_table.bin"
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError("Required autonomous PL file is missing: {} ({})".format(name, path))

    password = os.environ.get("PYNQ_PASSWORD") or getpass.getpass("PYNQ password: ")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, username=args.user, password=password, timeout=15)
    try:
        stdin, stdout, stderr = client.exec_command("mkdir -p {}".format(shlex.quote(args.remote_dir)))
        status = stdout.channel.recv_exit_status()
        if status:
            raise RuntimeError(stderr.read().decode("utf-8", errors="replace"))
        with client.open_sftp() as sftp:
            for name, path in files.items():
                sftp.put(str(path), posixpath.join(args.remote_dir, name))

        report_path = posixpath.join(args.remote_dir, "board_result.json")
        runtime_path = posixpath.join(args.remote_dir, "autonomous_board.py")
        command = "sudo -S python3 {} --app-dir {} --output {} --timeout {} --runs {}".format(
            shlex.quote(runtime_path), shlex.quote(args.remote_dir), shlex.quote(report_path),
            args.timeout, args.runs,
        )
        if args.clock_mhz is not None:
            command += " --clock-mhz {}".format(args.clock_mhz)
        if args.bus_probe:
            command += " --bus-probe"
        if args.memory_probe:
            command += " --memory-probe"
        if args.step_trace:
            command += " --step-trace"
        if args.smoke_one:
            command += " --smoke-one"
        if args.smoke_prefix is not None:
            command += " --smoke-prefix {}".format(args.smoke_prefix)
        if args.dump_logits:
            command += " --dump-logits"
        if args.performance_overlay:
            command += " --performance-overlay"
        if args.legacy_kv_mirror:
            command += " --legacy-kv-mirror"
        if args.dual_ffn_w3:
            command += " --dual-ffn-w3"
        if args.precomputed_rope:
            command += " --precomputed-rope"
        if args.parallel_kv:
            command += " --parallel-kv"
        stdin, stdout, stderr = client.exec_command(command, timeout=args.timeout + 60)
        stdin.write(password + "\n")
        stdin.flush()
        stdin.channel.shutdown_write()
        output = stdout.read().decode("utf-8", errors="replace")
        errors = stderr.read().decode("utf-8", errors="replace")
        status = stdout.channel.recv_exit_status()
        safe_log = (output + errors).replace(password, "<redacted>")
        (package_dir / "board_stdout.log").write_text(safe_log, encoding="utf-8")
        if status:
            raise RuntimeError("Autonomous PL board run failed with status {}:\n{}{}".format(status, output, errors))
        with client.open_sftp() as sftp:
            sftp.get(report_path, str(package_dir / "board_result.json"))
            if args.dump_logits:
                try:
                    sftp.get(posixpath.join(args.remote_dir, "board_logits.bin"),
                             str(package_dir / "board_logits.bin"))
                except OSError:
                    pass
            for name in ("board_trace.bin", "board_trace_logits.bin"):
                try:
                    sftp.get(posixpath.join(args.remote_dir, name),
                             str(package_dir / name))
                except OSError:
                    pass
    finally:
        client.close()

    report = json.loads((package_dir / "board_result.json").read_text(encoding="utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    generated = report.get("last_run", {}).get("generated_ids", [])
    tokenizer = args.model_repo.resolve() / "tok4096.model"
    if generated and tokenizer.is_file():
        code = (
            "import sys,json; from sentencepiece import SentencePieceProcessor; "
            "p=SentencePieceProcessor(model_file=sys.argv[1]); "
            "print(p.decode(json.loads(sys.argv[2])))"
        )
        decoded = subprocess.run(
            [sys.executable, "-c", code,
             str(tokenizer), json.dumps(generated)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        print("Decoded continuation: {}".format(decoded.stdout.strip()))


if __name__ == "__main__":
    main()
