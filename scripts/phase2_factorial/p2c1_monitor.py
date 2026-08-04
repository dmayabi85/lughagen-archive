#!/usr/bin/env python3
"""Real-time monitor for LughaGen Phase 2 cell RUN-P2-C01-S1.

Single-page dashboard. Refreshes every 5 sec. Ctrl-C to exit.
Training continues regardless of monitor state.

Usage:
  python3 /workspace/scripts/p2c1_monitor.py
  python3 /workspace/scripts/p2c1_monitor.py --refresh 10
"""
import argparse, json, math, re, subprocess, sys, time
from pathlib import Path

RUN_ID = "RUN-P2-C02-S2"
TMUX_SESSION = "lughagen-sweep"
RUN_DIR = Path(f"/workspace/checkpoints/{RUN_ID}")
LOG_GLOB = f"phase2_{RUN_ID}_*.log"

# Regex for log parsing
LOSS_RE = re.compile(r"\{'loss': '?([\d.eE+-]+)'?, 'grad_norm': '?([\d.eE+-]+)'?, 'learning_rate': '?([\d.eE+-]+)'?, 'epoch': '?([\d.eE+-]+)'?\}")
EVAL_RE = re.compile(r"\{'eval_loss': '?([\d.eE+-]+)'?, 'eval_runtime': '?([\d.eE+-]+)'?, 'eval_samples_per_second': '?([\d.eE+-]+)'?")
PROG_RE = re.compile(r"(\d+)/(\d+)\s*\[(\d+:\d+:\d+|\d+:\d+)<(\d+:\d+:\d+|\d+:\d+),\s+([\d.]+)s/it\]")
MFU_INIT_RE = re.compile(r"\[mfu\] init: n_params=([\d,]+) flops/step=([\d.eE+-]+) peak_aggregate=([\d.]+) TFLOPS/s world_size=(\d+)")


def find_latest_log():
    cands = sorted(
        Path("/workspace/logs").glob(LOG_GLOB),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    return cands[0] if cands else None


def parse_log(log_path):
    if not log_path or not log_path.exists():
        return None
    losses, evals, progress, mfu_init = [], [], None, None
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 2 * 1024 * 1024))  # last 2MB
        if size > 2 * 1024 * 1024:
            f.readline()
        text = f.read().decode("utf-8", errors="replace")
    for line in text.split("\n"):
        if m := LOSS_RE.search(line):
            try:
                losses.append({
                    "loss": float(m.group(1)),
                    "grad": float(m.group(2)),
                    "lr": float(m.group(3)),
                    "epoch": float(m.group(4)),
                })
            except ValueError: pass
        if m := EVAL_RE.search(line):
            try:
                evals.append({
                    "loss": float(m.group(1)),
                    "runtime": float(m.group(2)),
                    "sps": float(m.group(3)),
                })
            except ValueError: pass
        for m in PROG_RE.finditer(line):
            try:
                progress = {
                    "step": int(m.group(1)),
                    "max": int(m.group(2)),
                    "elapsed": m.group(3),
                    "eta": m.group(4),
                    "sec_per_step": float(m.group(5)),
                }
            except ValueError: pass
        if m := MFU_INIT_RE.search(line):
            try:
                mfu_init = {
                    "n_params": int(m.group(1).replace(",", "")),
                    "flops_per_step": float(m.group(2)),
                    "peak_tflops": float(m.group(3)),
                    "world_size": int(m.group(4)),
                }
            except ValueError: pass
    return {
        "losses": losses,
        "evals": evals,
        "progress": progress,
        "mfu_init": mfu_init,
    }


def parse_mfu_log():
    """Read mfu_log.jsonl entries written by MFUCallback."""
    mfu_path = RUN_DIR / "mfu_log.jsonl"
    if not mfu_path.exists():
        return []
    records = []
    try:
        with open(mfu_path) as f:
            for line in f:
                try: records.append(json.loads(line.strip()))
                except json.JSONDecodeError: pass
    except Exception: pass
    return records


def parse_per_lang():
    p = RUN_DIR / "per_language_grad_norms.jsonl"
    if not p.exists():
        return []
    out = []
    try:
        with open(p) as f:
            for line in f:
                try: out.append(json.loads(line.strip()))
                except json.JSONDecodeError: pass
    except Exception: pass
    return out[-5:]


def gpu_status():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode != 0: return None
        out = []
        for line in r.stdout.strip().split("\n"):
            p = [x.strip() for x in line.split(",")]
            if len(p) == 3:
                out.append({"idx": int(p[0]), "util": int(p[1]), "mem": int(p[2])})
        return out
    except Exception:
        return None


def disk_status():
    try:
        r = subprocess.run(["df", "-BG", "/", "/data", "/workspace"],
                          capture_output=True, text=True, timeout=5)
        if r.returncode != 0: return {}
        out = {}
        for line in r.stdout.strip().split("\n")[1:]:
            p = line.split()
            if len(p) >= 6:
                out[p[5]] = {"used": p[2].rstrip("G"), "avail": p[3].rstrip("G"),
                            "pct": int(p[4].rstrip("%"))}
        return out
    except Exception:
        return {}


def tmux_alive():
    try:
        return subprocess.run(["tmux", "has-session", "-t", TMUX_SESSION],
                            capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False


def fmt(s, w):
    if len(s) > w: return s[:w-3] + "..."
    return s.ljust(w)


def render(data, mfu_records, per_lang, gpus, disks, log_path, refresh):
    W = 80
    out = []
    bar = "─" * (W - 2)
    nl = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # ── Header ──
    out.append(f"┌{bar}┐")
    out.append(f"│ {fmt(f'LughaGen Phase 2 Cell 1 Monitor  ·  {nl}', W - 4)} │")
    out.append(f"├{bar}┤")
    out.append(f"│ {fmt(f'Run:  {RUN_ID}', W - 4)} │")
    sess = "ALIVE" if tmux_alive() else "DEAD"
    out.append(f"│ {fmt(f'tmux: {TMUX_SESSION} ({sess})', W - 4)} │")
    log_str = str(log_path)[-50:] if log_path else "(none)"
    out.append(f"│ {fmt(f'log:  ...{log_str}', W - 4)} │")
    out.append(f"├{bar}┤")

    # ── MFU init ──
    if data and data["mfu_init"]:
        m = data["mfu_init"]
        out.append(f"│ {fmt(f'MFU init: n_params={m['n_params']:,}  world_size={m['world_size']}', W - 4)} │")
        out.append(f"│ {fmt(f'  flops/step={m['flops_per_step']:.3e}  peak={m['peak_tflops']:.1f} TFLOPS/s', W - 4)} │")
    else:
        out.append(f"│ {fmt('MFU init: (not yet — waiting for training to start)', W - 4)} │")
    out.append(f"├{bar}┤")

    # ── Progress ──
    if data and data["progress"]:
        p = data["progress"]
        pct = p["step"] / p["max"] * 100
        bw = 50
        filled = int(bw * p["step"] / p["max"])
        pbar = "█" * filled + "░" * (bw - filled)
        out.append(f"│ {fmt(f'Step {p['step']:>5} / {p['max']} ({pct:5.2f}%)', W - 4)} │")
        out.append(f"│ {fmt(f'[{pbar}]', W - 4)} │")
        out.append(f"│ {fmt(f'Elapsed {p['elapsed']}  ETA {p['eta']}  ({p['sec_per_step']:.2f}s/step)', W - 4)} │")
    else:
        out.append(f"│ {fmt('Progress: (still tokenizing or starting up)', W - 4)} │")
    out.append(f"├{bar}┤")

    # ── Loss + grad ──
    if data and data["losses"]:
        first = data["losses"][0]["loss"]
        last = data["losses"][-1]
        drop = (first - last["loss"]) / first * 100 if first > 0 else 0
        out.append(f"│ {fmt(f'Train loss: {last['loss']:.4f}   (start {first:.3f}, ↓ {drop:.1f}%)', W - 4)} │")
        out.append(f"│ {fmt(f'grad_norm: {last['grad']:.4f}    lr: {last['lr']:.4e}', W - 4)} │")
    else:
        out.append(f"│ {fmt('Train loss: (no values yet)', W - 4)} │")
    out.append(f"├{bar}┤")

    # ── Eval ──
    if data and data["evals"]:
        ev = data["evals"][-1]
        pph = math.exp(ev["loss"]) if ev["loss"] < 100 else float("inf")
        out.append(f"│ {fmt(f'Eval loss: {ev['loss']:.4f}   (≈ perplexity {pph:.1f})', W - 4)} │")
        out.append(f"│ {fmt(f'Eval history ({len(data['evals'])}):', W - 4)} │")
        for e in data["evals"][-3:]:
            ph = math.exp(e["loss"]) if e["loss"] < 100 else float("inf")
            out.append(f"│ {fmt(f'  eval_loss={e['loss']:.4f}  pph≈{ph:.1f}', W - 4)} │")
    else:
        out.append(f"│ {fmt('Eval: (first eval at step ~50)', W - 4)} │")
    out.append(f"├{bar}┤")

    # ── MFU per-interval ──
    if mfu_records:
        out.append(f"│ {fmt(f'MFU intervals ({len(mfu_records)} entries):', W - 4)} │")
        for rec in mfu_records[-3:]:
            if rec.get("event") == "interval":
                step = rec.get("step", "?")
                mfu = rec.get("mfu", 0) * 100
                tflops = rec.get("achieved_tflops", 0)
                sps = rec.get("sec_per_step", 0)
                out.append(f"│ {fmt(f'  step {step}: MFU {mfu:.1f}%   {tflops:.0f} TFLOPS/s   {sps:.2f}s/step', W - 4)} │")
    else:
        out.append(f"│ {fmt('MFU intervals: (not yet — first at step 50)', W - 4)} │")
    out.append(f"├{bar}┤")

    # ── Per-language grad norms ──
    if per_lang:
        out.append(f"│ {fmt(f'Per-lang grad norms (last {len(per_lang)}):', W - 4)} │")
        for rec in per_lang:
            step = rec.get("step", "?")
            norms = rec.get("norms", {})
            ns = "  ".join(
                f"{lang}={(n if isinstance(n,(int,float)) else 'null')}"
                for lang, n in sorted(norms.items())
            )
            out.append(f"│ {fmt(f'  step {step}: {ns}', W - 4)} │")
    else:
        out.append(f"│ {fmt('Per-lang grad norms: (not yet)', W - 4)} │")
    out.append(f"├{bar}┤")

    # ── GPU ──
    if gpus:
        out.append(f"│ {fmt('GPUs (utilization / VRAM):', W - 4)} │")
        for i in range(0, len(gpus), 4):
            row = []
            for j in range(i, min(i + 4, len(gpus))):
                g = gpus[j]
                row.append(f"GPU{g['idx']}: {g['util']:>3}% / {g['mem']/1024:>4.1f}GB")
            out.append(f"│ {fmt('  ' + '  '.join(row), W - 4)} │")
    out.append(f"├{bar}┤")

    # ── Disk ──
    if disks:
        for mnt in ("/", "/data", "/workspace"):
            if mnt in disks:
                d = disks[mnt]
                warn = " ⚠" if d["pct"] > 85 else ""
                out.append(f"│ {fmt(f'Disk {mnt:11}: {d['used']:>5}GB used, {d['avail']:>5}GB free ({d['pct']:>3}%){warn}', W - 4)} │")

    out.append(f"├{bar}┤")
    out.append(f"│ {fmt(f'Refresh every {refresh}s. Ctrl-C to exit.', W - 4)} │")
    out.append(f"└{bar}┘")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", type=int, default=5)
    args = ap.parse_args()

    log_path = find_latest_log()
    if not log_path:
        print(f"ERROR: no log matching {LOG_GLOB} in /workspace/logs/")
        print("Is the training actually running?")
        sys.exit(1)

    print(f"Monitoring: {log_path}")
    print(f"Refreshing every {args.refresh}s. Ctrl-C to exit.")
    time.sleep(1)

    try:
        while True:
            log_path = find_latest_log()  # re-check in case of new log
            data = parse_log(log_path)
            mfu_records = parse_mfu_log()
            per_lang = parse_per_lang()
            gpus = gpu_status()
            disks = disk_status()
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(render(data, mfu_records, per_lang, gpus, disks, log_path, args.refresh))
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped. Training continues.")


if __name__ == "__main__":
    main()
