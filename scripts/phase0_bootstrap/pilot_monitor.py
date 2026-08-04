#!/usr/bin/env python3
"""Real-time monitor for LughaGen pilot v2-A (or v2-D).

Usage:
  python3 pilot_monitor.py                # auto-detects latest RUN-V2A-* log
  python3 pilot_monitor.py --run v2D      # monitor v2-D
  python3 pilot_monitor.py --refresh 10   # 10-sec refresh
  Ctrl-C to exit.
"""
import argparse, json, math, re, subprocess, sys, time
from pathlib import Path

LOSS_RE = re.compile(r"\{'loss': '([\d.]+)', 'grad_norm': '([\d.]+)', 'learning_rate': '([\d.eE+-]+)', 'epoch': '([\d.eE+-]+)'\}")
EVAL_RE = re.compile(r"\{'eval_loss': '([\d.]+)', 'eval_runtime': '([\d.]+)', 'eval_samples_per_second': '([\d.]+)'")
PROG_RE = re.compile(r"(\d+)/(\d+)\s*\[(\d+:\d+:\d+|\d+:\d+)<(\d+:\d+:\d+|\d+:\d+),\s+([\d.]+)s/it\]")


def find_latest(glob_pattern, root):
    cands = sorted(Path(root).glob(glob_pattern),
                   key=lambda p: p.stat().st_mtime if p.exists() else 0,
                   reverse=True)
    return cands[0] if cands else None


def parse_log(log_path):
    if not log_path or not log_path.exists():
        return None
    losses, evals, progress = [], [], None
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 1024 * 1024))  # last 1MB
        if size > 1024 * 1024:
            f.readline()
        text = f.read().decode("utf-8", errors="replace")
    for line in text.split("\n"):
        m = LOSS_RE.search(line)
        if m:
            try:
                losses.append({"loss": float(m.group(1)), "grad": float(m.group(2)),
                               "lr": float(m.group(3)), "epoch": float(m.group(4))})
            except ValueError:
                pass
        m = EVAL_RE.search(line)
        if m:
            try:
                evals.append({"loss": float(m.group(1)), "runtime": float(m.group(2)),
                              "sps": float(m.group(3))})
            except ValueError:
                pass
        for m in PROG_RE.finditer(line):
            try:
                progress = {"step": int(m.group(1)), "max": int(m.group(2)),
                            "elapsed": m.group(3), "eta": m.group(4),
                            "sec_per_step": float(m.group(5))}
            except ValueError:
                pass
    return {"losses": losses, "evals": evals, "progress": progress}


def parse_per_lang(jsonl_path):
    if not jsonl_path or not jsonl_path.exists():
        return []
    out = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                out.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                pass
    return out[-5:]


def gpu_status():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
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
        r = subprocess.run(["df", "-BG", "/", "/data"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return {}
        out = {}
        for line in r.stdout.strip().split("\n")[1:]:
            p = line.split()
            if len(p) >= 6:
                out[p[5]] = {"used": p[2].rstrip("G"), "avail": p[3].rstrip("G"),
                             "pct": int(p[4].rstrip("%"))}
        return out
    except Exception:
        return {}


def tmux_alive(session):
    try:
        return subprocess.run(["tmux", "has-session", "-t", session],
                              capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False


def fmt(s, w):
    """Truncate or pad string to width w."""
    s = str(s)
    if len(s) > w:
        return s[:w - 3] + "..."
    return s.ljust(w)


def render(data, per_lang, gpus, disks, run_dir, log_path, tmux_sess, refresh):
    W = 76
    out = []
    bar = "-" * (W - 2)
    nl = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    out.append("+" + bar + "+")
    header = "LughaGen Pilot Monitor   " + nl
    out.append("| " + fmt(header, W - 4) + " |")
    out.append("+" + bar + "+")

    run_name = run_dir.name if run_dir else "(none)"
    out.append("| " + fmt("Run: " + run_name, W - 4) + " |")
    alive = tmux_alive(tmux_sess)
    sess_text = "tmux '" + tmux_sess + "': " + ("ALIVE" if alive else "DEAD")
    out.append("| " + fmt(sess_text, W - 4) + " |")
    out.append("+" + bar + "+")

    # Progress
    if data and data["progress"]:
        p = data["progress"]
        pct = p["step"] / p["max"] * 100
        bar_w = 40
        filled = int(bar_w * p["step"] / p["max"])
        progress_bar = "#" * filled + "-" * (bar_w - filled)
        step_str = "Step %5d / %d (%.2f%%)" % (p["step"], p["max"], pct)
        out.append("| " + fmt("Progress: " + step_str, W - 4) + " |")
        out.append("| " + fmt("[" + progress_bar + "]", W - 4) + " |")
        rate_str = "Elapsed %s  ETA %s  (%.2fs/step)" % (
            p["elapsed"], p["eta"], p["sec_per_step"])
        out.append("| " + fmt(rate_str, W - 4) + " |")
    else:
        out.append("| " + fmt("Progress: waiting for first log line...", W - 4) + " |")
    out.append("+" + bar + "+")

    # Loss
    if data and data["losses"]:
        first = data["losses"][0]["loss"]
        last = data["losses"][-1]
        drop = (first - last["loss"]) / first * 100 if first else 0
        loss_line = "Current loss: %.4f  (started %.3f, down %.1f%%)" % (
            last["loss"], first, drop)
        out.append("| " + fmt(loss_line, W - 4) + " |")
        meta = "grad_norm: %.3f   learning_rate: %.4e" % (last["grad"], last["lr"])
        out.append("| " + fmt(meta, W - 4) + " |")
        out.append("| " + fmt("Last 5 loss values:", W - 4) + " |")
        for r in data["losses"][-5:]:
            ln = "   loss=%.4f  grad=%.3f  lr=%.4e" % (r["loss"], r["grad"], r["lr"])
            out.append("| " + fmt(ln, W - 4) + " |")
    else:
        out.append("| " + fmt("No loss values yet", W - 4) + " |")
    out.append("+" + bar + "+")

    # Eval
    if data and data["evals"]:
        last_eval = data["evals"][-1]
        pph = math.exp(last_eval["loss"])
        latest = "Latest eval_loss: %.4f  (~ perplexity %.1f)" % (last_eval["loss"], pph)
        out.append("| " + fmt(latest, W - 4) + " |")
        hist = "Eval history (%d eval(s)):" % len(data["evals"])
        out.append("| " + fmt(hist, W - 4) + " |")
        for ev in data["evals"][-3:]:
            pph = math.exp(ev["loss"])
            row = "   eval_loss=%.4f  pph~%.1f  (%.1f samples/s)" % (
                ev["loss"], pph, ev["sps"])
            out.append("| " + fmt(row, W - 4) + " |")
    else:
        out.append("| " + fmt("No eval yet (first eval at step 50)", W - 4) + " |")
    out.append("+" + bar + "+")

    # Per-language grad norms
    if per_lang:
        out.append("| " + fmt("Per-language grad norms (last %d):" % len(per_lang),
                              W - 4) + " |")
        for rec in per_lang:
            step = rec.get("step", "?")
            norms = rec.get("norms", {})
            parts = []
            for lang in sorted(norms.keys()):
                n = norms[lang]
                if isinstance(n, (int, float)):
                    parts.append("%s=%.2f" % (lang, n))
                else:
                    parts.append("%s=null" % lang)
            row = "   step %s: %s" % (step, "  ".join(parts))
            out.append("| " + fmt(row, W - 4) + " |")
    else:
        out.append("| " + fmt("Per-language grad norms: not yet recorded", W - 4) + " |")
    out.append("+" + bar + "+")

    # GPUs
    if gpus:
        out.append("| " + fmt("GPUs (utilization / VRAM):", W - 4) + " |")
        for i in range(0, len(gpus), 4):
            row_parts = []
            for j in range(i, min(i + 4, len(gpus))):
                g = gpus[j]
                row_parts.append("GPU%d: %3d%% / %4.1fGB" % (
                    g["idx"], g["util"], g["mem"] / 1024))
            out.append("| " + fmt("  ".join(row_parts), W - 4) + " |")
    out.append("+" + bar + "+")

    # Disks
    if disks:
        for mnt in ("/", "/data"):
            if mnt in disks:
                d = disks[mnt]
                warn = " WARN" if d["pct"] > 85 else ""
                ln = "Disk %-6s: %4sGB used, %4sGB free (%3d%%)%s" % (
                    mnt, d["used"], d["avail"], d["pct"], warn)
                out.append("| " + fmt(ln, W - 4) + " |")
    out.append("+" + bar + "+")
    foot = "Refresh every %ds.  Ctrl-C to exit." % refresh
    out.append("| " + fmt(foot, W - 4) + " |")
    out.append("+" + bar + "+")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v2A", help="v2A or v2D (default: v2A)")
    ap.add_argument("--refresh", type=int, default=5)
    args = ap.parse_args()

    tag = args.run[-1].upper()
    pattern_chk = "RUN-V2" + tag + "-*"
    pattern_log = "RUN-V2" + tag + "-*.log"
    tmux_sess = "lughagen_v2" + tag

    log_path = find_latest(pattern_log, "/workspace/logs")
    run_dir = find_latest(pattern_chk, "/workspace/checkpoints")

    if not log_path:
        print("ERROR: No log matching " + pattern_log + " in /workspace/logs/")
        sys.exit(1)

    print("Log:     " + str(log_path))
    print("Run dir: " + str(run_dir))
    print("Refreshing every " + str(args.refresh) + "s. Ctrl-C to exit.")
    time.sleep(1)

    try:
        while True:
            data = parse_log(log_path)
            per_lang = []
            if run_dir:
                per_lang = parse_per_lang(run_dir / "per_language_grad_norms.jsonl")
            gpus = gpu_status()
            disks = disk_status()
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(render(data, per_lang, gpus, disks, run_dir, log_path,
                                     tmux_sess, args.refresh))
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
