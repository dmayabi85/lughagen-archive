#!/usr/bin/env python3
"""Sweep-aware monitor for LughaGen Phase 2 factorial sweep.

Auto-detects the currently active cell. Shows sweep-level progress on top,
current-cell metrics below. Refreshes every 5 seconds. Ctrl-C to exit.
The sweep keeps running regardless of whether this monitor is on.
"""
import csv, math, re, subprocess, sys, time
from pathlib import Path

SWEEP_SESSION = "lughagen-sweep"
LOGS_DIR = Path("/workspace/logs")
REGISTRY = Path("/workspace/registry/runs.csv")

LOSS_RE = re.compile(r"\{'loss': '?([\d.eE+-]+)'?, 'grad_norm': '?([\d.eE+-]+)'?, 'learning_rate': '?([\d.eE+-]+)'?, 'epoch': '?([\d.eE+-]+)'?\}")
EVAL_RE = re.compile(r"\{'eval_loss': '?([\d.eE+-]+)'?")
PROG_RE = re.compile(r"(\d+)/(\d+)\s*\[(\d+:\d+:\d+|\d+:\d+)<(\d+:\d+:\d+|\d+:\d+),\s+([\d.]+)s/it\]")
MFU_INIT_RE = re.compile(r"\[mfu\] init: n_params=([\d,]+) flops/step=([\d.eE+-]+) peak_aggregate=([\d.]+) TFLOPS/s world_size=(\d+)")
PLANG_RE = re.compile(r"\[grad-norm\] step (\d+): norms=(\{[^}]*\})")
CELL_RE = re.compile(r"RUN-P2-C\d+-S\d+")


def sweep_alive():
    try:
        return subprocess.run(["tmux", "has-session", "-t", SWEEP_SESSION],
                              capture_output=True, timeout=3).returncode == 0
    except Exception:
        return False


def find_current_cell():
    logs = sorted(
        LOGS_DIR.glob("phase2_RUN-P2-C*.log"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    if not logs:
        return None, None
    latest = logs[0]
    m = CELL_RE.search(latest.name)
    return (m.group(0) if m else None), latest


def get_completed_cells():
    cells = []
    if not REGISTRY.exists():
        return cells
    try:
        with open(REGISTRY) as f:
            for r in csv.DictReader(f):
                rid = r.get("factorial_run_id", "")
                phase = r.get("phase", "")
                pph = r.get("metrics.validation_perplexity", "")
                if rid.startswith("RUN-P2-C") and phase == "Phase2_Factorial" and pph:
                    try:
                        cells.append({
                            "rid": rid,
                            "pph": float(pph),
                            "opt": r.get("factors.optimiser", "?"),
                            "seq": r.get("factors.sequence_length", "?"),
                            "seed": r.get("seed", "?"),
                            "wall_h": float(r.get("wall_clock_seconds", 0)) / 3600,
                        })
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return cells


def parse_log(log_path):
    if not log_path or not log_path.exists():
        return None
    losses, evals, progress, mfu_init, plang = [], [], None, None, []
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 3 * 1024 * 1024))
            if size > 3 * 1024 * 1024:
                f.readline()
            text = f.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    for line in text.split("\n"):
        m = LOSS_RE.search(line)
        if m:
            try:
                losses.append({
                    "loss": float(m.group(1)),
                    "grad": float(m.group(2)),
                    "lr": float(m.group(3)),
                })
            except ValueError:
                pass
        m = EVAL_RE.search(line)
        if m:
            try:
                evals.append(float(m.group(1)))
            except ValueError:
                pass
        for m in PROG_RE.finditer(line):
            try:
                progress = {
                    "step": int(m.group(1)),
                    "max": int(m.group(2)),
                    "elapsed": m.group(3),
                    "eta": m.group(4),
                    "sps": float(m.group(5)),
                }
            except ValueError:
                pass
        m = MFU_INIT_RE.search(line)
        if m:
            try:
                mfu_init = {
                    "flops_per_step": float(m.group(2)),
                    "peak_tflops": float(m.group(3)),
                    "world_size": int(m.group(4)),
                }
            except ValueError:
                pass
        m = PLANG_RE.search(line)
        if m:
            try:
                pairs = {}
                for nm in re.finditer(r"'?(\w+)'?:\s*([\d.eE+-]+)", m.group(2)):
                    try:
                        pairs[nm.group(1)] = float(nm.group(2))
                    except ValueError:
                        pass
                plang.append({"step": int(m.group(1)), "norms": pairs})
            except Exception:
                pass
    return {
        "losses": losses,
        "evals": evals,
        "progress": progress,
        "mfu_init": mfu_init,
        "plang": plang,
    }


def gpu_state():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return []
        out = []
        for line in r.stdout.strip().split("\n"):
            p = [x.strip() for x in line.split(",")]
            if len(p) == 3:
                out.append({"i": int(p[0]), "u": int(p[1]), "m": int(p[2])})
        return out
    except Exception:
        return []


def disk_root():
    try:
        r = subprocess.run(["df", "-h", "/"], capture_output=True, text=True, timeout=3)
        for line in r.stdout.strip().split("\n")[1:]:
            p = line.split()
            if len(p) >= 5:
                return {"used": p[2], "avail": p[3], "pct": int(p[4].rstrip("%"))}
    except Exception:
        pass
    return None


def fmt(s, w):
    if len(s) > w:
        return s[:w-1] + "..."
    return s.ljust(w)


def render():
    W = 80
    bar = "-" * (W - 2)
    out = []
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    out.append("+" + bar + "+")
    out.append("| " + fmt("LughaGen Phase 2 Sweep Monitor  ·  " + now, W-4) + " |")
    out.append("+" + bar + "+")

    sweep = "ALIVE" if sweep_alive() else "DEAD"
    completed = get_completed_cells()
    out.append("| " + fmt("Sweep session (" + SWEEP_SESSION + "): " + sweep, W-4) + " |")
    out.append("| " + fmt("Cells completed: {}/32".format(len(completed)), W-4) + " |")

    if completed:
        out.append("| " + fmt("Recent results:", W-4) + " |")
        for c in sorted(completed, key=lambda x: x["rid"])[-3:]:
            line = "  {:15s} {:6s} seq={:>4} seed={:>3}  pph={:>7.3f}  wall={:>5.1f}h".format(
                c["rid"], c["opt"], str(c["seq"]), str(c["seed"]), c["pph"], c["wall_h"])
            out.append("| " + fmt(line, W-4) + " |")
    out.append("+" + bar + "+")

    cell_id, log_path = find_current_cell()
    if not cell_id:
        out.append("| " + fmt("No active cell log found", W-4) + " |")
    else:
        out.append("| " + fmt("Current cell: " + cell_id, W-4) + " |")
        log_age = (time.time() - log_path.stat().st_mtime) if log_path else 0
        out.append("| " + fmt("log: ..." + str(log_path)[-50:] + "  ({:.0f}s old)".format(log_age), W-4) + " |")
        out.append("+" + bar + "+")

        data = parse_log(log_path)
        if data is None:
            out.append("| " + fmt("(log not yet readable)", W-4) + " |")
        else:
            if data["mfu_init"]:
                m = data["mfu_init"]
                line = "MFU init: flops/step={:.3e}  world_size={}".format(
                    m["flops_per_step"], m["world_size"])
                out.append("| " + fmt(line, W-4) + " |")

            if data["progress"]:
                p = data["progress"]
                pct = p["step"] / p["max"] * 100
                bw = 50
                filled = int(bw * p["step"] / p["max"])
                pbar = "#" * filled + "." * (bw - filled)
                line = "Step {:>5}/{} ({:5.2f}%)  ETA {}  ({:.2f}s/step)".format(
                    p["step"], p["max"], pct, p["eta"], p["sps"])
                out.append("| " + fmt(line, W-4) + " |")
                out.append("| " + fmt("[" + pbar + "]", W-4) + " |")
            else:
                out.append("| " + fmt("Progress: (starting up)", W-4) + " |")

            if data["losses"]:
                first = data["losses"][0]["loss"]
                last = data["losses"][-1]
                drop = (first - last["loss"]) / first * 100 if first > 0 else 0
                line = "Loss: {:.4f} (start {:.3f}, drop {:.1f}%)  grad: {:.3f}  lr: {:.3e}".format(
                    last["loss"], first, drop, last["grad"], last["lr"])
                out.append("| " + fmt(line, W-4) + " |")

            if data["evals"]:
                last_eval = data["evals"][-1]
                pph = math.exp(last_eval) if last_eval < 100 else float("inf")
                line = "Eval ({}x): loss={:.4f}  pph~{:.2f}".format(
                    len(data["evals"]), last_eval, pph)
                out.append("| " + fmt(line, W-4) + " |")

            if data["plang"]:
                last_plang = data["plang"][-1]
                pieces = []
                for k, v in last_plang["norms"].items():
                    pieces.append("{}={:.2f}".format(k, v))
                line = "Plang step {}: {}".format(last_plang["step"], "  ".join(pieces))
                out.append("| " + fmt(line, W-4) + " |")

    out.append("+" + bar + "+")

    gpus = gpu_state()
    if gpus:
        active = sum(1 for g in gpus if g["u"] > 50)
        avg_util = sum(g["u"] for g in gpus) / len(gpus)
        max_mem = max(g["m"] for g in gpus) / 1024
        line = "GPUs: {}/{} active  avg_util={:.0f}%  peak_VRAM={:.1f}GB".format(
            active, len(gpus), avg_util, max_mem)
        out.append("| " + fmt(line, W-4) + " |")

    d = disk_root()
    if d:
        warn = " WARN" if d["pct"] > 90 else ""
        line = "Disk /: {} used, {} free ({}%){}".format(d["used"], d["avail"], d["pct"], warn)
        out.append("| " + fmt(line, W-4) + " |")

    out.append("+" + bar + "+")
    out.append("| " + fmt("Refresh every 5s. Ctrl-C to exit. Sweep continues.", W-4) + " |")
    out.append("+" + bar + "+")
    return "\n".join(out)


def main():
    print("Starting sweep monitor...")
    time.sleep(0.5)
    try:
        while True:
            sys.stdout.write("\033[H\033[J")
            sys.stdout.write(render())
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor stopped. Sweep continues.")


if __name__ == "__main__":
    main()
