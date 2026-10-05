#!/usr/bin/env python3
"""C1-PROFILE-PLAN step A, end to end: llama-server on the MTP fixture with a word list, writing with the apprentice.

Three serves of the same paper (temperature 0, EOS ignored), all with the spec-rate trace on:
  off       GGML_METAL_CBLOG unset: no timeline, no sidecars, and no "round" in the trace
  on        the timeline plus both sidecars (<cblog>.phases, <cblog>.stops)
  cblog     the timeline only (GGML_METAL_CBLOG_SIDECARS=0): no sidecars
The words must be the same in all three. On the "on" serve:
  - every P line ends after it starts; P lines nest or are apart, never half-overlap
  - round ids start at 0 and rise one at a time; round -1 holds only the openings
  - exactly one check per round (a round with no draft, the answer's last word, included)
  - per round, draft_step lines = the trace cycle's steps, and accept's arg = the cycle's kept (no accept when the
    cycle checked nothing: a draft dropped whole, by p-min or a cap of 0, is checked as a plain word)
  - the words add up: 1 (from the reading in) + sum over rounds of (kept + 1) = the words written
  - every Metal ctx gets a role, one target and one apprentice
  - per check, one F line per target floor, all on the target's ctx, and each F names the graph before it
Stdlib only. Exit 77 (ctest skip) without the server binary.
"""

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

PROMPT = [5, 9, 17, 33, 41, 7, 19, 23, 11, 60, 14, 8, 27, 31, 44, 52]
N_PREDICT = 24
SERVE_ARGS = ["-ngl", "99", "--moe-stream", "--moe-stream-cache", "40s", "-c", "1024", "-np", "1",
              "--spec-type", "draft-mtp", "--spec-draft-n-max", "3", "--spec-draft-p-min", "0",
              "--spec-draft-ngl", "99", "--no-webui"]
TGT_PHASES = {"open_tgt", "save_tgt", "check", "restore_tgt", "accept"}
DFT_PHASES = {"open_dft", "save_dft", "draft_step", "load_dft", "process"}

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("FAIL: " + msg, file=sys.stderr)
    return cond


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(server, model, work, tag, env_extra):
    """One server: read the paper in, write N_PREDICT words, stop (SIGTERM, so the exit writes run)."""
    d = os.path.join(work, tag)
    os.makedirs(d)
    env = {k: v for k, v in os.environ.items()
           if k not in ("GGML_METAL_CBLOG", "GGML_METAL_CBLOG_SIDECARS", "LLAMA_SPEC_RATE_TRACE")}
    env["LLAMA_SPEC_RATE_TRACE"] = os.path.join(d, "trace.srtr")
    env.update({k: v.replace("{dir}", d) for k, v in env_extra.items()})
    port = free_port()
    log_path = os.path.join(d, "server.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen([server, "-m", model, "--port", str(port)] + SERVE_ARGS, env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"{tag}: the server died at start (rc {proc.returncode}), see {log_path}")
            if "listening on http" in open(log_path, errors="replace").read():
                break
            time.sleep(0.2)
        else:
            raise RuntimeError(f"{tag}: the server never came up, see {log_path}")
        body = json.dumps({"prompt": PROMPT, "n_predict": N_PREDICT, "temperature": 0, "ignore_eos": True,
                           "cache_prompt": False}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    check(proc.returncode == 0, f"{tag}: the server's exit code is {proc.returncode}")
    return d, out


def read_lines(path):
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


def parse_phases(path):
    rows = read_lines(path)
    check(rows and rows[0][0] == "A" and len(rows[0]) == 3, f"{path}: the first line is the anchor")
    out = []
    for r in rows[1:]:
        if check(r[0] == "P" and len(r) == 6, f"{path}: a P line has 6 fields: {r}"):
            out.append({"round": int(r[1]), "phase": r[2], "t0": float(r[3]), "t1": float(r[4]), "arg": int(r[5])})
    return out


def parse_stops(path):
    rows = read_lines(path)
    check(rows and rows[0][0] == "A" and len(rows[0]) == 3, f"{path}: the first line is the anchor")
    out = []
    for r in rows[1:]:
        if check(r[0] == "F" and len(r) == 10, f"{path}: an F line has 10 fields: {r}"):
            out.append({"ctx": r[1], "seq": int(r[2]), "il": int(r[3]), "t0": float(r[4]), "n_asked": int(r[5]),
                        "n_trips": int(r[6]), "lookup": float(r[7]), "send": float(r[8]), "wait": float(r[9])})
    return out


def parse_graphs(path):
    return {(r[1], int(r[2])): float(r[3]) for r in read_lines(path) if r[0] == "G"}


def parse_cycles(path):
    cycles = []
    for line in open(path):
        rec = json.loads(line)
        if "cycle" in rec:
            cycles.append(rec)
    return cycles


def innermost(phases, t):
    best = None
    for p in phases:
        if p["t0"] <= t <= p["t1"] and (best is None or p["t1"] - p["t0"] < best["t1"] - best["t0"]):
            best = p
    return best


def roles(phases, graphs):
    """A ctx's role: the opening its first graph falls in; failing that (the target under book streaming has no
    warm-up, and its RS memory skips the test decode) the first writing phase with a role that one of its graphs
    falls in."""
    first = {}
    for (ctx, seq), t in graphs.items():
        if ctx not in first or seq < first[ctx][0]:
            first[ctx] = (seq, t)
    out = {}
    for ctx, (_, t) in first.items():
        p = innermost([p for p in phases if p["phase"] in ("open_tgt", "open_dft")], t)
        if p is not None:
            out[ctx] = "tgt" if p["phase"] == "open_tgt" else "dft"
            continue
        for (c, seq) in sorted(k for k in graphs if k[0] == ctx):
            p = innermost([p for p in phases if p["round"] >= 0], graphs[(c, seq)])
            if p is not None and p["phase"] in TGT_PHASES | DFT_PHASES:
                out[ctx] = "tgt" if p["phase"] in TGT_PHASES else "dft"
                break
    return out


def check_on(d, out):
    cblog = os.path.join(d, "cblog.txt")
    phases = parse_phases(cblog + ".phases")
    stops = parse_stops(cblog + ".stops")
    graphs = parse_graphs(cblog)
    cycles = parse_cycles(os.path.join(d, "trace.srtr"))
    check(len(graphs) > 0 and len(phases) > 0 and len(stops) > 0, "the timeline, the phases and the stops have lines")

    for p in phases:
        check(p["t1"] >= p["t0"], f"a phase ends before it starts: {p}")
    srt = sorted(phases, key=lambda p: (p["t0"], -p["t1"]))
    for i, a in enumerate(srt):
        for b in srt[i + 1:]:
            if b["t0"] >= a["t1"]:
                break
            check(b["t1"] <= a["t1"], f"half-overlapping phases: {a} and {b}")

    rounds = [p["round"] for p in phases]
    check(all(p["phase"] in ("open_tgt", "open_dft") for p in phases if p["round"] < 0), "round -1 is openings only")
    seen = sorted(set(r for r in rounds if r >= 0))
    check(seen == list(range(len(seen))) and len(seen) > 2, f"round ids 0..n one at a time: {seen}")
    check(all(b >= a for a, b in zip(rounds, rounds[1:])), "round ids never go back")

    by_round = {r: [p for p in phases if p["round"] == r] for r in seen}
    for r, ps in by_round.items():
        n_check = sum(p["phase"] == "check" for p in ps)
        check(n_check == 1, f"round {r}: {n_check} checks, want exactly 1")
    plain = [r for r, ps in by_round.items() if not any(p["phase"] == "draft_step" for p in ps)]
    check(len(plain) >= 1, "a round with no draft (the last word) is a round of its own")

    cyc = {c["round"]: c for c in cycles if "round" in c}
    check(len(cyc) == len(cycles) and len(cycles) > 0, "every trace cycle carries its round")
    for r, ps in by_round.items():
        steps = sum(p["phase"] == "draft_step" for p in ps)
        acc = [p["arg"] for p in ps if p["phase"] == "accept"]
        if r in cyc:
            check(steps == cyc[r]["steps"], f"round {r}: {steps} draft_step lines, the trace says {cyc[r]['steps']}")
            want = [cyc[r]["kept"]] if cyc[r]["checked"] > 0 else []  # a draft dropped whole is checked as a plain word
            check(acc == want, f"round {r}: accept {acc}, the trace checked {cyc[r]['checked']} kept {cyc[r]['kept']}")
        else:
            check(steps == 0 and acc == [], f"round {r}: drafting with no trace cycle")
    words = 1 + sum((next((p["arg"] for p in ps if p["phase"] == "accept"), 0)) + 1 for ps in by_round.values())
    check(words == out["tokens_predicted"], f"rounds add up to {words} words, the server wrote {out['tokens_predicted']}")

    role = roles(phases, graphs)
    ctxs = {c for c, _ in graphs}
    check(set(role) == ctxs, f"every ctx gets a role: {role} of {ctxs}")
    check(sorted(role.values()) == ["dft", "tgt"], f"one target, one apprentice: {role}")
    tgt = next((c for c, r in role.items() if r == "tgt"), None)

    floors = None
    for p in (p for p in phases if p["phase"] == "check"):
        fs = [f for f in stops if p["t0"] <= f["t0"] <= p["t1"]]
        ils = sorted(f["il"] for f in fs)
        check(all(f["ctx"] == tgt for f in fs), f"round {p['round']}: a check's stop on another ctx")
        check(len(ils) == len(set(ils)) and ils, f"round {p['round']}: floors {ils}, want each target floor once")
        floors = ils if floors is None else floors
        check(ils == floors, f"round {p['round']}: floors {ils}, the first check's were {floors}")
    for f in stops:
        key = (f["ctx"], f["seq"])
        if check(key in graphs, f"a stop names no graph: {f}"):
            check(graphs[key] < f["t0"], f"a stop's graph comes after it: {f}")
            nxt = graphs.get((f["ctx"], f["seq"] + 1))
            check(nxt is None or nxt > f["t0"], f"a stop is not in the gap after its graph: {f}")
        check(min(f["lookup"], f["send"], f["wait"]) >= 0, f"a stop's split is negative: {f}")
    print(f"on: {len(seen)} rounds, {len(plain)} with no draft, {len(stops)} stops, floors {floors}, roles {role}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--keep", action="store_true", help="keep the work folder")
    a = ap.parse_args()
    if not os.path.exists(a.server):
        print(f"skip: no server at {a.server}")
        return 77
    work = tempfile.mkdtemp(prefix="test-c1-phase-server-")
    try:
        d_off, out_off = serve(a.server, a.model, work, "off", {})
        d_on, out_on = serve(a.server, a.model, work, "on", {"GGML_METAL_CBLOG": "{dir}/cblog.txt"})
        d_cb, out_cb = serve(a.server, a.model, work, "cblog", {"GGML_METAL_CBLOG": "{dir}/cblog.txt",
                                                                "GGML_METAL_CBLOG_SIDECARS": "0"})
        check(out_on["content"] == out_off["content"] == out_cb["content"], "the words moved with the logs on")
        check(out_off["tokens_predicted"] == N_PREDICT, f"wrote {out_off['tokens_predicted']} words, want {N_PREDICT}")

        check(os.listdir(d_off) and not any(n.startswith("cblog") for n in os.listdir(d_off)), "off: no log files")
        check(all('"round"' not in line for line in open(os.path.join(d_off, "trace.srtr"))),
              "off: the trace carries no round")
        check(os.path.exists(os.path.join(d_cb, "cblog.txt")), "cblog only: the timeline is written")
        check(not any(n.endswith((".phases", ".stops")) for n in os.listdir(d_cb)), "cblog only: no sidecars")
        check_on(d_on, out_on)
    finally:
        if a.keep or failures:
            print(f"work folder kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    if failures:
        print(f"{len(failures)} failure(s)", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
