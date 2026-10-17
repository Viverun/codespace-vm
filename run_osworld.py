"""Run the agent on OSWorld tasks and score them with OSWorld's own evaluators.

  cd /workspaces/OSWorld
  .venv/bin/python /workspaces/codespace-vm/run_osworld.py --mode jev --limit 1
  .venv/bin/python /workspaces/codespace-vm/run_osworld.py --mode luna_only

Must run from the OSWorld checkout (the Docker provider looks for ./docker_vm_data).
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent.loop import VM, Agent  # noqa: E402
from agent.models import Jev, Luna  # noqa: E402
from desktop_env.desktop_env import DesktopEnv  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def tasks(test_file: str, domain: str, ids: list[str] | None):
    listing = json.loads(Path("evaluation_examples", test_file).read_text())
    for task_id in ids or listing[domain]:
        config = json.loads(Path("evaluation_examples/examples", domain, f"{task_id}.json").read_text())
        if any(k in json.dumps(config).lower() for k in ("googledrive", "google_drive")):
            print(f"skip {task_id[:8]} (needs Google Drive)")
            continue
        yield config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["jev", "luna_only"], default="jev")
    p.add_argument("--domain", default="multi_apps")
    p.add_argument("--test-file", default="test_small.json")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-luna-calls", type=int, default=20)
    args = p.parse_args()

    out = RESULTS / args.mode
    out.mkdir(parents=True, exist_ok=True)
    env = DesktopEnv(provider_name="docker", os_type="Ubuntu", action_space="pyautogui", headless=True, require_a11y_tree=False)
    luna, jev = Luna(grounded=args.mode == "jev"), (Jev() if args.mode == "jev" else None)

    rows = []
    try:
        for n, config in enumerate(tasks(args.test_file, args.domain, args.ids)):
            if args.limit and n >= args.limit:
                break
            task_id = config["id"]
            print(f"\n=== {task_id[:8]} {config['instruction'][:100]}", flush=True)
            env.reset(task_config=config)
            agent = Agent(VM(env.controller), luna, jev, max_luna_calls=args.max_luna_calls)
            t0 = time.perf_counter()
            try:
                result = agent.run(config["instruction"])
            except Exception as e:
                result = {"status": "error", "error": repr(e), "timing": agent.timer.summary(), "history": []}
            agent_s = time.perf_counter() - t0
            if result["status"] == "infeasible":
                env.step("FAIL", pause=0)
            try:
                score = float(env.evaluate())
            except Exception as e:
                score, result["eval_error"] = 0.0, repr(e)
            row = {"id": task_id, "mode": args.mode, "score": score, "agent_s": round(agent_s, 1), **result}
            (out / f"{task_id}.json").write_text(json.dumps(row, indent=2))
            rows.append(row)
            print(f"score={score}  agent_time={agent_s:.1f}s  luna_calls={result.get('luna_calls')}  timing={result['timing']}", flush=True)
    finally:
        env.close()

    if rows:
        solved = sum(r["score"] >= 1.0 for r in rows)
        mean_s = sum(r["agent_s"] for r in rows) / len(rows)
        print(f"\n{args.mode}: solved {solved}/{len(rows)}, mean score {sum(r['score'] for r in rows) / len(rows):.2f}, mean agent time {mean_s:.1f}s")


if __name__ == "__main__":
    main()
