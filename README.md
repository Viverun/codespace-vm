# codespace-vm

Codespace for testing OSWorld (Docker + KVM check).

## Agent (Jev + GPT-6-luna) on OSWorld

Requires OSWorld at `/workspaces/OSWorld` with its venv and VM image, plus Codespaces secrets
`TYPESAFE_API_KEY` and `AZURE_OPENAI_API_KEY`.

```sh
cd /workspaces/OSWorld
.venv/bin/pip install typesafe-sdk   # or: uv pip install typesafe-sdk
.venv/bin/python /workspaces/codespace-vm/run_osworld.py --mode jev --limit 1        # one task
.venv/bin/python /workspaces/codespace-vm/run_osworld.py --mode jev                  # 14 multi_apps tasks (test_small, no Google Drive)
.venv/bin/python /workspaces/codespace-vm/run_osworld.py --mode luna_only            # baseline
```

- `jev`: Luna plans up to 4 subgoals; Jev grounds click targets (Box) and checks each subgoal's expected outcome.
- `luna_only`: Luna plans one subgoal per call with pixel coordinates; no Jev.
Both bypass `env.step` (no 2 s sleep, no a11y tree) and use a 960x540 in-VM JPEG screenshot. Scores come from OSWorld's own evaluators.
Results: `results/<mode>/<task_id>.json` with score, time, Luna calls and per-phase timing.
