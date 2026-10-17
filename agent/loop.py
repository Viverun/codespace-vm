"""The agent loop.

mode "jev": Luna plans several subgoals; actions run directly; Jev grounds click targets and
checks each subgoal's expected outcome. Luna is called again only when a plan finishes, a
check fails, or a target can't be found.
mode "luna_only": baseline. Luna plans one subgoal per call with pixel coordinates; no Jev.
"""

import base64
import time
from collections import defaultdict
from pathlib import Path

SCREEN_W, SCREEN_H = 1920, 1080
SHOT_W, SHOT_H = 960, 540
SETTLE_S = 0.4  # let the UI react before a checkpoint screenshot

FAST_SHOT = (
    "import pyautogui, io, base64; im = pyautogui.screenshot().convert('RGB'); "
    f"im = im.resize(({SHOT_W}, {SHOT_H})); b = io.BytesIO(); im.save(b, 'JPEG', quality=75); "
    "print(base64.b64encode(b.getvalue()).decode())"
)


class Timer:
    def __init__(self):
        self.ms = defaultdict(float)
        self.calls = defaultdict(int)

    def __call__(self, name, fn, *args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.ms[name] += (time.perf_counter() - t) * 1000
            self.calls[name] += 1

    def summary(self) -> dict:
        return {k: {"calls": self.calls[k], "ms": round(v)} for k, v in self.ms.items()}


class VM:
    """Direct access to the OSWorld VM, bypassing env.step's fixed sleep and full observation."""

    def __init__(self, controller):
        self.c = controller

    def act(self, code: str) -> None:
        self.c.execute_python_command("import pyautogui, time; pyautogui.FAILSAFE = False; " + code)

    def shot(self) -> bytes:
        return base64.b64decode(self.c.execute_python_command(FAST_SHOT)["output"].strip())


def _center(box: dict) -> tuple[int, int]:
    return (round((box["left"] + box["right"]) / 2 * SCREEN_W), round((box["top"] + box["bottom"]) / 2 * SCREEN_H))


def _code(action: dict, xy: tuple[int, int] | None) -> str | None:
    kind = action["type"]
    if kind in ("click", "double_click", "right_click"):
        fn = {"click": "click", "double_click": "doubleClick", "right_click": "rightClick"}[kind]
        return f"pyautogui.{fn}({xy[0]}, {xy[1]})"
    if kind == "type":
        return f"pyautogui.write({action['text']!r}, interval=0.005)"
    if kind == "hotkey":
        return f"pyautogui.hotkey(*{list(action['keys'])!r})"
    if kind == "press":
        return f"pyautogui.press({action['key']!r})"
    if kind == "scroll":
        return f"pyautogui.scroll({int(action.get('amount', -5))}, {xy[0]}, {xy[1]})"
    if kind == "wait":
        return f"time.sleep({min(float(action.get('seconds', 1)), 5)})"
    return None


class _LiveLog(list):
    """History list that also prints each entry as it is added."""

    def append(self, item: str) -> None:
        print("   " + item, flush=True)
        super().append(item)


class Agent:
    def __init__(self, vm: VM, luna, jev=None, max_luna_calls: int = 20, box_min_conf: float = 0.5, debug_dir: Path | None = None):
        self.vm, self.luna, self.jev = vm, luna, jev
        self.max_luna_calls, self.box_min_conf = max_luna_calls, box_min_conf
        self.timer = Timer()
        self.debug_dir, self._shot_no = debug_dir, 0
        if debug_dir:
            debug_dir.mkdir(parents=True, exist_ok=True)

    def _shot(self, label: str) -> bytes:
        """Screenshot, also saved to debug_dir as NNN_label.jpg for inspecting runs."""
        data = self.timer("screenshot", self.vm.shot)
        if self.debug_dir:
            self._shot_no += 1
            (self.debug_dir / f"{self._shot_no:03d}_{label}.jpg").write_bytes(data)
        return data

    def _target_xy(self, action: dict) -> tuple[int, int] | str:
        """Resolve a click/scroll target to screen pixels, or return a failure reason."""
        if self.jev is None:  # baseline: coordinates are in screenshot space
            return round(action["x"] * SCREEN_W / SHOT_W), round(action["y"] * SCREEN_H / SHOT_H)
        shot = self._shot("locate")
        found = self.timer("jev_locate", self.jev.locate, shot, action["target"])
        if found["visible"] < 0.5 or found["confidence"] < self.box_min_conf:
            return f"could not find '{action['target']}' (visible={found['visible']:.2f}, box conf={found['confidence']:.2f})"
        self._last_click = f"{_center(found['box'])} for '{action['target']}'"
        return _center(found["box"])

    def run(self, instruction: str) -> dict:
        history = _LiveLog()
        status = "continue"
        luna_calls = 0
        failures_in_a_row = 0
        while luna_calls < self.max_luna_calls:
            shot = self._shot("plan")
            luna_calls += 1
            try:
                reply = self.timer("luna", self.luna.next, shot, instruction, history)
            except Exception as e:  # malformed JSON or API hiccup: retry with a note
                history.append(f"[planner error: {e!s:.200}]")
                continue
            status = reply.get("status", "continue")
            print(f"  luna #{luna_calls}: {status}, {len(reply.get('plan', []))} subgoal(s)", flush=True)
            if status in ("done", "infeasible"):
                break

            for sub in reply.get("plan", [])[: 1 if self.jev is None else 4]:
                failure = None
                for action in sub.get("actions", []):
                    xy = None
                    if action.get("type") in ("click", "double_click", "right_click", "scroll"):
                        xy = self._target_xy(action)
                        if isinstance(xy, str):
                            failure = xy
                            break
                    code = _code(action, xy)
                    if code:
                        self.timer("act", self.vm.act, code)
                if failure:
                    history.append(f"- {sub.get('subgoal')}: STOPPED, {failure}")
                    failures_in_a_row += 1
                    self._loop_guard(history, failures_in_a_row)
                    break
                if self.jev is None:
                    history.append(f"- {sub.get('subgoal')}: executed {sub.get('actions')}")
                    continue
                time.sleep(SETTLE_S)
                check = self.timer("jev_check", self.jev.check, self._shot("check"), sub.get("expect", ""))
                clicked = getattr(self, "_last_click", None)
                if check["met"] < 0.5 or check["blocked"] > 0.5:
                    history.append(
                        f"- {sub.get('subgoal')}: automatic check says NOT met (expected '{sub.get('expect')}', "
                        f"P(met)={check['met']:.2f}, P(blocking dialog)={check['blocked']:.2f}; last click {clicked}). "
                        "The check can be wrong: verify on the current screenshot."
                    )
                    failures_in_a_row += 1
                    self._loop_guard(history, failures_in_a_row)
                    break
                failures_in_a_row = 0
                history.append(f"- {sub.get('subgoal')}: ok")

        return {"status": status, "luna_calls": luna_calls, "history": list(history), "timing": self.timer.summary()}

    @staticmethod
    def _loop_guard(history: list, failures_in_a_row: int) -> None:
        if failures_in_a_row >= 2:
            history.append(
                f"[{failures_in_a_row} attempts in a row failed. First look at the screenshot: if the step actually "
                "succeeded, move on. Otherwise do NOT repeat the same approach: use a different method "
                "(keyboard shortcut, terminal command, menu, or another UI path). Never click a toggle again "
                "without first checking its current state on the screenshot.]"
            )
