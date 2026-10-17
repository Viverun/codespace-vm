"""Model adapters: Jev for fast checks and grounding, GPT-6-luna for planning."""

import base64
import json
import os

from openai import OpenAI
from typesafe_sdk import Box, Choice, Image, Noul, TypeSafeClient

LUNA_ENDPOINT = os.environ.get(
    "AZURE_OPENAI_ENDPOINT", "https://khanjamilahmed202-3046-resource.openai.azure.com/openai/v1"
)
LUNA_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-6-luna")


class Jev:
    def __init__(self, model: str = "jev-preview"):
        self.client = TypeSafeClient(model=model, headers={"TypeSafe-Danger-Zone-Images": "true"})

    def locate(self, jpeg: bytes, target: str) -> dict:
        """Find `target` on screen. Returns fractional box, Box confidence and P(visible)."""
        r = self.client.system_one(
            state={"screen": Image(jpeg)},
            questions={
                "visible": Noul(instructions=f"Is this visible on `screen`: {target}?"),
                "box": Box(instructions=f"{target} on `screen`"),
            },
        )
        box = r.answers["box"]
        return {"box": box.box.model_dump(), "confidence": box.confidence, "visible": r.nouls["visible"].noul}

    def check(self, jpeg: bytes, expect: str) -> dict:
        """One checkpoint request: did the expected outcome happen, and is a dialog in the way?"""
        r = self.client.system_one(
            state={"screen": Image(jpeg)},
            questions={
                "met": Noul(instructions=f"Does `screen` show this: {expect}?"),
                "blocked": Noul(
                    instructions="Is an unexpected dialog, popup, error or permission prompt on `screen` "
                    "that must be handled before continuing?"
                ),
            },
        )
        return {"met": r.nouls["met"].noul, "blocked": r.nouls["blocked"].noul}

    def pick_mark(self, marked_jpeg: bytes, target: str, n: int, min_conf: float = 0.5) -> int | None:
        """Which numbered red box on the image is `target`? Returns its number, or None."""
        criteria = {str(i): None for i in range(1, n + 1)}
        criteria["none"] = "None of the numbered boxes is the target."
        r = self.client.system_one(
            state={"screen": Image(marked_jpeg)},
            questions={
                "mark": Choice(
                    instructions=f"`screen` shows UI elements outlined in red with a number above each. "
                    f"Which number marks {target}?",
                    criteria=criteria,
                )
            },
        )
        answer = r.choices["mark"]
        if answer.choice == "none" or answer.confidence < min_conf:
            return None
        return int(answer.choice)


GROUNDED_PROMPT = """You operate an Ubuntu 22.04 desktop (GNOME, 1920x1080) to complete the user's task.
You see a downscaled screenshot. Reply with JSON only:
{
  "status": "continue" | "done" | "infeasible",
  "plan": [
    {"subgoal": "...", "expect": "what the screen shows once this subgoal succeeded",
     "actions": [ACTION, ...]}
  ]
}
Plan 1-4 subgoals ahead when you are confident; one when unsure. Each "expect" must be checkable on a screenshot.
ACTION is one of:
  {"type": "click" | "double_click" | "right_click", "target": "precise visual description of the element"}
  {"type": "type", "text": "..."}            (types into the focused element)
  {"type": "hotkey", "keys": ["ctrl", "s"]}
  {"type": "press", "key": "enter"}
  {"type": "scroll", "target": "area to scroll", "amount": -5}
  {"type": "wait", "seconds": 1}
Describe click targets visually and uniquely (label text, position, color); never give coordinates.
Prefer keyboard shortcuts and the terminal (ctrl+alt+t) when they are faster than clicking.
Toggles and checkboxes flip on every click: check their current state on the screenshot before clicking.
History lines come from an automatic checker that can be wrong; trust what you see on the screenshot.
Use "done" only when the task is complete on screen, "infeasible" only when it cannot be done at all."""

BASELINE_PROMPT = GROUNDED_PROMPT.replace(
    '{"type": "click" | "double_click" | "right_click", "target": "precise visual description of the element"}',
    '{"type": "click" | "double_click" | "right_click", "x": int, "y": int}   (pixels in the screenshot you see)',
).replace(
    '{"type": "scroll", "target": "area to scroll", "amount": -5}',
    '{"type": "scroll", "x": int, "y": int, "amount": -5}',
).replace(
    "Describe click targets visually and uniquely (label text, position, color); never give coordinates.",
    "Give click coordinates in the pixel space of the screenshot you see.",
).replace("Plan 1-4 subgoals ahead when you are confident; one when unsure.", "Plan exactly one subgoal.")


class Luna:
    def __init__(self, grounded: bool):
        self.client = OpenAI(base_url=LUNA_ENDPOINT, api_key=os.environ["AZURE_OPENAI_API_KEY"])
        self.system = GROUNDED_PROMPT if grounded else BASELINE_PROMPT

    def next(self, jpeg: bytes, instruction: str, history: list[str]) -> dict:
        image = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        text = f"Task: {instruction}\n\nHistory so far:\n" + ("\n".join(history[-8:]) or "(none)")
        r = self.client.chat.completions.create(
            model=LUNA_DEPLOYMENT,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": self.system},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image}},
                ]},
            ],
        )
        return json.loads(r.choices[0].message.content)
