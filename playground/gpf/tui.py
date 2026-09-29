"""Terminal UI: pick a model, type a prompt, watch it continue the text."""
from __future__ import annotations

import time

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, Select, Static, TextArea

from .models import MODELS, generate

HELP = ("Pick a model, write a prompt, press Generate (ctrl+g). The prompt is fed to the model "
        "one character at a time, then the model writes on. The connectome model simulates "
        "166,700 neurons per character (~20 characters/s on a laptop GPU).")


class GPFApp(App):
    TITLE = "GPF: Generative Pretrained Fly"
    SUB_TITLE = "pretrained by evolution, fine-tuned on 20k characters of Shakespeare"
    CSS = """
    #side { width: 38; padding: 0 1; }
    #main { padding: 0 1; }
    #prompt { height: 7; }
    #out_scroll { height: 1fr; border: round $accent; }
    #status { height: 2; color: $text-muted; }
    Label { margin-top: 1; }
    Button { width: 100%; margin-top: 1; }
    """
    BINDINGS = [("ctrl+g", "generate", "Generate"), ("escape", "stop", "Stop"), ("ctrl+q", "quit", "Quit")]

    def __init__(self):
        super().__init__()
        self.models, self.busy, self._stop = {}, False, False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="side"):
                yield Label("Model")
                yield Select([(v[0], k) for k, v in MODELS.items()], value="kn7", allow_blank=False, id="model")
                yield Label("Characters to write")
                yield Input("300", id="n", type="integer")
                yield Label("Temperature")
                yield Input("0.7", id="temp", type="number")
                yield Label("Top-k (0 = off)")
                yield Input("0", id="topk", type="integer")
                yield Label("Seed")
                yield Input("0", id="seed", type="integer")
                yield Button("Generate", id="go", variant="primary")
                yield Button("Stop", id="stop")
                yield Static(HELP, id="help")
            with Vertical(id="main"):
                yield Label("Prompt")
                yield TextArea("ROMEO:\n", id="prompt")
                with VerticalScroll(id="out_scroll"):
                    yield Static("", id="out")
                yield Static("", id="status")
        yield Footer()

    def status(self, msg):
        self.query_one("#status", Static).update(msg)

    def on_button_pressed(self, ev: Button.Pressed):
        if ev.button.id == "go":
            self.action_generate()
        elif ev.button.id == "stop":
            self.action_stop()

    def action_stop(self):
        self._stop = True

    def action_generate(self):
        if self.busy:
            return
        try:
            n = int(self.query_one("#n", Input).value); temp = float(self.query_one("#temp", Input).value)
            topk = int(self.query_one("#topk", Input).value); seed = int(self.query_one("#seed", Input).value)
        except ValueError:
            self.status("check the numbers on the left"); return
        key = self.query_one("#model", Select).value
        prompt = self.query_one("#prompt", TextArea).text
        self.busy, self._stop = True, False
        self.run_generation(key, prompt, n, temp, topk, seed)

    @work(thread=True, exclusive=True)
    def run_generation(self, key, prompt, n, temp, topk, seed):
        out = self.query_one("#out", Static)
        buf = [prompt]
        self.call_from_thread(out.update, prompt)
        try:
            if key not in self.models:
                self.models[key] = MODELS[key][1](lambda m: self.call_from_thread(self.status, m))
            model = self.models[key]
            t0 = time.time(); count = [0]

            def on_char(c):
                buf.append(c); count[0] += 1
                if count[0] % 4 == 0 or c == "\n":
                    text = "".join(buf)
                    self.call_from_thread(out.update, text)
                    self.call_from_thread(self.query_one("#out_scroll", VerticalScroll).scroll_end, animate=False)

            self.call_from_thread(self.status, f"{model.name}: reading the prompt...")
            generate(model, prompt, n, temp, topk, seed, on_char=on_char, stop=lambda: self._stop)
            self.call_from_thread(out.update, "".join(buf))
            dt = time.time() - t0
            self.call_from_thread(self.status, f"{model.name}: {count[0]} characters in {dt:.1f}s "
                                               f"({(count[0] + len(prompt)) / max(dt, 1e-9):.0f} chars/s incl. prompt)")
        except Exception as e:                                   # noqa: BLE001
            self.call_from_thread(self.status, f"error: {e}")
        finally:
            self.busy = False


def main():
    GPFApp().run()
