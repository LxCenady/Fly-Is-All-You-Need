<p align="center"><img src="gpf/assets/gpf-app-icon.svg" width="112" alt="GPF logo"></p>

# GPF: Generative Pretrained Fly

**A text generator whose brain is a simulated fruit fly.**
*Pretrained by evolution. Fine-tuned on 20k characters of Shakespeare.*

Type the start of a line and GPF writes on, one character at a time, like the early
text generators that came before ChatGPT. The star of the show is **GPF-1**, a complete
fruit-fly brain (166,700 simulated neurons, wired exactly as in the real animal) that
reads your text letter by letter and guesses what comes next.

Is it any good? No. The fly writes the worst Shakespeare in the app, and that is the point.
Put it side by side with a simple word-statistics model and a small neural network and you
can see for yourself how much (and how little) a real brain's wiring gives you. The research
behind it is in [`paper_lm/`](../paper_lm).

## Get it

**Windows:** download `gpf-lite-…-windows-x64.exe` from
[Releases](https://github.com/LxCenady/Fly-Is-All-You-Need/releases) and double-click it.
GPF opens in your browser.

**Ubuntu / Debian:** download `gpf-lite_…_amd64.deb`, then
```
sudo apt install ./gpf-lite_*_amd64.deb
gpf
```

Nothing else to install: Python and everything GPF needs are inside. (Your antivirus may take
a second look at the .exe the first time; it is an unsigned app built by GitHub Actions from
this repository.)

## Using it

- Pick a model on the left, type the start of a line (or click a suggestion) and press Enter.
- **Temperature** sets how adventurous the writing is. Low is safe and repetitive; high is
  wild and full of made-up words.
- **Characters to write** sets the length. **Stop** ends a reply early.
- **Feed the whole conversation back as context** lets the model continue from everything
  so far instead of only your last message.
- GPF continues text. It is not a chatbot and will not answer questions.

Prefer the terminal? `gpf --cli --model gru --prompt "ROMEO:\n" --n 300`

## The models

| Model | What it is | Speed | In the Lite download |
|---|---|---|---|
| **GPF-1 (fly connectome)** | The whole fruit-fly brain, simulated for every character | ~20–30 characters/s, needs an NVIDIA GPU | no, see below |
| Kneser-Ney 7-gram | Classic word-statistics model: which character usually follows the last six | instant | yes |
| Kneser-Ney 5-gram, 20k | The same idea, trained on exactly the text the fly saw | instant | yes |
| GRU | A small trained neural network, the best writer here | instant | yes |

## Running the fly (GPF-1)

GPF-1 is built on **[flybrain](https://github.com/alextitonis/fly.ai)** by Alex Titonis
(MIT License). flybrain simulates the complete fruit-fly nervous system and is a **required
dependency**: without it, GPF-1 does not run. The Lite downloads leave it out and show GPF-1
as unavailable.

To run GPF-1 you need an **NVIDIA GPU** with a recent driver, and flybrain with GPU support:

```
pip install -r requirements.txt
pip install "flybrain[gpu]==0.1.0"
python -m gpf --web               # then pick GPF-1 on the left
```

The first time GPF-1 starts, flybrain downloads its prebuilt copy of the **MaleCNS v1.0
connectome** (about 260 MB, CC BY 4.0, FlyEM / HHMI Janelia and partners) into `~/fly-data`
(set `FLY_DATA` to put it elsewhere). It is the exact data GPF-1 was trained on; the
checksums match. GPF-1 runs at about 20–30 characters per second on a laptop GPU.

**Watch it think.** While GPF-1 writes, the *Inside the fly* panel on the right shows the brain
from the front: every neuron is a faint dot, and the ones firing on the current character
light up (odour-input neurons cyan, Kenyon cells yellow, mushroom-body outputs green,
dopamine neurons magenta), with live counts per group. The look follows the dashboard that
ships with fly.ai. Hide it with the button in the header if your GPU is struggling.

## Run from source

```
pip install -r requirements.txt   # numpy, plus textual for the terminal UI
python -m gpf --web               # web UI at http://127.0.0.1:8765 (your computer only)
python -m gpf                     # terminal UI
```

## Credits

- Fly brain simulation: [flybrain](https://github.com/alextitonis/fly.ai) (MIT).
- Connectome: MaleCNS v1.0, S. Berg et al., *Cell* 189(18), 2026 (CC BY 4.0).
- Brain view: after the fly.ai dashboard (`sshfighter/fly_dashboard.py`, MIT).
- Text: TinyShakespeare from Andrej Karpathy's char-rnn; Shakespeare is public domain.
- Full licence list: [NOTICE.md](NOTICE.md).
