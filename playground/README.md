# GPF: Generative Pretrained Fly

*Pretrained by evolution. Fine-tuned on 20k characters of Shakespeare.*

A small terminal app from the *Fly-Is-All-You-Need* project. (The name is a joke: the only
pretraining the fly connectome ever had is evolution, and it writes much worse than a GPT.) Pick a character model,
type a prompt, and watch it write on. The prompt is fed to the model one character at a
time (that is how the context gets in), then the model samples the next character, feeds
it back, and so on, the way early character-level language models were demonstrated.

```
pip install -r requirements.txt
python -m gpf                     # terminal UI (ctrl+g generate, esc stop, ctrl+q quit)
python -m gpf --cli --model gru --prompt "ROMEO:\n" --n 300 --temp 0.7
```

## Models

| key | model | needs |
|---|---|---|
| `kn7` | Kneser-Ney character 7-gram, fitted at start-up on 1M characters of TinyShakespeare (~10 s) | numpy |
| `kn5-20k` | Kneser-Ney 5-gram on 20k characters, the same data the connectome readout saw | numpy |
| `gru` | 1-layer GRU (hidden 256) trained on 1M characters (bundled weights) | torch |
| `brain` | the whole MaleCNS v1.0 connectome (166,700 leaky integrate-and-fire neurons) simulated one character at a time on the GPU, with the trained linear readout bundled here | CUDA GPU, the `flybrain` simulator, the MaleCNS data and this project's `mechanism/` code (set `FLYBRAIN_HOME` and `GPF_MECHANISM`) |

The `brain` model is the point of the exercise and also the weakest writer. On a laptop GPU
it produces about 20 characters per second, because each character runs the full
connectome for 120 ms of simulated time. Its text is Shakespeare-shaped but mostly not real
words, worse than a 5-gram trained on the same 20k characters. The accompanying report
(`paper_lm/`) explains why. With this input protocol and a linear readout, the connectome
behaves like a fading memory of about the last four characters, which is no more than a
5-gram knows. A degree-preserving rewired connectome does as well as the real one.

## Sample (prompt `ROMEO:`, temperature 0.7, seed 0)

**brain (20k characters)**
```
ha  turn tr the shalf the you.

LARTIUS:
I' the as fragent Roman now I
thrirlywerGere: yetty truduse,
```
**kn5-20k**
```
Shadow't.
I
tencorn answeryments of my lord friends you must have breast valive: and the rivesticurse to the re's
```
**kn7**
```
O, 'twas us
fellow of the young Princes have the triumph? And blessing thou owed not speak; I would make me but the way
```
**gru**
```
Shall I put was the bastards prepare to make the state and a far thoughts;
And so look unto any fury now of all
```

## Data

`data/tinyshakespeare.txt` is the TinyShakespeare corpus (Karpathy, char-rnn; the text is
Shakespeare, public domain). `data/brain_readout_20k/` is the readout from
`models/lm_s160` in the repository.
