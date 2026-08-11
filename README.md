# Personal Writing-Style Fine-Tuner — LoRA on Phi-3-mini-4k-instruct

Fine-tuning a small instruction model to write in **one specific author's voice**,
from only 75 style samples — and measuring whether it actually worked.

**Result: +0.12 mean cosine similarity (0.58 → 0.70) over a few-shot-prompted
GPT-5 baseline**, measured with SBERT against held-out samples of the target style.

[Demo video](https://drive.google.com/file/d/1OL_Yz3S_0qlk7k7piOPfmm0SPDV-voQD/view?usp=sharing)
· [Full write-up](REPORT.md)

---

## Why

I write for my institute's media body. LLM drafts are unusable for that —
the register is unmistakably artificial, and "humanisers" fix the register by
degrading everything else. Prompting alone doesn't transfer a voice; it
approximates one.

So the question this project asks is narrow and testable: **with 75 samples and
a 3.8B model, can parameter-efficient fine-tuning beat few-shot prompting of a
much larger model at style transfer?**

## Approach

| Stage | What happens |
|---|---|
| **1. Data preparation** | Own articles → sentence split → filter fragments under 5 words → tag as `[STYLE_SAMPLE]` |
| **2. Pair construction** | Each style sentence is rewritten by an LLM into neutral "GPT-voice" prose. That gives aligned `(neutral → my voice)` pairs — the inverse direction of the task, which is what makes 75 samples enough |
| **3. Fine-tuning** | LoRA adapters on `microsoft/Phi-3-mini-4k-instruct`, in the Phi-3 chat template (`<\|system\|>` / `<\|user\|>` / `<\|assistant\|>`) |
| **4. Evaluation** | SBERT (`all-MiniLM-L6-v2`) cosine similarity of generated text against real style references, compared against the base model and a few-shot GPT-5 baseline |

The pairing step in stage 2 is the part that carries the project. Rather than
training on 75 raw samples, generating a neutral counterpart for each turns an
unpaired style corpus into a supervised rewriting task.

## Results

| Model | Mean cosine similarity | Δ vs. baseline |
|---|---|---|
| Few-shot prompted GPT-5 | 0.58 | — |
| **Fine-tuned Phi-3 (LoRA)** | **0.70** | **+0.12** |

A 3.8B model with LoRA adapters beats few-shot prompting of a frontier model on
this specific style-transfer task.

### Honest limitations

- **n = 75**, single author, single domain. The result is a signal, not a
  benchmark.
- SBERT cosine similarity rewards semantic and register proximity — it is a
  proxy for "sounds like me", not a measurement of it. No human eval was run.
- The evaluation references come from the same corpus family as the training
  samples, so some stylistic leakage is likely.

## Repository layout

```
data_creation/   corpus prep + neutral-rewrite pair generation   (see its README)
training/        LoRA fine-tuning of Phi-3-mini                  (see its README)
testing/         SBERT similarity evaluation                     (see its README)
REPORT.md        full architecture + data-science write-up
```

## Running it

The notebooks are written for Colab (GPU required for the training step).

```bash
pip install -q -U google-generativeai langchain langchain_google_genai
```

The data-creation notebook calls the Gemini API. Provide the key at runtime —
it is read from the environment, and prompted for via `getpass` if unset:

```python
import os
os.environ["GEMINI_API_KEY"] = "..."   # or let the notebook prompt you
```

Never hard-code the key into a notebook cell — including as the prompt string
passed to `input()`, which gets echoed into saved cell output.

Then run in order: `data_creation/` → `training/` → `testing/`.

## Built with

`transformers` · `peft` (LoRA) · `sentence-transformers` · `langchain` ·
Phi-3-mini-4k-instruct · Gemini API

---

**Goutam Singh** — IIT Bombay
