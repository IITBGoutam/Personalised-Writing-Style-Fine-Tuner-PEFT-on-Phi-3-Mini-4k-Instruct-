# Data Science Report

## 2.2 Architectural Flow

```
 ┌────────────────────────────────────────────────────────┐
 │                 1. Data Preparation                    │
 │  User Articles → Cleaned Sentences → [STYLE_SAMPLE]     │
 │  GPT Rewrites → Paired with User Style Sentences        │
 └────────────────────────────────────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────────────────────┐
 │             2. Dataset Creation (Few-shot pairs)       │
 │  Each example:                                         │
 │  <|system|> You are a writing assistant... <|end|>     │
 │  <|user|> GPT-style text <|end|>                       │
 │  <|assistant|> User-style rewrite <|end|>              │
 └────────────────────────────────────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────────────────────┐
 │              3. Fine-tuning (PEFT LoRA)                │
 │  Model: Phi-3-mini-4k-instruct                         │
 │  Parameters trained: Low-rank adapter matrices         │
 │             							    │
 └────────────────────────────────────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────────────────────┐
 │            4. Evaluation Pipeline                      │
 │  Generate text from a fine-tuned model                 │
 │  Compute SBERT similarity with style reference         │
 │  Compare against base model outputs                    │
 └────────────────────────────────────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────────────────────┐
 │            5. Inference Agent                          │
 │  Accepts any GPT-like text → outputs user-style rewrite│
 │                                                        │
 └────────────────────────────────────────────────────────┘
```

## 3.1 Dataset
•	Source: User’s own writings (75 text samples)
•	Preprocessing:
o	Split into sentences using a period-based tokeniser.
o	Remove short (<5 words) fragments.
o	Format as [STYLE_SAMPLE]\n<sentence>.\n\n
•	Final Dataset:
o	75 style samples
o	Paired with ChatGPT-style neutral rewrites → produces (input → output) fine-tuning pairs
## 3.2 Evaluation Methodology

**Quantitative (objective) — SBERT style similarity**

- Model: `all-MiniLM-L6-v2`
- Metric: average cosine similarity between generated outputs and real
  user-style sentences.
- Baseline: few-shot prompted GPT-5, plus the untuned base model against the
  same style references.

| Model | Mean cosine similarity | Δ vs. baseline |
|---|---|---|
| Few-shot prompted GPT-5 | 0.58 | — |
| Fine-tuned Phi-3 (LoRA) | 0.70 | +0.12 |

