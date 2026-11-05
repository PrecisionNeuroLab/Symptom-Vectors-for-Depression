# Interpretable Symptom Vectors for Depression in Large Language Models

Code for v2 of the paper (ten models). The single-model arXiv preprint v1 is in `code/v1/`.

Run from `code/`:

```
pip install -r requirements.txt
```

| File | Purpose |
| --- | --- |
| `81.multimodel_rs_generation.ipynb` | Full-depth residual streams for each model |
| `82.multimodel_comparison.ipynb` | All analyses, figures, and tables |
| `82.*.csv` | Precomputed PERMANOVA, PERMDISP, and operating layers |
| `symptom_vectors/` | Released vectors per model; usage in its README |

Gemma checkpoints are gated on Hugging Face: accept the terms and run `huggingface-cli login`.
Distance matrices and residual streams are too large to include; `82` regenerates them.
ReDSM5 and the book excerpts cannot be redistributed, so the held-out analyses need them
assembled separately (see the paper's Data and Models section).
