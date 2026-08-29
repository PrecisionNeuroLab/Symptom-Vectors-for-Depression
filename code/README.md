# Code

Analysis code for *Symptom Vectors for Depression* (arXiv preprint). Run everything
from inside this directory.

```
pip install -r requirements.txt
```

## Residual stream extraction

Both scripts load `google/gemma-3-27b-pt` (62 blocks, d_model = 5,376) with
Hugging Face Transformers on Apple MPS, run one forward pass per text file, and
save the activations plus the token list next to them. Set `IN_FOLDER` /
`OUT_FOLDER` at the top of the script for the corpus you want.

| File | What it saves |
| --- | --- |
| `text_to_tensors_mac.py` | All 63 residual stream points (input embeddings + every block output) as `<text>.pt`. Input to the per-layer separability sweep. |
| `text_to_tensors_mac_21.py` | Layer 21 only (the operating layer), as `<text>_tensor`. Input to both projection notebooks. |

The leading `<bos>` activation is discarded in the analysis code (`[1:]`), not at
extraction time.

`utils.py` holds the shared primitives (`centroid`, `normalize`, `cos_dist`,
`euc`, `text_to_tokens`, …); `test_utils.py` is its unit-test suite
(`python -m unittest test_utils`).

## Notebooks

| Notebook | Paper output |
| --- | --- |
| `74.short_separation.ipynb` | Per-layer separability of the three symptom groups: 8 distance metric × normalization combinations, PERMANOVA gated by PERMDISP, 9,999 permutations. Produces Fig. 2, Table 1. |
| `75.short_projection_gram.ipynb` | Symptom Vectors at layer 21 and the Gram-pseudoinverse-decorrelated projection of held-out text onto them. Produces Fig. 3 (a–d), Fig. 4, and Table 2. |
| `79.short_projection_contrastive_depression.ipynb` | The single Depression Vector: `centroid(core_clinical) − centroid(positive_affect)`, scored by cosine similarity, with Mann-Whitney AUC for the held-out depressive vs. happy contrast. Produces Fig. 5. |

Each notebook has a `NEW` flag or an equivalent recompute cell. `74` ships with
its precomputed statistics (`74.permanova.csv`, `74.anosim_pd.csv`,
`74.permdisp.csv`) so the figures and tables reproduce without rerunning the
permutation sweep; set `NEW = True` to recompute from the distance matrix.
Figure export writes PDFs into a `manuscript/` directory — create one first if
you want the exports.

## Corpora

Included here: `core_clinical_short/` and `positive_affect/` (plain text) with
their layer-21 activations in `core_clinical_short_gemma3_27b_21/` and
`positive_affect_gemma3_27b_21/`, plus HappyDB (`happydb/`,
`happydb_gemma3_27b_21/`).

Not redistributable, and therefore absent — see the Data availability statement
in the paper: the raw clinical instrument text, the ReDSM5 corpus (`redsm5/`),
and the *Darkness Visible* / *Handbook of Depression* excerpts (`books/`). The
cells in `75` and `79` that read those folders will not run without them; the
Methods give the citations needed to assemble an equivalent corpus, which can
then be passed through `text_to_tensors_mac_21.py`.
