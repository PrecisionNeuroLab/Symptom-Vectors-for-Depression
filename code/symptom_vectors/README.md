# Symptom Vectors and Depression Vectors

One file per model, `<model>_layer<L>.npz`, written by `82.multimodel_comparison.ipynb` at the model's
operating layer `L`. Every array is float64.

| Key | Shape | Contents |
|---|---|---|
| `model` | | Hugging Face model id |
| `layer` | | operating layer: the residual stream entering decoder block `L` (`hidden_states[L]` in Transformers; 0 = embeddings) |
| `symptoms` | (3,) | axis order: mood, somatic, suicidality |
| `symptom_vectors` | (3, d) | Symptom Vectors `V`: token-weighted Core Clinical centroid per symptom group |
| `gram_inv` | (3, 3) | inverse of the Gram matrix `G = V Vᵀ` |
| `depression_vector` | (d,) | mean of the three Symptom Vectors minus the Positive Affect centroid |

To score a text, run it through the model as raw text (no chat template), take the layer-`L`
residual stream, drop the first (beginning-of-sequence) position, and average the remaining tokens
in float64 to get its centroid `c`. Then

```python
import numpy as np

v = np.load('gemma-3-27b-pt_layer21.npz')
beta = v['gram_inv'] @ (v['symptom_vectors'] @ c)            # (mood, somatic, suicidality)
d = v['depression_vector']
score = c @ d / (np.linalg.norm(c) * np.linalg.norm(d))        # Depression Score
```

OLMo-2's tokenizer adds no beginning-of-sequence token; prepend `tokenizer.bos_token` so the first
position can be dropped as for the other models.
