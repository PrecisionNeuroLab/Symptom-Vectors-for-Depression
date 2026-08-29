import numpy as np
import torch.nn.functional as f
from pandas import DataFrame
from torch import Tensor, topk
from torch import cdist
from transformers import AutoModelForCausalLM


def figures_to_html(figs, filename="dashboard.html"):
    with open(filename, 'w', encoding='utf-8') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")


def cos_dist(emb1, emb2):
    # Ensure the value is within [-1, 1] due to floating point errors
    # TODO: Copilot caught this, read further about floating point errors
    cos_sim = f.cosine_similarity(emb1, emb2, dim=0).item()
    return 1 - max(min(cos_sim, 1.0), -1.0)


def min_max_normalize_rows(df):
    norm_df = DataFrame(index=df.index, columns=df.columns)
    for idx in df.index:
        row = df.loc[idx].astype(float)
        min_val = row.min()
        max_val = row.max()
        if max_val != min_val:
            norm_df.loc[idx] = (row - min_val) / (max_val - min_val)
        else:
            norm_df.loc[idx] = np.nan  # or 0 if preferred
    return norm_df


def euc(emb1, emb2):
    return cdist(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()


def remove_first_vector(emb: Tensor):
    return emb[1:]


def normalize(emb: Tensor):
    return f.normalize(emb, p=2, dim=-1, eps=1e-12)


def centroid(emb: Tensor):
    return emb.mean(dim=0)


def magnitude(emb: Tensor):
    return emb.norm().item()


def jaccard_distance(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return 1 - intersection / union


def text_to_tokens(tokenizer, text):
    tokens = []
    criteria_tokens = tokenizer(text, return_tensors='pt').input_ids.squeeze().tolist()
    for i in range(len(criteria_tokens)):
        word = tokenizer.decode(criteria_tokens[i])
        tokens.append(word)
    return tokens


def top_cosine_neighbors(query: Tensor, embeddings: Tensor):
    """
    Generator yielding (index, distance) pairs ordered by increasing cosine distance to query.
    Args:
        query: Tensor of shape (D,)
        embeddings: Tensor of shape (N, D)
    Yields:
        (index, distance): index as int, distance as float
    """
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape (N, D)")
    if query.dim() != 1:
        raise ValueError("query must have shape (D,)")

    if embeddings.size(0) == 0:
        return

    # Vectorized cosine distance: 1 - clamp(cosine\_similarity, [-1, 1])
    query_exp = query.unsqueeze(0).expand_as(embeddings)
    sims = f.cosine_similarity(embeddings, query_exp, dim=1).clamp(-1.0, 1.0)
    dists = 1 - sims

    # Sort ascending by distance and yield one by one
    N = dists.size(0)
    distances, indices = topk(dists, k=N, largest=False, sorted=True)
    for idx, dist in zip(indices.tolist(), distances.tolist()):
        yield idx, float(dist)


def top_euc_neighbors(query: Tensor, embeddings: Tensor):
    """
    Generator yielding (index, distance) pairs ordered by increasing Euclidean distance to query.
    Args:
        query: Tensor of shape (D,)
        embeddings: Tensor of shape (N, D)
    Yields:
        (index, distance): index as int, distance as float
    """
    if embeddings.dim() != 2:
        raise ValueError("embeddings must have shape (N, D)")
    if query.dim() != 1:
        raise ValueError("query must have shape (D,)")

    if embeddings.size(0) == 0:
        return

    # Vectorized Euclidean distances to the query
    dists = cdist(embeddings, query.unsqueeze(0)).squeeze(1)  # shape (N,)

    # Sort ascending by distance and yield
    N = dists.size(0)
    distances, indices = topk(dists, k=N, largest=False, sorted=True)
    for idx, dist in zip(indices.tolist(), distances.tolist()):
        yield idx, float(dist)


def bg(n01: float, text: str) -> str:
    """
    Colorize `text` with a background picked by a float in [0, 1].
    """
    try:
        x = float(n01)
    except (TypeError, ValueError):
        x = 0.0
    if x != x:  # NaN guard
        x = 0.0
    x = max(0.0, min(1.0, x))
    n = int(round(x * 255))
    return f"\x1b[48;5;{n}m{text}\x1b[0m"


def load_model(model: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        device_map=device,
        dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model
