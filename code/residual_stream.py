import gc
import os
import sys
import warnings
from functools import partial

import torch
import transformers
from torch import float32, inference_mode, save
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.simplefilter(action='ignore', category=Warning)


_HAS_DTYPE_KWARG = tuple(int(v) for v in transformers.__version__.split(".")[:2]) >= (4, 56)


def _default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _empty_cache(device):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()


def _decoder_layers(model):
    # Multimodal Gemma 3 (4b/12b/27b) keeps the text stack under language_model;
    # text-only causal LMs (e.g. gemma-3-1b, Llama) keep it under model.layers.
    for path in ("language_model.layers", "model.language_model.layers", "model.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    raise AttributeError(f"Could not locate decoder layers on {type(model).__name__}")


def input_ids_with_bos(tokenizer, text):
    """
    Token ids for text with BOS at position 0. Downstream analysis drops position 0 as the
    BOS / attention sink, so tokenizers that don't add one themselves (e.g. OLMo-2, whose
    bos_token is the <|endoftext|> document separator) get their bos_token prepended.
    """
    ids = tokenizer(text).input_ids
    if ids[0] != tokenizer.bos_token_id:
        ids = [tokenizer.bos_token_id] + ids
    return ids


def output_folder_name(text_folder, model_name, layer=None):
    """
    'core_clinical_short/', 'google/gemma-3-27b-pt'     -> 'core_clinical_short_gemma-3-27b-pt'
    'core_clinical_short/', 'google/gemma-3-27b-pt', 21 -> 'core_clinical_short_gemma-3-27b-pt_layer21'
    """
    folder = os.path.basename(os.path.normpath(text_folder))
    model = model_name.rstrip("/").split("/")[-1]
    name = f"{folder}_{model}"
    return name if layer is None else f"{name}_layer{layer}"


def generate_residual_stream(model_name, text_folder, layer=None, device=None, out_root=".",
                             dtype="auto"):
    """
    Run every text file in text_folder through model_name and save its residual stream.

    layer=None writes the full stream to <out_root>/<text_folder>_<model>/, for each file <t>:
        <t>.pt         list of (seq_len, d_model) float32 tensors, one per layer
                       (index 0 = input embeddings, then one per decoder layer)
        <t>_tokens.pt  list of decoded token strings

    layer=N writes only index N of that list to <out_root>/<text_folder>_<model>_layer<N>/,
    in the format of the existing single-layer folders (model dtype, no file extension):
        <t>_tensor     [ (seq_len, d_model) tensor ]
        <t>_tokens     list of decoded token strings

    dtype="auto" uses the checkpoint's stored dtype; pass torch.bfloat16 to load float32
    checkpoints (e.g. OLMo-2-0325-32B) in bf16.

    Files whose residual stream already exists are skipped. Returns the output folder path.
    """
    device = device or _default_device()
    out_folder = os.path.join(out_root, output_folder_name(text_folder, model_name, layer))
    tensor_suffix, tokens_suffix = (".pt", "_tokens.pt") if layer is None else ("_tensor", "_tokens")
    os.makedirs(out_folder, exist_ok=True)

    _empty_cache(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        # transformers >= 4.56 reads dtype, older versions read torch_dtype and ignore dtype.
        **({"dtype": dtype} if _HAS_DTYPE_KWARG else {"torch_dtype": dtype}),
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    decoder_layers = _decoder_layers(model)
    print(f"{len(decoder_layers) + 1} layers with embedding layer! ({model.dtype} on {model.device})")
    if layer is not None and not 0 <= layer <= len(decoder_layers):
        raise ValueError(f"layer must be in [0, {len(decoder_layers)}], got {layer}")
    text_files = sorted(f for f in os.listdir(text_folder)
                        if os.path.isfile(os.path.join(text_folder, f)))

    # Save residual stream.
    for t in tqdm(text_files, desc=model_name.split('/')[-1]):
        out_path = os.path.join(out_folder, t + tensor_suffix)
        if os.path.exists(out_path):
            continue

        with open(os.path.join(text_folder, t), 'r', encoding='utf-8') as file:
            text = file.read()

        raw = []

        def hook(module, input, output, layer_id):
            if layer_id == 0:  # put in initial embeddings
                raw.append(input[0].squeeze().to(float32))
            hidden = output[0] if isinstance(output, tuple) else output
            raw.append(hidden.squeeze().to(float32))

        def layer_hook(module, input, output):
            # Stream index N is the input to decoder layer N (the output of layer N-1).
            raw.append(input[0].squeeze().detach())

        if layer is None:
            handles = [dl.register_forward_hook(partial(hook, layer_id=l))
                       for l, dl in enumerate(decoder_layers)]
        elif layer < len(decoder_layers):
            handles = [decoder_layers[layer].register_forward_hook(layer_hook)]
        else:  # final layer output
            handles = [decoder_layers[-1].register_forward_hook(
                lambda m, i, o: raw.append((o[0] if isinstance(o, tuple) else o).squeeze().detach()))]
        try:
            with inference_mode():
                input_ids = torch.tensor([input_ids_with_bos(tokenizer, text)], device=model.device)
                model.model(input_ids)  # trigger forward pass, skip lm_head
        finally:
            for h in handles:
                h.remove()
        save([r.cpu() for r in raw], out_path)

        del raw
        _empty_cache(device)

    # Save list of tokens.
    for t in text_files:
        with open(os.path.join(text_folder, t), 'r', encoding='utf-8') as file:
            text = file.read()
        tokens_path = os.path.join(out_folder, t + tokens_suffix)
        save([tokenizer.decode(i) for i in input_ids_with_bos(tokenizer, text)], tokens_path)

    del model
    _empty_cache(device)
    return out_folder


def corpus_files(text_folder):
    """
    Sorted text filenames in text_folder. generate_residual_stream runs every regular file, so a
    stray .DS_Store would be "generated" too: refuse hidden files instead.
    """
    files = sorted(f for f in os.listdir(text_folder) if os.path.isfile(os.path.join(text_folder, f)))
    hidden = [f for f in files if f.startswith(".")]
    assert not hidden, f"hidden files in {text_folder}: {hidden}; delete them before generating"
    return files


def _suffixes(layer):
    """(tensor, tokens) filename suffixes generate_residual_stream writes for this layer."""
    return (".pt", "_tokens.pt") if layer is None else ("_tensor", "_tokens")


def is_complete(model_name, text_folder, layer=None):
    """True if every text in text_folder already has its residual stream saved."""
    out = output_folder_name(text_folder, model_name, layer)
    return all(os.path.exists(os.path.join(out, t + _suffixes(layer)[0])) for t in corpus_files(text_folder))


def generate_missing(jobs, dtype="auto"):
    """
    Run generate_residual_stream for each (model_name, text_folder, layer) job, skipping jobs whose
    output is already complete, so no model is loaded just to find nothing left to do. A failure
    is recorded and the loop moves on. Returns {(model_name, text_folder): error}.
    """
    failed = {}
    for model_name, text_folder, layer in jobs:
        if is_complete(model_name, text_folder, layer):
            continue
        print(f"\n===== {model_name} / {text_folder} / {'full stream' if layer is None else f'layer {layer}'} =====")
        try:
            generate_residual_stream(model_name, text_folder, layer=layer, dtype=dtype)
        except Exception as e:
            failed[(model_name, text_folder)] = repr(e)
            print(f"FAILED {model_name} / {text_folder}: {e!r}")
    print("\nfailed:", failed)
    return failed


def check_residual_stream(model_name, text_folder, width, layer=None, blocks=None):
    """
    Problems in one output folder (empty list = ok): every text present, one tensor per layer
    (blocks + 1 for a full stream, 1 for a single layer), shape (n_tokens, width), finite values.
    A single-layer tensor must also equal its index in the full stream exactly, where that stream
    is on this machine (same forward pass, saved in the model dtype vs float32).
    """
    out, full = output_folder_name(text_folder, model_name, layer), output_folder_name(text_folder, model_name)
    tensor_suffix, tokens_suffix = _suffixes(layer)
    n_layers = blocks + 1 if layer is None else 1
    problems = []
    for t in corpus_files(text_folder):
        path = os.path.join(out, t + tensor_suffix)
        if not os.path.exists(path):
            problems.append(f"missing {t}")
            continue
        rs = torch.load(path, map_location="cpu")
        tokens = torch.load(os.path.join(out, t + tokens_suffix))
        full_path = os.path.join(full, t + ".pt")
        if len(rs) != n_layers:
            problems.append(f"{t}: {len(rs)} layers")
        elif any(r.shape != (len(tokens), width) for r in rs):
            problems.append(f"{t}: shape {tuple(rs[0].shape)} vs {len(tokens)} tokens")
        elif not all(torch.isfinite(r).all() for r in rs):
            problems.append(f"{t}: non-finite values")
        elif (layer is not None and os.path.exists(full_path)
              and not torch.equal(torch.load(full_path, map_location="cpu")[layer], rs[0].float())):
            problems.append(f"{t}: differs from full-stream index {layer}")
    return problems


if __name__ == "__main__":
    # python residual_stream.py google/gemma-3-27b-pt core_clinical_short [layer]
    generate_residual_stream(sys.argv[1], sys.argv[2], *(int(a) for a in sys.argv[3:4]))
