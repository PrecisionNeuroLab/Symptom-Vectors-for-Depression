import gc
import os
import warnings
from collections import defaultdict, OrderedDict
from functools import partial

from torch import cuda, save
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import float32
from utils import text_to_tokens

warnings.simplefilter(action='ignore', category=Warning)

IN_FOLDER = 'core_clinical_short/'
OUT_FOLDER = 'core_clinical_short_gemma_27b/'

MODEL = "google/gemma-3-27b-pt"
cuda.empty_cache()
gc.collect()
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="mps",
    dtype="auto",
    trust_remote_code=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL)

os.makedirs(OUT_FOLDER, exist_ok=True)
layers = list(range(len(model.language_model.layers) + 1))
print(str(layers) + ' layers with embedding layer!')
text_files = os.listdir(IN_FOLDER)
tensors, unit_tensors = defaultdict(dict), defaultdict(dict)
tokens = defaultdict(list)

# Save residual stream.
for t in tqdm(text_files):
    out_path = os.path.join(OUT_FOLDER, f"{t}.pt")
    if os.path.exists(out_path):
        print(f"Skipping residual stream (already exists): {out_path}")
        continue

    raw = []
    with open(IN_FOLDER + t, 'r', encoding='utf-8') as file:
        text = file.read()

    print("Generating residual stream for " + t)
    def hook(module, input, output, layer_id):
        if layer_id == 0:  # put in initial embeddings
            raw.append(input[0].squeeze().to(float32)) # move to CPU for local processing
        raw.append(output[0].squeeze().to(float32))

    for l in tqdm(layers[:-1]):
        model.language_model.layers[l]._forward_hooks = OrderedDict()  # clear all the old hooks first
        model.language_model.layers[l].register_forward_hook(partial(hook, layer_id=l))

    model.model(tokenizer(text, return_tensors='pt').input_ids.to(model.device)) # trigger forward pass
    save(raw, out_path)
    print("Saved: " + out_path)

    cuda.empty_cache()
    gc.collect()

# Save list of token.
for t in text_files:
    with open(IN_FOLDER + t, 'r', encoding='utf-8') as file:
        text = file.read()
    tokens = text_to_tokens(tokenizer, text)
    print(tokens)
    save(tokens, OUT_FOLDER + f'{t}_tokens.pt')
    print(OUT_FOLDER + f'{t}_tokens.pt')
