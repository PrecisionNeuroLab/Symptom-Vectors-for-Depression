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

IN_FOLDER = 'books/'
OUT_FOLDER = 'books_gemma3_27b_21/'

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
if '.ipynb_checkpoints' in text_files:
    text_files.remove('.ipynb_checkpoints')
tensors, unit_tensors = defaultdict(dict), defaultdict(dict)
tokens = defaultdict(list)

# Save residual stream.
for t in tqdm(text_files):
    out_path = os.path.join(OUT_FOLDER, f"{t}_tensor")
    if os.path.exists(out_path):
        print(f"Skipping residual stream (already exists): {out_path}")
        continue

    raw = []
    with open(IN_FOLDER + t, 'r', encoding='utf-8') as file:
        text = file.read()

    print("Generating residual stream for " + t)
    def hook(module, input, output, layer_id):
        if layer_id == 21:  # save layer 21 only
            raw.append(input[0].squeeze().detach().cpu()) # move to CPU for local processing

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
    save(tokens, OUT_FOLDER + f'{t}_tokens')
    print(OUT_FOLDER + f'{t}_tokens')
