"""
Belebele eval for larger multilingual models. Assume Belebele jsonl files are in
[BELEBELE_DIR]/[lang].jsonl. Run on GPU.

Outputs a TSV of Belebele accuracies per language.
"""

import codecs
from collections import defaultdict
import json
import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import PeftModel  # Only needed for Mala500.


# Tested for: MaLA-LM/mala-500-10b, bigscience/bloom-7b1, facebook/xglm-4.5B,
# facebook/xglm-7.5B.
MODEL_NAME = 'bigscience/bloom-7b1'
BELEBELE_DIR = 'belebele_eval'
OUTPATH = 'belebele_results/bloom7b1.tsv'
MAX_SEQ_LEN = 512


langs = [fname.lower().replace('.jsonl', '') for fname in os.listdir(BELEBELE_DIR)]
cache_dir = 'hf_cache'
# Open outfile and write header.
with codecs.open(OUTPATH, 'wb', encoding='utf-8') as outfile:
    outfile.write('lang\tacc\n')

print('Loading model.')
if MODEL_NAME == 'MaLA-LM/mala-500-10b':
    base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', cache_dir=cache_dir)
    base_model.resize_token_embeddings(260164)
    tokenizer = AutoTokenizer.from_pretrained('MaLA-LM/mala-500-10b', cache_dir=cache_dir)
    model = PeftModel.from_pretrained(base_model, 'MaLA-LM/mala-500-10b', cache_dir=cache_dir).to('cuda')
else:
    config = AutoConfig.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, config=config, cache_dir=cache_dir).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)

for belebele_lang in sorted(langs):
    print('Loading dataset.')
    inpath = os.path.join(DATASET_DIR, belebele_lang + '.jsonl')
    dataset = []
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    for line in infile:
        dataset.append(json.loads(line.strip()))
    infile.close()
    assert len(dataset) == 900  # Expected number of examples.

    print('Running model.')
    # Model-specific BOS token.
    if MODEL_NAME in ['facebook/xglm-4.5B', 'facebook/xglm-7.5B']:
        # ID 2 for XGLM 4.5b and 7.5b, as is default with their tokenizer.
        prepend_token_id = tokenizer.eos_token_id
    elif MODEL_NAME in ['MaLA-LM/mala-500-10b', 'bigscience/bloom-7b1']:
        prepend_token_id = tokenizer.bos_token_id
    else:
        prepend_token_id = tokenizer.cls_token_id
    pad_token_id = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    is_correct = []
    loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
    for r in tqdm(dataset):
        prefix = r['flores_passage'].strip() + ' ' + r['question'].strip() + ' '
        mc_texts = []
        for answer_i in range(4):
            mc_texts.append(prefix + r[f'mc_answer{answer_i+1}'].strip())
        correct_i = int(r['correct_answer_num']) - 1

        # Run model.
        option_losses = torch.zeros(len(mc_texts))
        for mc_i, mc_text in enumerate(mc_texts):
            inputs = tokenizer([mc_text], add_special_tokens=False)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            input_ids[0].insert(0, prepend_token_id)
            attention_mask[0].insert(0, 1)
            if len(input_ids[0]) > MAX_SEQ_LEN:
                input_ids[0] = input_ids[0][:MAX_SEQ_LEN]
                attention_mask[0] = attention_mask[0][:MAX_SEQ_LEN]
            input_ids = torch.tensor(input_ids).cuda()
            attention_mask = torch.tensor(attention_mask).cuda()
            # input_ids shape: (batch, seq_length).
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)
            # Note: logits pre-softmax.
            logits = outputs['logits'].detach()
            del outputs
            # Labels are the next token prediction for each index.
            labels = input_ids[:, 1:]  # Shape: (n_examples, seq_len-1).
            # Next token probabilities ignored for last token.
            logits = logits[:, :-1, :]
            logits = torch.transpose(logits, 1, 2)  # The token probabilities should be index 1.
            # Shape: (n_examples=1, seq_len-1).
            # These are negative log probabilities (natural log).
            losses = loss(logits, labels).cpu()
            option_loss = torch.sum(losses, dim=-1).item()
            option_losses[mc_i] = option_loss

        # Minimum loss for completion.
        pred_i = int(torch.argmin(option_losses).item())
        is_correct.append(pred_i == correct_i)

    acc = np.mean(is_correct)
    print(f'{belebele_lang} acc: {acc}')
    with codecs.open(OUTPATH, 'ab', encoding='utf-8') as outfile:
        outfile.write(f'{belebele_lang}\t{acc}\n')
