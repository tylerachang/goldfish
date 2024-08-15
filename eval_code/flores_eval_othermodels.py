"""
FLORES eval for larger multilingual models. Assume FLORES lines (raw text file)
are in [FLORES_DIR]/[lang].txt. Run on GPU.

Outputs a numpy array for each language containing the per-sequence
log-perplexity (shape: n_sequences,).
"""

import codecs
import json
import os
import torch
from torch import nn
from tqdm import tqdm
import numpy as np

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import PeftModel  # Only needed for Mala500.


FLORES_DIR = 'flores_texts'
OUTPUT_DIR = 'flores_model_surprisals'
MAX_SEQ_LEN = 512
# Whether to evaluate as P(second_half | first_half), where the halfway point is
# determined by characters. Probabilities during the first half may be lower for
# the multilingual baseline models, because they may still be determining the
# input language during the first half.
ONLY_SECOND_HALF = True

MODELS = {'mala500': 'MaLA-LM/mala-500-10b',
          'bloom7b1': 'bigscience/bloom-7b1',
          'xglm4b5': 'facebook/xglm-4.5B',
          'xglm7b5': 'facebook/xglm-7.5B'}

os.makedirs(OUTPUT_DIR, exist_ok=True)
langs = [fname.lower().replace('.txt', '') for fname in os.listdir(FLORES_DIR)]
cache_dir = 'hf_cache'
for model_id, model_name in MODELS.items():
    # Load model.
    print('Loading model.')
    if model_name == 'MaLA-LM/mala-500-10b':
        base_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', cache_dir=cache_dir)
        base_model.resize_token_embeddings(260164)
        tokenizer = AutoTokenizer.from_pretrained('MaLA-LM/mala-500-10b', cache_dir=cache_dir)
        model = PeftModel.from_pretrained(base_model, 'MaLA-LM/mala-500-10b', cache_dir=cache_dir).to('cuda')
    else:
        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, cache_dir=cache_dir).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # For each FLORES lang:
    for lang in langs:
        flores_inpath = os.path.join(FLORES_DIR, f'{lang}.txt')
        outpath = os.path.join(OUTPUT_DIR, f'{lang}_{model_id}_surprisals.npy')
        if os.path.isfile(outpath):
            print(f'Already ran lang: {lang}')
            continue
        # Load FLORES.
        with codecs.open(flores_inpath, 'rb', encoding='utf-8') as f_in:
            flores_lines = [l.strip() for l in f_in]
        print('Running model.')
        pad_token_id = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        unk_token_id = tokenizer.unk_token_id
        # Model-specific BOS token.
        if model_name in ['facebook/xglm-4.5B', 'facebook/xglm-7.5B']:
            # ID 2 for XGLM 4.5b and 7.5b, as is default with their tokenizer.
            prepend_token_id = tokenizer.eos_token_id
        elif model_name in ['MaLA-LM/mala-500-10b', 'bigscience/bloom-7b1']:
            prepend_token_id = tokenizer.bos_token_id
        else:
            prepend_token_id = tokenizer.cls_token_id
        # Prep for running model.
        loss = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
        sequence_surprisals = []
        # Run model.
        for flores_line in tqdm(flores_lines):
            # Prepare inputs.
            inputs = tokenizer([flores_line], add_special_tokens=False)
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
            # Logits shape: (n_examples, seq_len, vocab_size).
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
            # Set to log2 (multiply by log2(e)):
            losses = losses * np.log2(np.e)
            # Set unk to random chance (note: vocab_size does not include added tokens).
            # -log2(1/vocab_size) = log2(vocab_size)
            losses[labels==unk_token_id] = np.log2(tokenizer.vocab_size)

            # If desired, ignore loss for first half of sequence, by characters.
            # This prevents high perplexities for multilingual models that
            # are still determining the input language during the first
            # half of the sequence.
            if ONLY_SECOND_HALF:
                halfline = flores_line[:(len(flores_line)//2)]
                halfline_len_tokens = len(tokenizer([halfline], add_special_tokens=False)['input_ids'][0])
                losses[0, :halfline_len_tokens] = 0.0

            # Because loss is set to ignore pad index, just take sum.
            # This is the sum of the negative log probabilities.
            summed_loss = torch.sum(losses, dim=-1).item()
            sequence_surprisals.append(summed_loss)
        sequence_surprisals = np.array(sequence_surprisals).flatten()
        np.save(outpath, sequence_surprisals, allow_pickle=False)
        print(f'Saved to {outpath}')
    del model
