"""
Belebele eval for goldfish models. Assume Belebele jsonl files are in
[BELEBELE_DIR]/[lang].jsonl. Assume the goldfish models are in
models/[dataset_size]/[lang]_[dataset_size].

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

from constants import LANG_SETS


BELEBELE_DIR = 'belebele_eval'
OUTPATH = 'belebele_results/goldfish.tsv'
MAX_SEQ_LEN = 512


# Missing or ambiguous FLORES languages. Map to Goldfish languages.
FLORES_LANG_MAPPING = {
        'ace_arab': 'urd_arab', 'acm_arab': 'arb_arab', 'acq_arab': 'arb_arab',
        'aeb_arab': 'arb_arab', 'ajp_arab': 'arb_arab', 'als_latn': 'sqi_latn',
        'arb_latn': 'mlt_latn', 'ars_arab': 'arb_arab', 'ary_arab': 'arb_arab',
        'awa_deva': 'hin_deva', 'ayr_latn': 'aym_latn', 'azb_arab': 'aze_arab',
        'azj_latn': 'aze_latn', 'bjn_arab': 'urd_arab', 'dik_latn': 'din_latn',
        'gaz_latn': 'orm_latn', 'kam_latn': 'kik_latn', 'kas_arab': 'urd_arab',
        'khk_cyrl': 'mon_cyrl', 'kmr_latn': 'kur_latn', 'lvs_latn': 'lav_latn',
        'min_arab': 'urd_arab', 'mni_beng': 'ben_beng', 'npi_deva': 'nep_deva',
        'nus_latn': 'din_latn', 'ory_orya': 'ori_orya', 'pbt_arab': 'pus_arab',
        'plt_latn': 'mlg_latn', 'quy_latn': 'que_latn', 'swh_latn': 'swa_latn',
        'taq_latn': 'kab_latn', 'taq_tfng': None, 'tzm_tfng': None,
        'uzn_latn': 'uzb_latn', 'ydd_hebr': 'yid_hebr', 'yue_hant': 'zho_hant',
        'zsm_latn': 'msa_latn'}
# For additional Belebele langs:
FLORES_LANG_MAPPING['ben_latn'] = 'hin_latn'
FLORES_LANG_MAPPING['npi_latn'] = 'hin_latn'
FLORES_LANG_MAPPING['sin_latn'] = 'hin_latn'
FLORES_LANG_MAPPING['urd_latn'] = 'hin_latn'


def load_model(model_dir, cache_dir='hf_cache'):
    # Load config.
    config_path = os.path.join(model_dir, 'config.json')
    config = AutoConfig.from_pretrained(config_path, cache_dir=cache_dir)
    # Load tokenizer. Assume in model directory.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
    # Load model.
    print('Loading from directory: {}'.format(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
            model_dir, config=config, cache_dir=cache_dir)
    # Load onto GPU.
    if torch.cuda.is_available():
        model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return config, tokenizer, model


langs = [fname.lower().replace('.jsonl', '') for fname in os.listdir(BELEBELE_DIR)]
# Open outfile and write header.
with codecs.open(OUTPATH, 'wb', encoding='utf-8') as outfile:
    outfile.write('lang\tacc\n')

for belebele_lang in sorted(langs):
    print('Loading dataset.')
    inpath = os.path.join(DATASET_DIR, belebele_lang + '.jsonl')
    dataset = []
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    for line in infile:
        dataset.append(json.loads(line.strip()))
    infile.close()
    assert len(dataset) == 900  # Expected number of examples.

    # Goldfish language to use.
    glang = FLORES_LANG_MAPPING[belebele_lang] if belebele_lang in FLORES_LANG_MAPPING else belebele_lang
    # Largest model.
    print('Loading model.')
    dataset_size = '1000mb' if glang in LANG_SETS['1000mb'] else 'full'
    model_dir = f'models/{dataset_size}/{glang}_{dataset_size}'
    config, tokenizer, model = load_model(model_dir)

    print('Running model.')
    loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
    pad_token_id = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    prepend_token_id = tokenizer.cls_token_id
    is_correct = []
    loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
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
            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
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
