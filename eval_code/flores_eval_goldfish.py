"""
FLORES eval for goldfish models. Assume FLORES lines (raw text file) are in
[FLORES_DIR]/[lang].txt. Assume the goldfish models are in
models/[dataset_size]/[lang]_[dataset_size].

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

from constants import LANG_SETS


FLORES_DIR = 'flores_texts'
OUTPUT_DIR = 'flores_model_surprisals'
MAX_SEQ_LEN = 512
# Whether to evaluate as P(second_half | first_half), where the halfway point is
# determined by characters. Probabilities during the first half may be lower for
# the multilingual baseline models, because they may still be determining the
# input language during the first half.
ONLY_SECOND_HALF = True


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


os.makedirs(OUTPUT_DIR, exist_ok=True)
for dataset_size in ['1000mb', 'full', '100mb', '10mb', '5mb']:
    if dataset_size == 'full':
        langs = sorted(LANG_SETS['5mb'].difference(LANG_SETS['1000mb']))
    else:
        langs = sorted(LANG_SETS[dataset_size])
    # For each goldfish lang:
    for goldfish_lang in langs:
        # Get corresponding flores langs. Languages which are mapped to this
        # language, and the language itself if available.
        flores_langs = [flang for flang, glang in FLORES_LANG_MAPPING.items() if glang == goldfish_lang]
        # If the source of a mapping (i.e. should be mapped to another language),
        # then we shouldn't run it through FLORES as its own language! Other
        # languages should be run through FLORES as their own language.
        if goldfish_lang not in FLORES_LANG_MAPPING.keys(): flores_langs.append(goldfish_lang)

        for flores_lang in sorted(set(flores_langs)):
            print(f'Running goldfish {goldfish_lang}_{dataset_size} for FLORES lang {flores_lang}')
            flores_inpath = os.path.join(FLORES_DIR, f'{flores_lang}.txt')
            if not os.path.isfile(flores_inpath):
                print(f'Non-FLORES language: {flores_lang}')
                continue
            outpath = os.path.join(OUTPUT_DIR, f'{flores_lang}_goldfish_{dataset_size}_surprisals.npy')
            if os.path.isfile(outpath):
                print(f'Already ran FLORES lang: {flores_lang}')
                continue
            # Load model.
            print('Loading model.')
            model_dir = f'models/{dataset_size}/{goldfish_lang}_{dataset_size}'
            config, tokenizer, model = load_model(model_dir)
            # Load FLORES.
            with codecs.open(flores_inpath, 'rb', encoding='utf-8') as f_in:
                flores_lines = [l.strip() for l in f_in]
            print('Running model.')
            loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
            unk_token_id = tokenizer.unk_token_id
            pad_token_id = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
            prepend_token_id = tokenizer.cls_token_id
            sequence_surprisals = []
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
                input_ids = torch.tensor(input_ids)
                attention_mask = torch.tensor(attention_mask)
                if torch.cuda.is_available():
                    input_ids = input_ids.cuda()
                    attention_mask = attention_mask.cuda()
                # input_ids shape: (batch, seq_length).
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                output_hidden_states=False, return_dict=True)
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
