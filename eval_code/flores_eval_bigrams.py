"""
FLORES eval for bigram models trained on different amounts of text data.
Assume:
* FLORES lines (raw text file) in: [FLORES_DIR]/[lang].txt
* Tokenizers in: [TOKENIZERS_DIR]/[lang]_[tokenizer_dataset_size]
* Tokenized training data in: [TOKENIZED_TRAIN_DIR]/[lang]_[dataset_size].txt
* Tokenization code downloaded:
https://github.com/tylerachang/word-acquisition-language-models

Outputs a numpy array for each language containing the per-sequence
log-perplexity (shape: n_sequences,).
"""

import codecs
import math
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter
import gc

from constants import LANG_SETS
from transformers import AutoTokenizer  # Only required if using ONLY_SECOND_HALF.


TOKENIZERS_DIR = 'tokenizers/monolingual'
TOKENIZED_TRAIN_DIR = 'tokenized_data_split'
FLORES_DIR = 'flores_texts'
TMP_DIR = 'tmp_flores_bigrams'
OUTPUT_DIR = 'flores_ngram_surprisals'
MAX_VOCAB_SIZE = 51200  # Constant for all Goldfish.
UNK_ID = 0  # Constant for all Goldfish.
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


# Returns a list of lists of integers (token ids), given an input text file
# of tokenized examples (space-separated token ids).
def get_file_examples(filepath, max_examples=-1):
    if max_examples == -1:
        max_examples = math.inf
    # Load examples.
    total_tokens = 0
    examples = []
    examples_file = codecs.open(filepath, 'rb', encoding='utf-8')
    for line in examples_file:
        if len(examples) >= max_examples:
            break
        stripped_line = line.strip()
        if stripped_line == '':
            continue
        example = [int(token_id) for token_id in stripped_line.split()]
        total_tokens += len(example)
        examples.append(example)
    examples_file.close()
    return examples


# For each sequence in eval_sequences_path, returns the n-gram surprisal for
# each token conditioned on the previous tokens. For tokens with n-gram
# probability zero, the surprisal is np.nan. Use get_ngram_surprisals_with_backoff()
# to ensure no np.nans. For padding tokens, surprisal is -1.0.
#
# Output shape: (n_sequences, max_seq_length).
# Note: cache_prefix allows for caching n-gram probabilities for later use.
#
# Note: when counting, this prunes counts less than prune_minimum every
# prune_every sequences (and once at the end).
# The final counts are cached.
#
# Output cache files:
# [cache_prefix]_[ngram_n]gram_surprisals.npy
# [cache_prefix]_[ngram_n]gram_counts.pickle
def get_ngram_surprisals(ngram_n, eval_sequences_path, train_sequences_path,
                         vocab_size, cache_prefix, prune_every=1000000,
                         prune_minimum=1, unk_id=None):
    # Load from cache if possible.
    outpath = cache_prefix + f'_{ngram_n}gram_surprisals.npy'
    if os.path.isfile(outpath):
        print(f'Using cached {ngram_n}-gram surprisals.')
        return np.load(outpath, allow_pickle=False)
    # Array of n-gram counts.
    # Entry i_0, ..., i_{n-1} is the count of
    # i_0, ..., i_{n-2}, i_{n-1}.
    # Dictionary mapping context tuples to Counters:
    # ngrams[(i-n+1, ..., i-1)][i] = count
    # Note: for unigrams, the first key is an empty tuple.
    ngrams_path = cache_prefix + f'_{ngram_n}gram_counts.pickle'
    # Get ngram counts.
    if os.path.isfile(ngrams_path):
        print(f'Loading {ngram_n}-gram counts.')
        with open(ngrams_path, 'rb') as handle:
            ngrams = pickle.load(handle)
    else:
        print(f'Computing {ngram_n}-gram counts.')
        # Function to prune the ngrams dictionary.
        def prune_ngrams(ngrams, min_count=2):
            if min_count is None:
                # No pruning.
                return ngrams
            context_keys_to_remove = []
            for context, counts in ngrams.items():
                target_keys_to_remove = []
                for target, count in counts.items():
                    if count < min_count:
                        target_keys_to_remove.append(target)
                for target in target_keys_to_remove:
                    counts.pop(target)
                del target_keys_to_remove
                # If all zero, prune this entire counter.
                if len(counts) == 0:
                    context_keys_to_remove.append(context)
            for context in context_keys_to_remove:
                ngrams.pop(context)
            # To resize the dictionary in memory after the removed keys.
            ngrams = ngrams.copy()
            del context_keys_to_remove
            gc.collect()
            return ngrams
        # Count ngrams. Create dictionary mapping:
        # ngrams[(i-n+1, ..., i-1)][i] = count
        # Note: for unigrams, the first key is an empty tuple.
        ngrams = defaultdict(lambda: Counter())
        train_file = codecs.open(train_sequences_path, 'rb', encoding='utf-8')
        line_count = 0
        for line_i, line in tqdm(enumerate(train_file)):
            stripped_line = line.strip()
            if stripped_line == '': continue
            sequence = [int(token_id) for token_id in stripped_line.split()]
            # Initialize with the extra pre-sequence tokens.
            # This represents the token_ids for the current ngram_n positions.
            curr = np.ones(ngram_n, dtype=int) * vocab_size
            for token_id in sequence:
                # Increment to the next token.
                curr = np.roll(curr, -1)
                curr[-1] = token_id
                # Increment the corresponding ngram:
                ngrams[tuple(curr[:-1])][curr[-1]] += 1
            # Pruning.
            line_count += 1
            if line_count % prune_every == 0:
                print(f'Pruning ngram counts <{prune_minimum}.')
                orig_len = len(ngrams)
                ngrams = prune_ngrams(ngrams, min_count=prune_minimum)
                print(f'Pruned: {orig_len} keys -> {len(ngrams)} keys.')
        print(f'Final prune: pruning ngram counts <{prune_minimum}.')
        orig_len = len(ngrams)
        ngrams = prune_ngrams(ngrams, min_count=prune_minimum)
        print(f'Pruned: {orig_len} keys -> {len(ngrams)} keys.')
        # To allow pickling.
        ngrams.default_factory = None
        with open(ngrams_path, 'wb') as handle:
            pickle.dump(ngrams, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ngrams.default_factory = lambda: Counter()
    # Convert counts to conditional probabilities.
    # Entry i_0, ..., i_{n-1} is the probability of
    # i_{n-1} given i_0, ..., i_{n-2}.
    print('Converting counts to probabilities.')
    for context_key in ngrams:
        # Convert the counts to probabilities.
        counts = ngrams[context_key]
        total = np.sum(list(counts.values()))
        probs_dict = defaultdict(lambda: 0.0)
        for target_key, count in counts.items():
            prob = count / total
            probs_dict[target_key] = prob
        ngrams[context_key] = probs_dict

    # Get scores for all tokens.
    # This is because we want the surprisal for every token in every sequence,
    # to aggregate depending on each example and window size.
    # Surprisal is np.nan for n-grams with probability zero.
    print(f'Computing {ngram_n}-gram surprisals.')
    sequences = get_file_examples(eval_sequences_path)
    max_seq_len = np.max([len(sequence) for sequence in sequences])
    print(f'Max sequence length: {max_seq_len} tokens')
    surprisals = -1.0 * np.ones((len(sequences), max_seq_len))
    for sequence_i, sequence in tqdm(enumerate(sequences)):
        # Fill previous tokens with placeholder.
        curr = np.ones(ngram_n, dtype=int) * vocab_size
        for token_i, token_id in enumerate(sequence):
            # Increment to the next token.
            curr = np.roll(curr, -1)
            curr[-1] = token_id
            # Get the corresponding ngram:
            conditional_prob = ngrams[tuple(curr[:-1])][curr[-1]]
            if np.isclose(conditional_prob, 0.0):
                surprisal = np.nan
            elif (unk_id is not None) and (token_id == unk_id):
                # Random chance: -log2(1/max_vocab_size) = log2(max_vocab_size)
                surprisal = np.log2(vocab_size)
            else:
                surprisal = -1.0 * np.log2(conditional_prob)
            surprisals[sequence_i, token_i] = surprisal
    np.save(outpath, surprisals, allow_pickle=False)
    return surprisals

# Returns n-gram suprisals as in get_ngram_surprisals(), but using backoff
# to ensure no np.nans. For backoff, zero probabilities for ngram n are
# replaced with the probabilities for ngram n-1, multiplied by a backoff factor.
# Backoff factor is set to 0.4, as in previous work (introduced in
# https://aclanthology.org/D07-1090.pdf, "stupid backoff").
# Unigram surprisals are smoothed to the minimum nonzero probability.
#
# Assumes that get_ngram_surprisals() has already been run for all
# 1 <= n <= ngram_n.
def get_ngram_surprisals_with_backoff(cache_prefix, ngram_n, backoff_factor=0.4):
    to_return = None
    has_nan = True
    curr_ngram_n = ngram_n
    # The backoff equation is:
    # (p * backoff_factor^{n_times_backed_off})
    # The equivalent of multiplying for surprisal is:
    # -log2(p*backoff_factor^{n_times})
    # = -log2(p) - log2(backoff_factor^{n_times})
    # = -log2(p) - n_times * log2(backoff_factor)
    # So we add -log2(backoff_factor) for each time backed off.
    backoff_penalty = 0.0
    while curr_ngram_n > 0 and has_nan:
        # Load from cache.
        inpath = cache_prefix + f'_{curr_ngram_n}gram_surprisals.npy'
        if os.path.isfile(inpath):
            curr_ngrams = np.load(inpath, allow_pickle=False)
        else:
            print(f'Cannot find cached {curr_ngram_n}-gram surprisals; run get_ngram_surprisals() first.')
        # If unigrams, fill in with maximum surprisal (minimum nonzero probability).
        if curr_ngram_n == 1:
            max_surprisal = np.nanmax(curr_ngrams)
            curr_ngrams[np.isnan(curr_ngrams)] = max_surprisal
        # Fill in np.nans (backoff).
        if to_return is None:
            to_return = curr_ngrams
        else:
            nan_mask = np.isnan(to_return)
            to_return[nan_mask] = curr_ngrams[nan_mask] + backoff_penalty
        if not np.any(np.isnan(to_return)):
            has_nan = False
        # Decrement.
        curr_ngram_n -= 1
        backoff_penalty += -1.0 * np.log2(backoff_factor)
    return to_return

# Tokenize a dataset.
def tokenize_dataset(inpath, outpath, tokenizer_path,
                     max_segments=-1, max_seq_len=512, max_examples=999999999):
    if os.path.isfile(outpath):
        print(f'Already found file: {outpath}')
        return
    print(f'\nTokenizing for tokenizer: {tokenizer_path}')
    command = f"""python3 word-acquisition-language-models/scripts/tokenize_dataset.py \
    --tokenizer={tokenizer_path} \
    --input_file={inpath} \
    --output_file={outpath} \
    --max_segments={max_segments} --max_seq_len={max_seq_len} --max_examples={max_examples}"""
    result = os.popen(command).read()
    print(result)
    print('Finished tokenization.')


"""
Main script to run for FLORES dataset.
"""

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
for dataset_size in ['5mb', '10mb', '100mb', 'full', '1000mb']:
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
            flores_inpath = os.path.join(FLORES_DIR, f'{flores_lang}.txt')
            if not os.path.isfile(flores_inpath):
                print(f'Non-FLORES language: {flores_lang}')
                continue

            # Tokenize FLORES dataset with this tokenizer.
            # 1000mb uses the 100mb tokenizer, because tokenizer datasets are
            # capped.
            tok_dataset_size = '100mb' if ((goldfish_lang in LANG_SETS['100mb']) and (dataset_size in ['100mb', '1000mb', 'full'])) else dataset_size
            tokenizer_path = os.path.join(TOKENIZERS_DIR, f'{goldfish_lang}_{tok_dataset_size}')
            train_sequences_path = os.path.join(TOKENIZED_TRAIN_DIR, f'{goldfish_lang}_{dataset_size}.txt')
            assert os.path.isfile(train_sequences_path)
            # Check if all already exist.
            outpaths = []
            for ngram_n in [1, 2]:
                outpath = os.path.join(OUTPUT_DIR, f'{flores_lang}_{dataset_size}_{ngram_n}gram_surprisals.npy')
                outpaths.append(outpath)
            if all([os.path.isfile(outpath) for outpath in outpaths]):
                print(f'Already completed language: {flores_lang}')
                continue

            # Run.
            print(f'Running bigrams {goldfish_lang}_{dataset_size} for FLORES lang {flores_lang}')
            tokenized_flores_path = os.path.join(TMP_DIR, f'{flores_lang}_{dataset_size}_flores_tokenized.txt')
            # One FLORES text per line.
            tokenize_dataset(flores_inpath, tokenized_flores_path, tokenizer_path,
                             max_segments=1, max_seq_len=MAX_SEQ_LEN)
            # Compute n-grams.
            eval_sequences_path = tokenized_flores_path
            cache_prefix = os.path.join(TMP_DIR, f'{flores_lang}_{dataset_size}_flores')
            # No pruning for unigrams and bigrams.
            for ngram_n in [1, 2]:
                ngram_scores = get_ngram_surprisals(ngram_n, eval_sequences_path, train_sequences_path,
                        MAX_VOCAB_SIZE, cache_prefix, prune_every=999999999,
                        prune_minimum=None, unk_id=UNK_ID)
                print(f'Computed {ngram_n}-gram surprisals.')
                del ngram_scores

            # For later use.
            if ONLY_SECOND_HALF:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                with codecs.open(flores_inpath, 'rb', encoding='utf-8') as f_in:
                    flores_lines = [l.strip() for l in f_in.readlines()]

            # Get surprisals with backoff.
            for ngram_n in [1, 2]:
                print(f'Computing {ngram_n}-gram surprisals with backoff.')
                outpath = os.path.join(OUTPUT_DIR, f'{flores_lang}_{dataset_size}_{ngram_n}gram_surprisals.npy')
                # Shape: (n_sequences, max_seq_length). Should be no nans.
                # -1.0 indicates padding token.
                ngram_scores = get_ngram_surprisals_with_backoff(cache_prefix, ngram_n, backoff_factor=0.4)
                print(f'ngram token surprisals shape: {ngram_scores.shape}')
                sequence_surprisals = []
                for seq_i in range(ngram_scores.shape[0]):
                    # Exclude first token surprisal (beginning of sequence token).
                    token_surprisals = ngram_scores[seq_i, 1:]

                    # If desired, ignore loss for first half of sequence, by characters.
                    # This prevents high perplexities for multilingual models that
                    # are still determining the input language during the first
                    # half of the sequence.
                    if ONLY_SECOND_HALF:
                        assert len(flores_lines) == ngram_scores.shape[0]
                        flores_line = flores_lines[seq_i]
                        halfline = flores_line[:(len(flores_line)//2)]
                        halfline_len_tokens = len(tokenizer([halfline], add_special_tokens=False)['input_ids'][0])
                        # Check: half line length should be less than the length of the original line (in tokens).
                        assert halfline_len_tokens < len([s for s in token_surprisals if s > 0.0])
                        token_surprisals[:halfline_len_tokens] = 0.0

                    token_surprisals = [s for s in token_surprisals if s > 0.0]
                    # Exclude last token surprisal (end of sequence token).
                    token_surprisals.pop()
                    sequence_surprisal = np.sum(token_surprisals)
                    sequence_surprisals.append(sequence_surprisal)
                sequence_surprisals = np.array(sequence_surprisals)
                np.save(outpath, sequence_surprisals, allow_pickle=False)
                print(f'Saved {ngram_n}-gram surprisals.')
            # Delete cache.
            print('Deleting cached files.')
            for filename in os.listdir(TMP_DIR):
                if filename.startswith(f'{flores_lang}_{dataset_size}'):
                    path = os.path.join(TMP_DIR, filename)
                    os.remove(path)
            print('Deleted cached files.')
