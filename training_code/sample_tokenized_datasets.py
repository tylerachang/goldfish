"""
Samples tokenized datasets based on MB size, after running tokenize_datasets.py.
Tokenized data should be shuffled and placed in SHUFFLED_TOKENIZED_DATA.
Shuffling for large files can be done with terashuf:

mkdir terashuf_tmp_output
export TMPDIR=terashuf_tmp_output
export SEED=42
terashuf/terashuf < "inpath.txt" > "inpath_shuffled.txt"

"""

import os
import codecs
from transformers import AutoTokenizer
from tqdm import tqdm

TOKENIZERS_DIR = 'tokenizers/monolingual'
SHUFFLED_TOKENIZED_DATA = 'shuffled_tokenized_data'
OUTPUT_DIR = 'tokenized_data_split'
BYTE_PREMIUMS_PATH = 'goldfish-models/byte_premiums.tsv'

# Get byte premiums.
# Dictionary mapping language to byte premium.
with codecs.open(BYTE_PREMIUMS_PATH, 'rb', encoding='utf-8') as f:
    byte_premiums = f.read()
byte_premiums = byte_premiums.strip().split('\n')[1:]  # Skip header.
byte_premiums = [line.split('\t') for line in byte_premiums]
byte_premiums = [(split_line[0], float(split_line[1])) for split_line in byte_premiums]
byte_premiums = dict(byte_premiums)

def write_file(lines, lang, suffix_str):
    outpath = os.path.join(OUTPUT_DIR, f'{lang}_{suffix_str}.txt')
    if os.path.isfile(outpath):
        print(f'Already found outpath: {outpath}')
        return
    outfile = codecs.open(outpath, 'wb', encoding='utf-8')
    for line in lines:
        outfile.write(line.strip() + '\n')
    outfile.close()
    print(f'Wrote: {outpath}')

# For each tokenizer, create dataset with the desired amount of text (scaled
# by byte premiums):
# 5mb -> 5mb of text.
# 10mb -> 10mb of text.
# full -> full text (will be under 100mb, based on how we sampled tokenizer data).
# 100mb -> 100mb of text, and full (if <1gb) or 1gb (if >=1gb).
#
# Inversely, each text dataset uses the following tokenizer:
# full -> full tokenizer if no 100mb tokenizer, otherwise 100mb tokenizer. This
#         dataset only exists if 1gb does not exist.
# 1gb -> 100mb tokenizer.
# 100mb -> 100mb tokenizer.
# 10mb -> 10mb tokenizer.
# 5mb -> 5mb tokenizer.

os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer_names = sorted(os.listdir(TOKENIZERS_DIR))
final_dataset_sizes = dict()

for tokenizer_name in sorted(tokenizer_names):
    print('\nRunning for tokenizer: {}'.format(tokenizer_name))
    lang = tokenizer_name[:8]
    byte_premium = byte_premiums[lang]
    tokenizer_quantity_str = tokenizer_name.split('_')[-1]
    hf_tokenizer_path = os.path.join(TOKENIZERS_DIR, tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_path, cache_dir='hf_cache')

    # Get target amount(s).
    if tokenizer_quantity_str == '5mb': target_mb_sizes = [5]
    if tokenizer_quantity_str == '10mb': target_mb_sizes = [10]
    if tokenizer_quantity_str == 'full': target_mb_sizes = [1000]
    if tokenizer_quantity_str == '100mb': target_mb_sizes = [100, 1000]
    target_sizes = [target_size * 1000000 for target_size in target_mb_sizes]  # In bytes.
    last_line_train = dict()  # To save the last line for the target size.
    # Read inpath.
    inpath = os.path.join(SHUFFLED_TOKENIZED_DATA, f'{tokenizer_name}.txt')
    infile = codecs.open(inpath, 'rb', encoding='utf-8')
    lines = []
    curr_data_size = 0.0  # In bytes.
    target_size_i = 0
    reached_max = False
    for line_i, line in tqdm(enumerate(infile)):
        lines.append(line)
        tokenized_sequence = [int(id) for id in line.strip().split()]
        text_sequence = tokenizer.decode(tokenized_sequence)
        text_sequence = text_sequence.replace('<unk>', '')
        text_sequence = text_sequence.replace('<s>', '')
        text_sequence = text_sequence.replace('</s>', '')
        text_sequence = text_sequence.replace('[CLS]', '')
        text_sequence = text_sequence.replace('[SEP]', '')
        text_sequence = text_sequence.replace('<pad>', '')
        text_sequence = text_sequence.replace('[MASK]', '')
        # Replace consecutive whitespace with single space.
        text_sequence = ' '.join(text_sequence.strip().split())
        line_size_bytes = len(text_sequence.encode('utf-8')) / byte_premium
        curr_data_size += line_size_bytes
        if curr_data_size >= target_sizes[target_size_i]:
            quantity_str = str(round(target_sizes[target_size_i] / 1000000)) + 'mb'
            last_line_train[quantity_str] = line_i  # Save the last line used for this training set.
            write_file(lines, lang, quantity_str)
            final_dataset_sizes[lang + '_' + quantity_str] = curr_data_size
            target_size_i += 1  # Now wait for the next target size.
            if target_size_i == len(target_sizes):
                reached_max = True
                break
    infile.close()
    # The tokenizers were trained on this quantity. However, even if there was
    # enough data prior to tokenization, the sequence truncations slightly
    # reduce the data quantity. This can reduce some languages to below their
    # target cutoff.
    if tokenizer_quantity_str == '5mb':
        if curr_data_size < 5000000:
            final_dataset_sizes[lang + '_5mb_dropped'] = curr_data_size
            continue  # This language will be dropped entirely, <5mb.
    if tokenizer_quantity_str == '10mb':
        if curr_data_size < 10000000:
            final_dataset_sizes[lang + '_10mb_dropped'] = curr_data_size
            # This language will be dropped from the 10mb category.
            # Note that there is a "full" tokenizer considered separately.
            continue
    if tokenizer_quantity_str == '100mb':
        if curr_data_size < 100000000:
            # This language will be dropped from the 100mb category.
            # However, in this case, a full tokenizer will have been skipped
            # because there were 100mb of text. We still want the "full" dataset
            # size, using a "full" tokenizer. In this case, the full tokenizer
            # is essentially the 100mb tokenizer, so we will need to rename it
            # manually. We give this dataset a funky name so we remember to
            # rename the "100mb" tokenizer to "full".
            # Hopefully, this situation never occurs.
            write_file(lines, lang, 'full_but_using_100mb_tokenizer')
            final_dataset_sizes[lang + '_full_but_using_100mb_tokenizer'] = curr_data_size
            continue
    if tokenizer_quantity_str == 'full':
        if curr_data_size < 5000000:
            final_dataset_sizes[lang + '_full_dropped'] = curr_data_size
            continue  # This language will be dropped entirely, <5mb.
    # Now, write full tokenizer data file if the maximum target data was not
    # reached. After excluding the cases above, this only happens for 100mb
    # or full tokenizers that do not reach 1000mb.
    if not reached_max:
        write_file(lines, lang, 'full')
        final_dataset_sizes[lang + '_full'] = curr_data_size
        print(f'Full size (scaled): {curr_data_size/1000000} MB')

    # Create eval set(s) just in case we want them later.
    # Saving 2K and 16K sequences (note that 16K can be subsampled to other
    # desired sizes).
    for quantity_str, last_line_train_i in last_line_train.items():
        print(f'Saving eval for {quantity_str}, starting at line {last_line_train_i}.')
        infile = codecs.open(inpath, 'rb', encoding='utf-8')
        lines = []
        for line_i, line in enumerate(infile):
            if line_i <= last_line_train_i: continue
            lines.append(line)
            if len(lines) == 2000:
                write_file(lines, lang, quantity_str + '_eval2k')
            if len(lines) == 16000:
                write_file(lines, lang, quantity_str + '_eval16k')
            if len(lines) > 16000: break
        infile.close()

print('Done writing files!')
dataset_names = sorted(final_dataset_sizes, key=final_dataset_sizes.get)
for dataset_name in dataset_names:
    # Note: accounts for byte premiums.
    n_bytes = final_dataset_sizes[dataset_name]
    print(f'{dataset_name}: {n_bytes}')
