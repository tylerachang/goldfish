"""
Create datasets for tokenizer training.

Samples lines in order, so assumes that a shuffled version of the original text
datasets exists in SHUFFLED_DATASET_DIR. Assumes byte premiums have been
computed and saved in BYTE_PREMIUMS_PATH.

For each language, creates text files that sample target_mb_sizes.
Creates 5mb, 10mb (if available), and either 100mb or full for each language.
Note that the tokenizer data sizes are slightly different from the model
training dataset size categories, because the tokenizers are cut off at 100mb
instead of 1gb (to avoid memory errors).
"""

import codecs
import numpy as np
import os

SHUFFLED_DATASET_DIR = '../shuffled_low_resource_dataset'
OUTPUT_DIR = 'tokenizer_training_datasets'
BYTE_PREMIUMS_PATH = 'goldfish-models/training_code/byte_premiums.tsv'

# In megabytes.
target_mb_sizes = [5, 10, 100]
target_mb_sizes = sorted(target_mb_sizes)

# Get byte premiums.
# Dictionary mapping language to byte premium.
with codecs.open(BYTE_PREMIUMS_PATH, 'rb', encoding='utf-8') as f:
    byte_premiums = f.read()
byte_premiums = byte_premiums.strip().split('\n')[1:]  # Skip header.
byte_premiums = [line.split('\t') for line in byte_premiums]
byte_premiums = [(split_line[0], float(split_line[1])) for split_line in byte_premiums]
byte_premiums = dict(byte_premiums)

def write_file(lines, lang, quantity_str):
    outpath = os.path.join(OUTPUT_DIR, f'{lang}_{quantity_str}.txt')
    outfile = codecs.open(outpath, 'wb', encoding='utf-8')
    for line in lines:
        outfile.write(line.strip() + '\n')
    outfile.close()
    print(f'Wrote: {outpath}')

# Read and write datasets.
for filename in os.listdir(SHUFFLED_DATASET_DIR):
    lang = filename[:-4]
    filepath = os.path.join(SHUFFLED_DATASET_DIR, filename)
    # Skip small datasets.
    if lang not in byte_premiums:
        with codecs.open(filepath, 'rb', encoding='utf-8') as f:
            data = f.read()
        mb_size = len(data.encode('utf-8')) / 1000000
        print(f'No byte premium found for lang {lang} with raw dataset size: {mb_size} MB')
        continue
    byte_premium = byte_premiums[lang]

    infile = codecs.open(filepath, 'rb', encoding='utf-8')
    lines = []
    curr_data_size = 0.0  # In bytes.
    target_size_i = 0
    reached_max = False
    for line in infile:
        lines.append(line)
        line_size_bytes = len(line.encode('utf-8')) / byte_premium
        curr_data_size += line_size_bytes
        if curr_data_size >= target_mb_sizes[target_size_i] * 1000000:
            quantity_str = str(target_mb_sizes[target_size_i]) + 'mb'
            write_file(lines, lang, quantity_str)
            target_size_i += 1  # Now wait for the next target size.
            if target_size_i == len(target_mb_sizes):
                reached_max = True
                break
    infile.close()
    min_size_bytes = target_mb_sizes[0] * 1000000
    if curr_data_size < min_size_bytes: continue  # Skip if not enough.
    # Only write full tokenizer data file if the 100 MB file (maximum tokenizer
    # training dataset size) does not already exist.
    if not reached_max:
        write_file(lines, lang, 'full')
        print(f'Full size (scaled): {curr_data_size/1000000} MB')

print('Done!')
