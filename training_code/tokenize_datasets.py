"""
Tokenizes datasets, after running train_tokenizers.py.

Assumes unshuffled text datasets exist in TEXT_DIR.
"""

import os
import codecs

TOKENIZERS_DIR = 'tokenizers/monolingual'
TEXT_DIR = '../low_resource_dataset/final_merged_dedup'
OUTPUT_DIR = 'tokenized_data'

os.makedirs(OUTPUT_DIR, exist_ok=True)
fnames = os.listdir(TOKENIZERS_DIR)
for tokenizer_name in sorted(tokenizer_names):
    outpath = os.path.join(OUTPUT_DIR, '{}.txt'.format(tokenizer_name))
    if os.path.isfile(outpath):
        print('Already found file: {}'.format(outpath))
        continue
    print('\nTokenizing for tokenizer: {}'.format(tokenizer_name))
    hf_tokenizer_path = os.path.join(TOKENIZERS_DIR, tokenizer_name)
    lang = tokenizer_name[:8]
    inpath = os.path.join(TEXT_DIR, f'{lang}.txt')
    print(f'Tokenizer path: {hf_tokenizer_path}')
    print(f'Text path: {inpath}')
    command = """python3 word-acquisition-language-models/scripts/tokenize_dataset.py \
    --tokenizer={0} \
    --input_file={1} \
    --output_file={2} \
    --max_segments=-1 --max_seq_len=512 --max_examples=2000000""".format(hf_tokenizer_path, inpath, outpath)
    result = os.popen(command).read()
    print(result)
    print('Finished for tokenizer: {}'.format(tokenizer_name))
