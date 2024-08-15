"""
Trains tokenizers, after running create_tokenizer_datasets.py.
"""

import os
import codecs
import sentencepiece as spm

VOCAB_SIZE = 50000
TOKENIZER_DATA_DIR = 'tokenizer_training_datasets'
TOKENIZERS_DIR = 'tokenizers'

os.makedirs(os.path.join(TOKENIZERS_DIR, 'spm'), exist_ok=True)
os.makedirs(os.path.join(TOKENIZERS_DIR, 'monolingual'), exist_ok=True)
fnames = os.listdir(TOKENIZER_DATA_DIR)
tokenizer_names = [fname.split('.')[0] for fname in fnames]
for tokenizer_name in sorted(tokenizer_names):
    hf_tokenizer_path = os.path.join(TOKENIZERS_DIR, f'monolingual/{tokenizer_name}')
    if os.path.isdir(hf_tokenizer_path):
        print('Already found directory: {}'.format(hf_tokenizer_path))
        continue
    inpath = os.path.join(TOKENIZER_DATA_DIR, f'{tokenizer_name}.txt')
    spm_tokenizer_path = os.path.join(TOKENIZERS_DIR, f'spm/{tokenizer_name}')
    try:
        spm.SentencePieceTrainer.train(input=inpath,
                model_prefix=spm_tokenizer_path,
                vocab_size=VOCAB_SIZE,
                input_sentence_size=999999999,
                train_extremely_large_corpus=True,
                shuffle_input_sentence=True,
                num_threads=16)
    except RuntimeError as e:
        error_message = str(e)
        assert 'Vocabulary size too high' in error_message
        # Max vocab size is the last word in the message.
        new_vocab_size = int(error_message.split()[-1].replace('.', ''))
        print('Changing vocab size to {}.'.format(new_vocab_size))
        spm.SentencePieceTrainer.train(input=inpath,
                model_prefix=spm_tokenizer_path,
                vocab_size=new_vocab_size,
                input_sentence_size=999999999,
                train_extremely_large_corpus=True,
                shuffle_input_sentence=True,
                num_threads=16)
    command = """python3 word-acquisition-language-models/scripts/convert_spm_to_hf_tokenizer.py \
    --input={0}.model \
    --output_dir={1} \
    --keep_accents=True \
    --multiple_of=2048""".format(spm_tokenizer_path, hf_tokenizer_path)
    result = os.popen(command).read()
    print(result)
    print(f'Finished tokenizer: {hf_tokenizer_path}')
