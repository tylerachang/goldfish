"""
Prepares scripts to run pre-training, after sample_tokenized_datasets.py.
Outputs bash scripts to run language model training for all languages.

To run the bash scripts for training:

chmod u+x training_scripts/*.sh
training_scripts/train_5mb.sh
training_scripts/train_10mb.sh
training_scripts/train_100mb.sh
training_scripts/train_1000mb.sh
training_scripts/train_full.sh

"""

import os
import codecs
import argparse

from constants import LANG_SETS

# Defaults for goldfish.
WARMUP_PROPORTION = 0.10
EPOCHS = 10
LEARNING_RATE = 0.0001
BATCH_SIZES = {'5mb': 4, '10mb': 8,
               '100mb': 32, '1000mb': 64}
TOKENS_PER_SEQUENCE = 512  # Must also be set in model config.

OUTPUT_DIR = 'training_scripts'
MODELS_OUTDIR = 'models'
DATASETS_DIR = 'tokenized_data_split'
TOKENIZERS_DIR = 'tokenizers/monolingual'
MAX_BATCH_PER_DEVICE = 8  # How many fit on one device.
TRAIN_SIZES_TOKENS_PATH = 'goldfish-models/train_sizes_tokens.tsv'


# Get token counts for all train datasets.
with codecs.open(TRAIN_SIZES_TOKENS_PATH, 'rb', encoding='utf-8') as f:
    train_sizes = f.read()
train_sizes = train_sizes.strip().split('\n')[1:]  # Skip header.
train_sizes = [line.split('\t') for line in train_sizes]
train_sizes = [(split_line[0], int(split_line[1])) for split_line in train_sizes]
train_sizes = dict(train_sizes)


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_OUTDIR, exist_ok=True)
for dataset_size in ['5mb', '10mb', '100mb', '1000mb', 'full']:
    model_dir = os.path.join(MODELS_OUTDIR, dataset_size)
    os.makedirs(model_dir, exist_ok=True)

    # Determine which languages to run for this dataset size.
    if dataset_size == 'full':
        # Run full data model for all languages that did not reach 1GB limit.
        langs = LANG_SETS['5mb'].difference(LANG_SETS['1000mb'])
    else:
        # Standard dataset sizes.
        langs = LANG_SETS[dataset_size]

    # Select model size.
    if dataset_size in ['5mb', '10mb']:
        model_size = 'small'
    else:
        model_size = 'base'

    # Write script.
    script_outname = f'train_{dataset_size}.sh'
    script_outpath = os.path.join(OUTPUT_DIR, script_outname)
    script = ''
    script += '\nexport CUDA_VISIBLE_DEVICES=0\n'
    for lang in langs:
        # Determine batch size.
        if dataset_size == 'full':
            if lang in LANG_SETS['5mb'].difference(LANG_SETS['10mb']):
                # Dataset size rounds down to 5mb.
                batch_size = BATCH_SIZES['5mb']
                tokenizer_path = os.path.join(TOKENIZERS_DIR, f'{lang}_full')
            elif lang in LANG_SETS['10mb'].difference(LANG_SETS['100mb']):
                # Dataset size rounds down to 10mb.
                batch_size = BATCH_SIZES['10mb']
                tokenizer_path = os.path.join(TOKENIZERS_DIR, f'{lang}_full')
            elif lang in LANG_SETS['100mb'].difference(LANG_SETS['1000mb']):
                # Dataset size rounds down to 100mb.
                batch_size = BATCH_SIZES['100mb']
                tokenizer_path = os.path.join(TOKENIZERS_DIR, f'{lang}_100mb')
                # In this case, no full tokenizer was trained, because 100mb is
                # the maximum dataset size for tokenizer training.
                assert not os.path.isdir(os.path.join(TOKENIZERS_DIR, f'{lang}_full'))
        else:
            batch_size = BATCH_SIZES[dataset_size]
            # Use 100mb tokenizer for 1000mb models.
            if dataset_size == '1000mb':
                tokenizer_path = os.path.join(TOKENIZERS_DIR, f'{lang}_100mb')
            else:
                tokenizer_path = os.path.join(TOKENIZERS_DIR, f'{lang}_{dataset_size}')
        # Check that tokenizer path exists.
        assert os.path.isdir(tokenizer_path)
        # Determine epoch steps.
        n_train_tokens = train_sizes[lang + '_' + dataset_size]
        n_examples_per_epoch = int(n_train_tokens / TOKENS_PER_SEQUENCE)
        epoch_steps = n_examples_per_epoch / batch_size
        # Determine whether eval set exists.
        train_path = os.path.join(DATASETS_DIR, lang + '_' + dataset_size + '.txt')
        assert os.path.isfile(train_path)
        eval_path = os.path.join(DATASETS_DIR, lang + '_' + dataset_size + '_eval2k.txt')
        if os.path.isfile(eval_path):
            evaluation_strategy = 'steps'
            eval_steps = int(epoch_steps / 2)  # Eval twice per epoch.
        else:
            evaluation_strategy = 'no'
            eval_steps = 999999999

        # Other hyperparameters and settings.
        max_steps = int(EPOCHS * epoch_steps)
        warmup_steps = int(max_steps * WARMUP_PROPORTION)
        model_outname = lang + '_' + dataset_size
        model_outpath = os.path.join(model_dir, model_outname)

        # Save for 1000mb models.
        if dataset_size == '1000mb':
            save_strategy = 'steps'
            save_steps = int(epoch_steps / 2)  # Save twice per epoch.
        else:
            save_strategy = 'no'
            save_steps = 999999999

        # Call training script.
        script += f'\n# {lang}\n'
        # Check if model already exists.
        model_filepath = os.path.join(model_outpath, 'pytorch_model.bin')
        script += f"""
        if test -f {model_filepath}; then
        echo "Model already found: {model_outname}."
        fi
        """
        # Pre-training script.
        script += """
        python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \\
        """
        batch_per_device = min(batch_size, MAX_BATCH_PER_DEVICE)
        gradient_accumulation_steps = batch_size // batch_per_device
        assert batch_size % batch_per_device == 0
        script += f"""
        --tokenizer_name={tokenizer_path} \\
        --config_name="gpt_{model_size}_config.json" \\
        --do_train --train_iterable --eval_iterable \\
        --eval_data_file={eval_path} \\
        --per_device_train_batch_size={batch_per_device} \\
        --gradient_accumulation_steps={gradient_accumulation_steps} \\
        --per_device_eval_batch_size=8 \\
        --evaluation_strategy={evaluation_strategy} --save_strategy={save_strategy} \\
        --eval_steps={eval_steps} --save_steps={save_steps} \\
        --max_steps={max_steps} \\
        --warmup_steps={warmup_steps} \\
        --learning_rate={LEARNING_RATE} --adam_epsilon=1e-6 --weight_decay=0.01 \\
        --train_data_file={train_path} \\
        --seed=43 \\
        --override_n_examples={n_examples_per_epoch} \\
        --output_dir={model_outpath}
        """

        # Copy tokenizer files to output directory, for convenience.
        # Four files:
        # added_tokens.json
        # special_tokens_map.json
        # spiece.model
        # tokenizer_config.json
        script += f"""
        cp {tokenizer_path}/* {model_outpath}
        """

    # Write script to output.
    outfile = codecs.open(script_outpath, 'wb', encoding='utf-8')
    for line in script.split('\n'):
        stripped_line = line.strip()
        if stripped_line == '': continue
        # Add extra line before a comment.
        if stripped_line.startswith('# '): outfile.write('\n')
        outfile.write(stripped_line + '\n')
    outfile.close()
    print('Wrote script: {}'.format(script_outpath))

print('Done.')
