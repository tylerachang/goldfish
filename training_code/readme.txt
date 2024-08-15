These scripts were used to train the Goldfish models after dataset merging.
Dataset merging was done using the datasets:
* MADLAD-400 (https://huggingface.co/datasets/allenai/MADLAD-400)
* Glot500 (https://huggingface.co/datasets/cis-lmu/Glot500)
* Chang et al. (2023) (https://github.com/tylerachang/curse-of-multilinguality)
We merge per language as described in the Goldfish paper, with several manual language code adjustments (described in the paper). We deduplicate repeated sequences of 100 UTF-8 bytes using the code in https://github.com/google-research/deduplicate-text-datasets.

We assume that pre-training and tokenization code has already been downloaded:
git clone https://github.com/tylerachang/word-acquisition-language-models.git
And shuffling code:
git clone https://github.com/alexandres/terashuf.git
(cd terashuf && make)

Note: byte_premiums.tsv has dataset sizes before tokenization; tokenization truncates some text lines, so the final dataset sizes are slightly smaller than this. For final dataset sizes (after tokenization, truncation, then detokenization, including byte premium scaling), see goldfish_data_info.tsv.

For pre-training, we run:
create_tokenizer_datasets.py
train_tokenizers.py
tokenize_datasets.py
sample_tokenized_datasets.py
prepare_training_scripts.py

We then run the training scripts generated:

chmod u+x training_scripts/*.sh
training_scripts/train_5mb.sh
training_scripts/train_10mb.sh
training_scripts/train_100mb.sh
training_scripts/train_1000mb.sh
training_scripts/train_full.sh
