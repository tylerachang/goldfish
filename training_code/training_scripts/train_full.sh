export CUDA_VISIBLE_DEVICES=0

# abk_cyrl
if test -f models/full/abk_cyrl_full/pytorch_model.bin; then
echo "Model already found: abk_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/abk_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/abk_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=15645 \
--warmup_steps=1564 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/abk_cyrl_full.txt \
--seed=43 \
--override_n_examples=12516 \
--output_dir=models/full/abk_cyrl_full
cp tokenizers/monolingual/abk_cyrl_full/* models/full/abk_cyrl_full

# ace_latn
if test -f models/full/ace_latn_full/pytorch_model.bin; then
echo "Model already found: ace_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ace_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ace_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=52897 \
--warmup_steps=5289 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ace_latn_full.txt \
--seed=43 \
--override_n_examples=42318 \
--output_dir=models/full/ace_latn_full
cp tokenizers/monolingual/ace_latn_full/* models/full/ace_latn_full

# ady_cyrl
if test -f models/full/ady_cyrl_full/pytorch_model.bin; then
echo "Model already found: ady_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ady_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ady_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10068 \
--warmup_steps=1006 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ady_cyrl_full.txt \
--seed=43 \
--override_n_examples=8055 \
--output_dir=models/full/ady_cyrl_full
cp tokenizers/monolingual/ady_cyrl_full/* models/full/ady_cyrl_full

# afb_arab
if test -f models/full/afb_arab_full/pytorch_model.bin; then
echo "Model already found: afb_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/afb_arab_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/afb_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7928 \
--warmup_steps=792 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/afb_arab_full.txt \
--seed=43 \
--override_n_examples=6343 \
--output_dir=models/full/afb_arab_full
cp tokenizers/monolingual/afb_arab_full/* models/full/afb_arab_full

# aka_latn
if test -f models/full/aka_latn_full/pytorch_model.bin; then
echo "Model already found: aka_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aka_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aka_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=55056 \
--warmup_steps=5505 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aka_latn_full.txt \
--seed=43 \
--override_n_examples=44045 \
--output_dir=models/full/aka_latn_full
cp tokenizers/monolingual/aka_latn_full/* models/full/aka_latn_full

# als_latn
if test -f models/full/als_latn_full/pytorch_model.bin; then
echo "Model already found: als_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/als_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/als_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=53472 \
--warmup_steps=5347 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/als_latn_full.txt \
--seed=43 \
--override_n_examples=171112 \
--output_dir=models/full/als_latn_full
cp tokenizers/monolingual/als_latn_100mb/* models/full/als_latn_full

# alt_cyrl
if test -f models/full/alt_cyrl_full/pytorch_model.bin; then
echo "Model already found: alt_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/alt_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/alt_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6577 \
--warmup_steps=657 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/alt_cyrl_full.txt \
--seed=43 \
--override_n_examples=5262 \
--output_dir=models/full/alt_cyrl_full
cp tokenizers/monolingual/alt_cyrl_full/* models/full/alt_cyrl_full

# ang_latn
if test -f models/full/ang_latn_full/pytorch_model.bin; then
echo "Model already found: ang_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ang_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ang_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8160 \
--warmup_steps=816 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ang_latn_full.txt \
--seed=43 \
--override_n_examples=3264 \
--output_dir=models/full/ang_latn_full
cp tokenizers/monolingual/ang_latn_full/* models/full/ang_latn_full

# apc_arab
if test -f models/full/apc_arab_full/pytorch_model.bin; then
echo "Model already found: apc_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/apc_arab_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/apc_arab_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8237 \
--warmup_steps=823 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/apc_arab_full.txt \
--seed=43 \
--override_n_examples=3295 \
--output_dir=models/full/apc_arab_full
cp tokenizers/monolingual/apc_arab_full/* models/full/apc_arab_full

# arg_latn
if test -f models/full/arg_latn_full/pytorch_model.bin; then
echo "Model already found: arg_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arg_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arg_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=30443 \
--warmup_steps=3044 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arg_latn_full.txt \
--seed=43 \
--override_n_examples=24355 \
--output_dir=models/full/arg_latn_full
cp tokenizers/monolingual/arg_latn_full/* models/full/arg_latn_full

# arz_arab
if test -f models/full/arz_arab_full/pytorch_model.bin; then
echo "Model already found: arz_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arz_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arz_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=18507 \
--warmup_steps=1850 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arz_arab_full.txt \
--seed=43 \
--override_n_examples=59223 \
--output_dir=models/full/arz_arab_full
cp tokenizers/monolingual/arz_arab_100mb/* models/full/arz_arab_full

# asm_beng
if test -f models/full/asm_beng_full/pytorch_model.bin; then
echo "Model already found: asm_beng_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/asm_beng_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/asm_beng_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=47129 \
--warmup_steps=4712 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/asm_beng_full.txt \
--seed=43 \
--override_n_examples=150813 \
--output_dir=models/full/asm_beng_full
cp tokenizers/monolingual/asm_beng_100mb/* models/full/asm_beng_full

# ast_latn
if test -f models/full/ast_latn_full/pytorch_model.bin; then
echo "Model already found: ast_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ast_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ast_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=56966 \
--warmup_steps=5696 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ast_latn_full.txt \
--seed=43 \
--override_n_examples=182292 \
--output_dir=models/full/ast_latn_full
cp tokenizers/monolingual/ast_latn_100mb/* models/full/ast_latn_full

# ava_cyrl
if test -f models/full/ava_cyrl_full/pytorch_model.bin; then
echo "Model already found: ava_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ava_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ava_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=19555 \
--warmup_steps=1955 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ava_cyrl_full.txt \
--seed=43 \
--override_n_examples=15644 \
--output_dir=models/full/ava_cyrl_full
cp tokenizers/monolingual/ava_cyrl_full/* models/full/ava_cyrl_full

# aym_latn
if test -f models/full/aym_latn_full/pytorch_model.bin; then
echo "Model already found: aym_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aym_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aym_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=22463 \
--warmup_steps=2246 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aym_latn_full.txt \
--seed=43 \
--override_n_examples=17971 \
--output_dir=models/full/aym_latn_full
cp tokenizers/monolingual/aym_latn_full/* models/full/aym_latn_full

# ayr_latn
if test -f models/full/ayr_latn_full/pytorch_model.bin; then
echo "Model already found: ayr_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ayr_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ayr_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=18655 \
--warmup_steps=1865 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ayr_latn_full.txt \
--seed=43 \
--override_n_examples=14924 \
--output_dir=models/full/ayr_latn_full
cp tokenizers/monolingual/ayr_latn_full/* models/full/ayr_latn_full

# azb_arab
if test -f models/full/azb_arab_full/pytorch_model.bin; then
echo "Model already found: azb_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/azb_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/azb_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=16432 \
--warmup_steps=1643 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/azb_arab_full.txt \
--seed=43 \
--override_n_examples=52583 \
--output_dir=models/full/azb_arab_full
cp tokenizers/monolingual/azb_arab_100mb/* models/full/azb_arab_full

# aze_arab
if test -f models/full/aze_arab_full/pytorch_model.bin; then
echo "Model already found: aze_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=34501 \
--warmup_steps=3450 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_arab_full.txt \
--seed=43 \
--override_n_examples=110404 \
--output_dir=models/full/aze_arab_full
cp tokenizers/monolingual/aze_arab_100mb/* models/full/aze_arab_full

# aze_cyrl
if test -f models/full/aze_cyrl_full/pytorch_model.bin; then
echo "Model already found: aze_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8855 \
--warmup_steps=885 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_cyrl_full.txt \
--seed=43 \
--override_n_examples=7084 \
--output_dir=models/full/aze_cyrl_full
cp tokenizers/monolingual/aze_cyrl_full/* models/full/aze_cyrl_full

# azj_latn
if test -f models/full/azj_latn_full/pytorch_model.bin; then
echo "Model already found: azj_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/azj_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/azj_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=16505 \
--warmup_steps=1650 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/azj_latn_full.txt \
--seed=43 \
--override_n_examples=52816 \
--output_dir=models/full/azj_latn_full
cp tokenizers/monolingual/azj_latn_100mb/* models/full/azj_latn_full

# bak_cyrl
if test -f models/full/bak_cyrl_full/pytorch_model.bin; then
echo "Model already found: bak_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bak_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bak_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=72246 \
--warmup_steps=7224 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bak_cyrl_full.txt \
--seed=43 \
--override_n_examples=231190 \
--output_dir=models/full/bak_cyrl_full
cp tokenizers/monolingual/bak_cyrl_100mb/* models/full/bak_cyrl_full

# bak_latn
if test -f models/full/bak_latn_full/pytorch_model.bin; then
echo "Model already found: bak_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bak_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bak_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8755 \
--warmup_steps=875 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bak_latn_full.txt \
--seed=43 \
--override_n_examples=3502 \
--output_dir=models/full/bak_latn_full
cp tokenizers/monolingual/bak_latn_full/* models/full/bak_latn_full

# bam_latn
if test -f models/full/bam_latn_full/pytorch_model.bin; then
echo "Model already found: bam_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bam_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bam_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11015 \
--warmup_steps=1101 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bam_latn_full.txt \
--seed=43 \
--override_n_examples=8812 \
--output_dir=models/full/bam_latn_full
cp tokenizers/monolingual/bam_latn_full/* models/full/bam_latn_full

# ban_latn
if test -f models/full/ban_latn_full/pytorch_model.bin; then
echo "Model already found: ban_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ban_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ban_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=22366 \
--warmup_steps=2236 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ban_latn_full.txt \
--seed=43 \
--override_n_examples=17893 \
--output_dir=models/full/ban_latn_full
cp tokenizers/monolingual/ban_latn_full/* models/full/ban_latn_full

# bar_latn
if test -f models/full/bar_latn_full/pytorch_model.bin; then
echo "Model already found: bar_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bar_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bar_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=19437 \
--warmup_steps=1943 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bar_latn_full.txt \
--seed=43 \
--override_n_examples=15550 \
--output_dir=models/full/bar_latn_full
cp tokenizers/monolingual/bar_latn_full/* models/full/bar_latn_full

# bbc_latn
if test -f models/full/bbc_latn_full/pytorch_model.bin; then
echo "Model already found: bbc_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bbc_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bbc_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9017 \
--warmup_steps=901 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bbc_latn_full.txt \
--seed=43 \
--override_n_examples=3607 \
--output_dir=models/full/bbc_latn_full
cp tokenizers/monolingual/bbc_latn_full/* models/full/bbc_latn_full

# bcl_latn
if test -f models/full/bcl_latn_full/pytorch_model.bin; then
echo "Model already found: bcl_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bcl_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bcl_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12882 \
--warmup_steps=1288 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bcl_latn_full.txt \
--seed=43 \
--override_n_examples=5153 \
--output_dir=models/full/bcl_latn_full
cp tokenizers/monolingual/bcl_latn_full/* models/full/bcl_latn_full

# bem_latn
if test -f models/full/bem_latn_full/pytorch_model.bin; then
echo "Model already found: bem_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bem_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bem_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=24846 \
--warmup_steps=2484 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bem_latn_full.txt \
--seed=43 \
--override_n_examples=19877 \
--output_dir=models/full/bem_latn_full
cp tokenizers/monolingual/bem_latn_full/* models/full/bem_latn_full

# bew_cyrl
if test -f models/full/bew_cyrl_full/pytorch_model.bin; then
echo "Model already found: bew_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bew_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bew_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12912 \
--warmup_steps=1291 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bew_cyrl_full.txt \
--seed=43 \
--override_n_examples=10330 \
--output_dir=models/full/bew_cyrl_full
cp tokenizers/monolingual/bew_cyrl_full/* models/full/bew_cyrl_full

# bew_latn
if test -f models/full/bew_latn_full/pytorch_model.bin; then
echo "Model already found: bew_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bew_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bew_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7778 \
--warmup_steps=777 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bew_latn_full.txt \
--seed=43 \
--override_n_examples=6223 \
--output_dir=models/full/bew_latn_full
cp tokenizers/monolingual/bew_latn_full/* models/full/bew_latn_full

# bgp_latn
if test -f models/full/bgp_latn_full/pytorch_model.bin; then
echo "Model already found: bgp_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bgp_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bgp_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8475 \
--warmup_steps=847 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bgp_latn_full.txt \
--seed=43 \
--override_n_examples=3390 \
--output_dir=models/full/bgp_latn_full
cp tokenizers/monolingual/bgp_latn_full/* models/full/bgp_latn_full

# bho_deva
if test -f models/full/bho_deva_full/pytorch_model.bin; then
echo "Model already found: bho_deva_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bho_deva_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bho_deva_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=15031 \
--warmup_steps=1503 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bho_deva_full.txt \
--seed=43 \
--override_n_examples=12025 \
--output_dir=models/full/bho_deva_full
cp tokenizers/monolingual/bho_deva_full/* models/full/bho_deva_full

# bik_latn
if test -f models/full/bik_latn_full/pytorch_model.bin; then
echo "Model already found: bik_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bik_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bik_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=13282 \
--warmup_steps=1328 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bik_latn_full.txt \
--seed=43 \
--override_n_examples=10626 \
--output_dir=models/full/bik_latn_full
cp tokenizers/monolingual/bik_latn_full/* models/full/bik_latn_full

# bjn_latn
if test -f models/full/bjn_latn_full/pytorch_model.bin; then
echo "Model already found: bjn_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bjn_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bjn_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=61065 \
--warmup_steps=6106 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bjn_latn_full.txt \
--seed=43 \
--override_n_examples=48852 \
--output_dir=models/full/bjn_latn_full
cp tokenizers/monolingual/bjn_latn_full/* models/full/bjn_latn_full

# bod_tibt
if test -f models/full/bod_tibt_full/pytorch_model.bin; then
echo "Model already found: bod_tibt_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bod_tibt_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bod_tibt_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=14320 \
--warmup_steps=1432 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bod_tibt_full.txt \
--seed=43 \
--override_n_examples=45827 \
--output_dir=models/full/bod_tibt_full
cp tokenizers/monolingual/bod_tibt_100mb/* models/full/bod_tibt_full

# bpy_beng
if test -f models/full/bpy_beng_full/pytorch_model.bin; then
echo "Model already found: bpy_beng_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bpy_beng_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bpy_beng_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9860 \
--warmup_steps=986 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bpy_beng_full.txt \
--seed=43 \
--override_n_examples=3944 \
--output_dir=models/full/bpy_beng_full
cp tokenizers/monolingual/bpy_beng_full/* models/full/bpy_beng_full

# bqc_latn
if test -f models/full/bqc_latn_full/pytorch_model.bin; then
echo "Model already found: bqc_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bqc_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bqc_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8820 \
--warmup_steps=882 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bqc_latn_full.txt \
--seed=43 \
--override_n_examples=3528 \
--output_dir=models/full/bqc_latn_full
cp tokenizers/monolingual/bqc_latn_full/* models/full/bqc_latn_full

# bre_latn
if test -f models/full/bre_latn_full/pytorch_model.bin; then
echo "Model already found: bre_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bre_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bre_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=26511 \
--warmup_steps=2651 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bre_latn_full.txt \
--seed=43 \
--override_n_examples=84838 \
--output_dir=models/full/bre_latn_full
cp tokenizers/monolingual/bre_latn_100mb/* models/full/bre_latn_full

# bsb_latn
if test -f models/full/bsb_latn_full/pytorch_model.bin; then
echo "Model already found: bsb_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bsb_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bsb_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12015 \
--warmup_steps=1201 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bsb_latn_full.txt \
--seed=43 \
--override_n_examples=4806 \
--output_dir=models/full/bsb_latn_full
cp tokenizers/monolingual/bsb_latn_full/* models/full/bsb_latn_full

# bua_cyrl
if test -f models/full/bua_cyrl_full/pytorch_model.bin; then
echo "Model already found: bua_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bua_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bua_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=21855 \
--warmup_steps=2185 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bua_cyrl_full.txt \
--seed=43 \
--override_n_examples=17484 \
--output_dir=models/full/bua_cyrl_full
cp tokenizers/monolingual/bua_cyrl_full/* models/full/bua_cyrl_full

# bug_latn
if test -f models/full/bug_latn_full/pytorch_model.bin; then
echo "Model already found: bug_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bug_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bug_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7982 \
--warmup_steps=798 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bug_latn_full.txt \
--seed=43 \
--override_n_examples=6386 \
--output_dir=models/full/bug_latn_full
cp tokenizers/monolingual/bug_latn_full/* models/full/bug_latn_full

# bxr_cyrl
if test -f models/full/bxr_cyrl_full/pytorch_model.bin; then
echo "Model already found: bxr_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bxr_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bxr_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=14796 \
--warmup_steps=1479 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bxr_cyrl_full.txt \
--seed=43 \
--override_n_examples=11837 \
--output_dir=models/full/bxr_cyrl_full
cp tokenizers/monolingual/bxr_cyrl_full/* models/full/bxr_cyrl_full

# cak_latn
if test -f models/full/cak_latn_full/pytorch_model.bin; then
echo "Model already found: cak_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cak_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cak_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=20302 \
--warmup_steps=2030 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cak_latn_full.txt \
--seed=43 \
--override_n_examples=8121 \
--output_dir=models/full/cak_latn_full
cp tokenizers/monolingual/cak_latn_full/* models/full/cak_latn_full

# ceb_latn
if test -f models/full/ceb_latn_full/pytorch_model.bin; then
echo "Model already found: ceb_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ceb_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ceb_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=85633 \
--warmup_steps=8563 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ceb_latn_full.txt \
--seed=43 \
--override_n_examples=274026 \
--output_dir=models/full/ceb_latn_full
cp tokenizers/monolingual/ceb_latn_100mb/* models/full/ceb_latn_full

# cfm_latn
if test -f models/full/cfm_latn_full/pytorch_model.bin; then
echo "Model already found: cfm_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cfm_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cfm_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10536 \
--warmup_steps=1053 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cfm_latn_full.txt \
--seed=43 \
--override_n_examples=8429 \
--output_dir=models/full/cfm_latn_full
cp tokenizers/monolingual/cfm_latn_full/* models/full/cfm_latn_full

# che_cyrl
if test -f models/full/che_cyrl_full/pytorch_model.bin; then
echo "Model already found: che_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/che_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/che_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=57593 \
--warmup_steps=5759 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/che_cyrl_full.txt \
--seed=43 \
--override_n_examples=46075 \
--output_dir=models/full/che_cyrl_full
cp tokenizers/monolingual/che_cyrl_full/* models/full/che_cyrl_full

# chm_cyrl
if test -f models/full/chm_cyrl_full/pytorch_model.bin; then
echo "Model already found: chm_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/chm_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/chm_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=27565 \
--warmup_steps=2756 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/chm_cyrl_full.txt \
--seed=43 \
--override_n_examples=22052 \
--output_dir=models/full/chm_cyrl_full
cp tokenizers/monolingual/chm_cyrl_full/* models/full/chm_cyrl_full

# chv_cyrl
if test -f models/full/chv_cyrl_full/pytorch_model.bin; then
echo "Model already found: chv_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/chv_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/chv_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=51448 \
--warmup_steps=5144 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/chv_cyrl_full.txt \
--seed=43 \
--override_n_examples=164635 \
--output_dir=models/full/chv_cyrl_full
cp tokenizers/monolingual/chv_cyrl_100mb/* models/full/chv_cyrl_full

# cjk_latn
if test -f models/full/cjk_latn_full/pytorch_model.bin; then
echo "Model already found: cjk_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cjk_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cjk_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8843 \
--warmup_steps=884 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cjk_latn_full.txt \
--seed=43 \
--override_n_examples=7075 \
--output_dir=models/full/cjk_latn_full
cp tokenizers/monolingual/cjk_latn_full/* models/full/cjk_latn_full

# ckb_arab
if test -f models/full/ckb_arab_full/pytorch_model.bin; then
echo "Model already found: ckb_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ckb_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ckb_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=116312 \
--warmup_steps=11631 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ckb_arab_full.txt \
--seed=43 \
--override_n_examples=372199 \
--output_dir=models/full/ckb_arab_full
cp tokenizers/monolingual/ckb_arab_100mb/* models/full/ckb_arab_full

# cnh_latn
if test -f models/full/cnh_latn_full/pytorch_model.bin; then
echo "Model already found: cnh_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cnh_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cnh_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=25305 \
--warmup_steps=2530 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cnh_latn_full.txt \
--seed=43 \
--override_n_examples=20244 \
--output_dir=models/full/cnh_latn_full
cp tokenizers/monolingual/cnh_latn_full/* models/full/cnh_latn_full

# cor_latn
if test -f models/full/cor_latn_full/pytorch_model.bin; then
echo "Model already found: cor_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cor_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cor_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9457 \
--warmup_steps=945 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cor_latn_full.txt \
--seed=43 \
--override_n_examples=3783 \
--output_dir=models/full/cor_latn_full
cp tokenizers/monolingual/cor_latn_full/* models/full/cor_latn_full

# cos_latn
if test -f models/full/cos_latn_full/pytorch_model.bin; then
echo "Model already found: cos_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cos_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cos_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=76996 \
--warmup_steps=7699 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cos_latn_full.txt \
--seed=43 \
--override_n_examples=246388 \
--output_dir=models/full/cos_latn_full
cp tokenizers/monolingual/cos_latn_100mb/* models/full/cos_latn_full

# crh_cyrl
if test -f models/full/crh_cyrl_full/pytorch_model.bin; then
echo "Model already found: crh_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/crh_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/crh_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6180 \
--warmup_steps=618 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/crh_cyrl_full.txt \
--seed=43 \
--override_n_examples=2472 \
--output_dir=models/full/crh_cyrl_full
cp tokenizers/monolingual/crh_cyrl_full/* models/full/crh_cyrl_full

# crh_latn
if test -f models/full/crh_latn_full/pytorch_model.bin; then
echo "Model already found: crh_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/crh_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/crh_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=31725 \
--warmup_steps=3172 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/crh_latn_full.txt \
--seed=43 \
--override_n_examples=25380 \
--output_dir=models/full/crh_latn_full
cp tokenizers/monolingual/crh_latn_full/* models/full/crh_latn_full

# ctd_latn
if test -f models/full/ctd_latn_full/pytorch_model.bin; then
echo "Model already found: ctd_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ctd_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ctd_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=27846 \
--warmup_steps=2784 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ctd_latn_full.txt \
--seed=43 \
--override_n_examples=22277 \
--output_dir=models/full/ctd_latn_full
cp tokenizers/monolingual/ctd_latn_full/* models/full/ctd_latn_full

# dar_cyrl
if test -f models/full/dar_cyrl_full/pytorch_model.bin; then
echo "Model already found: dar_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dar_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dar_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11002 \
--warmup_steps=1100 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dar_cyrl_full.txt \
--seed=43 \
--override_n_examples=8802 \
--output_dir=models/full/dar_cyrl_full
cp tokenizers/monolingual/dar_cyrl_full/* models/full/dar_cyrl_full

# dik_latn
if test -f models/full/dik_latn_full/pytorch_model.bin; then
echo "Model already found: dik_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dik_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dik_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9165 \
--warmup_steps=916 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dik_latn_full.txt \
--seed=43 \
--override_n_examples=7332 \
--output_dir=models/full/dik_latn_full
cp tokenizers/monolingual/dik_latn_full/* models/full/dik_latn_full

# din_latn
if test -f models/full/din_latn_full/pytorch_model.bin; then
echo "Model already found: din_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/din_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/din_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10072 \
--warmup_steps=1007 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/din_latn_full.txt \
--seed=43 \
--override_n_examples=8058 \
--output_dir=models/full/din_latn_full
cp tokenizers/monolingual/din_latn_full/* models/full/din_latn_full

# diq_latn
if test -f models/full/diq_latn_full/pytorch_model.bin; then
echo "Model already found: diq_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/diq_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/diq_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=24257 \
--warmup_steps=2425 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/diq_latn_full.txt \
--seed=43 \
--override_n_examples=19406 \
--output_dir=models/full/diq_latn_full
cp tokenizers/monolingual/diq_latn_full/* models/full/diq_latn_full

# div_thaa
if test -f models/full/div_thaa_full/pytorch_model.bin; then
echo "Model already found: div_thaa_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/div_thaa_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/div_thaa_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=69891 \
--warmup_steps=6989 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/div_thaa_full.txt \
--seed=43 \
--override_n_examples=223653 \
--output_dir=models/full/div_thaa_full
cp tokenizers/monolingual/div_thaa_100mb/* models/full/div_thaa_full

# dov_latn
if test -f models/full/dov_latn_full/pytorch_model.bin; then
echo "Model already found: dov_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dov_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dov_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6782 \
--warmup_steps=678 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dov_latn_full.txt \
--seed=43 \
--override_n_examples=2713 \
--output_dir=models/full/dov_latn_full
cp tokenizers/monolingual/dov_latn_full/* models/full/dov_latn_full

# dyu_latn
if test -f models/full/dyu_latn_full/pytorch_model.bin; then
echo "Model already found: dyu_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dyu_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dyu_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9397 \
--warmup_steps=939 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dyu_latn_full.txt \
--seed=43 \
--override_n_examples=7518 \
--output_dir=models/full/dyu_latn_full
cp tokenizers/monolingual/dyu_latn_full/* models/full/dyu_latn_full

# dzo_tibt
if test -f models/full/dzo_tibt_full/pytorch_model.bin; then
echo "Model already found: dzo_tibt_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dzo_tibt_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dzo_tibt_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9860 \
--warmup_steps=986 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dzo_tibt_full.txt \
--seed=43 \
--override_n_examples=3944 \
--output_dir=models/full/dzo_tibt_full
cp tokenizers/monolingual/dzo_tibt_full/* models/full/dzo_tibt_full

# ekk_latn
if test -f models/full/ekk_latn_full/pytorch_model.bin; then
echo "Model already found: ekk_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ekk_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ekk_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=30213 \
--warmup_steps=3021 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ekk_latn_full.txt \
--seed=43 \
--override_n_examples=24171 \
--output_dir=models/full/ekk_latn_full
cp tokenizers/monolingual/ekk_latn_full/* models/full/ekk_latn_full

# ell_latn
if test -f models/full/ell_latn_full/pytorch_model.bin; then
echo "Model already found: ell_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ell_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ell_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=56290 \
--warmup_steps=5629 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ell_latn_full.txt \
--seed=43 \
--override_n_examples=180128 \
--output_dir=models/full/ell_latn_full
cp tokenizers/monolingual/ell_latn_100mb/* models/full/ell_latn_full

# ewe_latn
if test -f models/full/ewe_latn_full/pytorch_model.bin; then
echo "Model already found: ewe_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ewe_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ewe_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=45093 \
--warmup_steps=4509 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ewe_latn_full.txt \
--seed=43 \
--override_n_examples=36075 \
--output_dir=models/full/ewe_latn_full
cp tokenizers/monolingual/ewe_latn_full/* models/full/ewe_latn_full

# fao_latn
if test -f models/full/fao_latn_full/pytorch_model.bin; then
echo "Model already found: fao_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fao_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fao_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=58952 \
--warmup_steps=5895 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fao_latn_full.txt \
--seed=43 \
--override_n_examples=188648 \
--output_dir=models/full/fao_latn_full
cp tokenizers/monolingual/fao_latn_100mb/* models/full/fao_latn_full

# fij_latn
if test -f models/full/fij_latn_full/pytorch_model.bin; then
echo "Model already found: fij_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fij_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fij_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=25983 \
--warmup_steps=2598 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fij_latn_full.txt \
--seed=43 \
--override_n_examples=20787 \
--output_dir=models/full/fij_latn_full
cp tokenizers/monolingual/fij_latn_full/* models/full/fij_latn_full

# fon_latn
if test -f models/full/fon_latn_full/pytorch_model.bin; then
echo "Model already found: fon_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fon_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fon_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=34165 \
--warmup_steps=3416 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fon_latn_full.txt \
--seed=43 \
--override_n_examples=27332 \
--output_dir=models/full/fon_latn_full
cp tokenizers/monolingual/fon_latn_full/* models/full/fon_latn_full

# frr_latn
if test -f models/full/frr_latn_full/pytorch_model.bin; then
echo "Model already found: frr_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/frr_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/frr_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7785 \
--warmup_steps=778 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/frr_latn_full.txt \
--seed=43 \
--override_n_examples=3114 \
--output_dir=models/full/frr_latn_full
cp tokenizers/monolingual/frr_latn_full/* models/full/frr_latn_full

# fry_latn
if test -f models/full/fry_latn_full/pytorch_model.bin; then
echo "Model already found: fry_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fry_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fry_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=81220 \
--warmup_steps=8122 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fry_latn_full.txt \
--seed=43 \
--override_n_examples=259907 \
--output_dir=models/full/fry_latn_full
cp tokenizers/monolingual/fry_latn_100mb/* models/full/fry_latn_full

# ful_latn
if test -f models/full/ful_latn_full/pytorch_model.bin; then
echo "Model already found: ful_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ful_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ful_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=19058 \
--warmup_steps=1905 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ful_latn_full.txt \
--seed=43 \
--override_n_examples=15247 \
--output_dir=models/full/ful_latn_full
cp tokenizers/monolingual/ful_latn_full/* models/full/ful_latn_full

# fur_latn
if test -f models/full/fur_latn_full/pytorch_model.bin; then
echo "Model already found: fur_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fur_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fur_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=13397 \
--warmup_steps=1339 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fur_latn_full.txt \
--seed=43 \
--override_n_examples=10718 \
--output_dir=models/full/fur_latn_full
cp tokenizers/monolingual/fur_latn_full/* models/full/fur_latn_full

# fuv_latn
if test -f models/full/fuv_latn_full/pytorch_model.bin; then
echo "Model already found: fuv_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fuv_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fuv_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=15038 \
--warmup_steps=1503 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fuv_latn_full.txt \
--seed=43 \
--override_n_examples=12031 \
--output_dir=models/full/fuv_latn_full
cp tokenizers/monolingual/fuv_latn_full/* models/full/fuv_latn_full

# gaz_latn
if test -f models/full/gaz_latn_full/pytorch_model.bin; then
echo "Model already found: gaz_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gaz_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gaz_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=62415 \
--warmup_steps=6241 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gaz_latn_full.txt \
--seed=43 \
--override_n_examples=49932 \
--output_dir=models/full/gaz_latn_full
cp tokenizers/monolingual/gaz_latn_full/* models/full/gaz_latn_full

# gla_latn
if test -f models/full/gla_latn_full/pytorch_model.bin; then
echo "Model already found: gla_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gla_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gla_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=75522 \
--warmup_steps=7552 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gla_latn_full.txt \
--seed=43 \
--override_n_examples=241672 \
--output_dir=models/full/gla_latn_full
cp tokenizers/monolingual/gla_latn_100mb/* models/full/gla_latn_full

# gle_latn
if test -f models/full/gle_latn_full/pytorch_model.bin; then
echo "Model already found: gle_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gle_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gle_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=247084 \
--warmup_steps=24708 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gle_latn_full.txt \
--seed=43 \
--override_n_examples=790670 \
--output_dir=models/full/gle_latn_full
cp tokenizers/monolingual/gle_latn_100mb/* models/full/gle_latn_full

# glk_arab
if test -f models/full/glk_arab_full/pytorch_model.bin; then
echo "Model already found: glk_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/glk_arab_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/glk_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=62302 \
--warmup_steps=6230 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/glk_arab_full.txt \
--seed=43 \
--override_n_examples=49842 \
--output_dir=models/full/glk_arab_full
cp tokenizers/monolingual/glk_arab_full/* models/full/glk_arab_full

# glv_latn
if test -f models/full/glv_latn_full/pytorch_model.bin; then
echo "Model already found: glv_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/glv_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/glv_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9472 \
--warmup_steps=947 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/glv_latn_full.txt \
--seed=43 \
--override_n_examples=3789 \
--output_dir=models/full/glv_latn_full
cp tokenizers/monolingual/glv_latn_full/* models/full/glv_latn_full

# gom_deva
if test -f models/full/gom_deva_full/pytorch_model.bin; then
echo "Model already found: gom_deva_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gom_deva_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gom_deva_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5418 \
--warmup_steps=541 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gom_deva_full.txt \
--seed=43 \
--override_n_examples=4335 \
--output_dir=models/full/gom_deva_full
cp tokenizers/monolingual/gom_deva_full/* models/full/gom_deva_full

# gom_latn
if test -f models/full/gom_latn_full/pytorch_model.bin; then
echo "Model already found: gom_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gom_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gom_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6852 \
--warmup_steps=685 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gom_latn_full.txt \
--seed=43 \
--override_n_examples=5482 \
--output_dir=models/full/gom_latn_full
cp tokenizers/monolingual/gom_latn_full/* models/full/gom_latn_full

# grc_grek
if test -f models/full/grc_grek_full/pytorch_model.bin; then
echo "Model already found: grc_grek_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/grc_grek_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/grc_grek_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=29065 \
--warmup_steps=2906 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/grc_grek_full.txt \
--seed=43 \
--override_n_examples=93009 \
--output_dir=models/full/grc_grek_full
cp tokenizers/monolingual/grc_grek_100mb/* models/full/grc_grek_full

# grn_latn
if test -f models/full/grn_latn_full/pytorch_model.bin; then
echo "Model already found: grn_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/grn_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/grn_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=37516 \
--warmup_steps=3751 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/grn_latn_full.txt \
--seed=43 \
--override_n_examples=30013 \
--output_dir=models/full/grn_latn_full
cp tokenizers/monolingual/grn_latn_full/* models/full/grn_latn_full

# gsw_latn
if test -f models/full/gsw_latn_full/pytorch_model.bin; then
echo "Model already found: gsw_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gsw_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gsw_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=23563 \
--warmup_steps=2356 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gsw_latn_full.txt \
--seed=43 \
--override_n_examples=75402 \
--output_dir=models/full/gsw_latn_full
cp tokenizers/monolingual/gsw_latn_100mb/* models/full/gsw_latn_full

# guj_latn
if test -f models/full/guj_latn_full/pytorch_model.bin; then
echo "Model already found: guj_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/guj_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/guj_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=15036 \
--warmup_steps=1503 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/guj_latn_full.txt \
--seed=43 \
--override_n_examples=48116 \
--output_dir=models/full/guj_latn_full
cp tokenizers/monolingual/guj_latn_100mb/* models/full/guj_latn_full

# hat_latn
if test -f models/full/hat_latn_full/pytorch_model.bin; then
echo "Model already found: hat_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hat_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hat_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=113118 \
--warmup_steps=11311 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hat_latn_full.txt \
--seed=43 \
--override_n_examples=361979 \
--output_dir=models/full/hat_latn_full
cp tokenizers/monolingual/hat_latn_100mb/* models/full/hat_latn_full

# haw_latn
if test -f models/full/haw_latn_full/pytorch_model.bin; then
echo "Model already found: haw_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/haw_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/haw_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=52946 \
--warmup_steps=5294 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/haw_latn_full.txt \
--seed=43 \
--override_n_examples=169428 \
--output_dir=models/full/haw_latn_full
cp tokenizers/monolingual/haw_latn_100mb/* models/full/haw_latn_full

# hif_latn
if test -f models/full/hif_latn_full/pytorch_model.bin; then
echo "Model already found: hif_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hif_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hif_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=21406 \
--warmup_steps=2140 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hif_latn_full.txt \
--seed=43 \
--override_n_examples=17125 \
--output_dir=models/full/hif_latn_full
cp tokenizers/monolingual/hif_latn_full/* models/full/hif_latn_full

# hil_latn
if test -f models/full/hil_latn_full/pytorch_model.bin; then
echo "Model already found: hil_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hil_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hil_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=22057 \
--warmup_steps=2205 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hil_latn_full.txt \
--seed=43 \
--override_n_examples=17646 \
--output_dir=models/full/hil_latn_full
cp tokenizers/monolingual/hil_latn_full/* models/full/hil_latn_full

# hin_latn
if test -f models/full/hin_latn_full/pytorch_model.bin; then
echo "Model already found: hin_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hin_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hin_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=23000 \
--warmup_steps=2300 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hin_latn_full.txt \
--seed=43 \
--override_n_examples=73601 \
--output_dir=models/full/hin_latn_full
cp tokenizers/monolingual/hin_latn_100mb/* models/full/hin_latn_full

# hmn_latn
if test -f models/full/hmn_latn_full/pytorch_model.bin; then
echo "Model already found: hmn_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hmn_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hmn_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=61066 \
--warmup_steps=6106 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hmn_latn_full.txt \
--seed=43 \
--override_n_examples=195414 \
--output_dir=models/full/hmn_latn_full
cp tokenizers/monolingual/hmn_latn_100mb/* models/full/hmn_latn_full

# hne_deva
if test -f models/full/hne_deva_full/pytorch_model.bin; then
echo "Model already found: hne_deva_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hne_deva_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hne_deva_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10287 \
--warmup_steps=1028 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hne_deva_full.txt \
--seed=43 \
--override_n_examples=4115 \
--output_dir=models/full/hne_deva_full
cp tokenizers/monolingual/hne_deva_full/* models/full/hne_deva_full

# hsb_latn
if test -f models/full/hsb_latn_full/pytorch_model.bin; then
echo "Model already found: hsb_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hsb_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hsb_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6112 \
--warmup_steps=611 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hsb_latn_full.txt \
--seed=43 \
--override_n_examples=4890 \
--output_dir=models/full/hsb_latn_full
cp tokenizers/monolingual/hsb_latn_full/* models/full/hsb_latn_full

# iba_latn
if test -f models/full/iba_latn_full/pytorch_model.bin; then
echo "Model already found: iba_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/iba_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/iba_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12886 \
--warmup_steps=1288 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/iba_latn_full.txt \
--seed=43 \
--override_n_examples=10309 \
--output_dir=models/full/iba_latn_full
cp tokenizers/monolingual/iba_latn_full/* models/full/iba_latn_full

# ibo_latn
if test -f models/full/ibo_latn_full/pytorch_model.bin; then
echo "Model already found: ibo_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ibo_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ibo_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=73062 \
--warmup_steps=7306 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ibo_latn_full.txt \
--seed=43 \
--override_n_examples=233801 \
--output_dir=models/full/ibo_latn_full
cp tokenizers/monolingual/ibo_latn_100mb/* models/full/ibo_latn_full

# ido_latn
if test -f models/full/ido_latn_full/pytorch_model.bin; then
echo "Model already found: ido_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ido_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ido_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=17991 \
--warmup_steps=1799 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ido_latn_full.txt \
--seed=43 \
--override_n_examples=14393 \
--output_dir=models/full/ido_latn_full
cp tokenizers/monolingual/ido_latn_full/* models/full/ido_latn_full

# iku_cans
if test -f models/full/iku_cans_full/pytorch_model.bin; then
echo "Model already found: iku_cans_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/iku_cans_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/iku_cans_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=33687 \
--warmup_steps=3368 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/iku_cans_full.txt \
--seed=43 \
--override_n_examples=26950 \
--output_dir=models/full/iku_cans_full
cp tokenizers/monolingual/iku_cans_full/* models/full/iku_cans_full

# ilo_latn
if test -f models/full/ilo_latn_full/pytorch_model.bin; then
echo "Model already found: ilo_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ilo_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ilo_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=62135 \
--warmup_steps=6213 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ilo_latn_full.txt \
--seed=43 \
--override_n_examples=49708 \
--output_dir=models/full/ilo_latn_full
cp tokenizers/monolingual/ilo_latn_full/* models/full/ilo_latn_full

# ina_latn
if test -f models/full/ina_latn_full/pytorch_model.bin; then
echo "Model already found: ina_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ina_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ina_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=13627 \
--warmup_steps=1362 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ina_latn_full.txt \
--seed=43 \
--override_n_examples=10902 \
--output_dir=models/full/ina_latn_full
cp tokenizers/monolingual/ina_latn_full/* models/full/ina_latn_full

# inh_cyrl
if test -f models/full/inh_cyrl_full/pytorch_model.bin; then
echo "Model already found: inh_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/inh_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/inh_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=13500 \
--warmup_steps=1350 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/inh_cyrl_full.txt \
--seed=43 \
--override_n_examples=5400 \
--output_dir=models/full/inh_cyrl_full
cp tokenizers/monolingual/inh_cyrl_full/* models/full/inh_cyrl_full

# iso_latn
if test -f models/full/iso_latn_full/pytorch_model.bin; then
echo "Model already found: iso_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/iso_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/iso_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12882 \
--warmup_steps=1288 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/iso_latn_full.txt \
--seed=43 \
--override_n_examples=5153 \
--output_dir=models/full/iso_latn_full
cp tokenizers/monolingual/iso_latn_full/* models/full/iso_latn_full

# jav_latn
if test -f models/full/jav_latn_full/pytorch_model.bin; then
echo "Model already found: jav_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/jav_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/jav_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=70393 \
--warmup_steps=7039 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/jav_latn_full.txt \
--seed=43 \
--override_n_examples=225258 \
--output_dir=models/full/jav_latn_full
cp tokenizers/monolingual/jav_latn_100mb/* models/full/jav_latn_full

# kaa_latn
if test -f models/full/kaa_latn_full/pytorch_model.bin; then
echo "Model already found: kaa_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaa_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaa_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=23661 \
--warmup_steps=2366 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaa_latn_full.txt \
--seed=43 \
--override_n_examples=75717 \
--output_dir=models/full/kaa_latn_full
cp tokenizers/monolingual/kaa_latn_100mb/* models/full/kaa_latn_full

# kab_latn
if test -f models/full/kab_latn_full/pytorch_model.bin; then
echo "Model already found: kab_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kab_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kab_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=34266 \
--warmup_steps=3426 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kab_latn_full.txt \
--seed=43 \
--override_n_examples=27413 \
--output_dir=models/full/kab_latn_full
cp tokenizers/monolingual/kab_latn_full/* models/full/kab_latn_full

# kac_latn
if test -f models/full/kac_latn_full/pytorch_model.bin; then
echo "Model already found: kac_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kac_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kac_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10873 \
--warmup_steps=1087 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kac_latn_full.txt \
--seed=43 \
--override_n_examples=8699 \
--output_dir=models/full/kac_latn_full
cp tokenizers/monolingual/kac_latn_full/* models/full/kac_latn_full

# kal_latn
if test -f models/full/kal_latn_full/pytorch_model.bin; then
echo "Model already found: kal_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kal_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kal_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=18360 \
--warmup_steps=1836 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kal_latn_full.txt \
--seed=43 \
--override_n_examples=58754 \
--output_dir=models/full/kal_latn_full
cp tokenizers/monolingual/kal_latn_100mb/* models/full/kal_latn_full

# kas_deva
if test -f models/full/kas_deva_full/pytorch_model.bin; then
echo "Model already found: kas_deva_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kas_deva_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kas_deva_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9720 \
--warmup_steps=972 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kas_deva_full.txt \
--seed=43 \
--override_n_examples=3888 \
--output_dir=models/full/kas_deva_full
cp tokenizers/monolingual/kas_deva_full/* models/full/kas_deva_full

# kat_latn
if test -f models/full/kat_latn_full/pytorch_model.bin; then
echo "Model already found: kat_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kat_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kat_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6945 \
--warmup_steps=694 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kat_latn_full.txt \
--seed=43 \
--override_n_examples=2778 \
--output_dir=models/full/kat_latn_full
cp tokenizers/monolingual/kat_latn_full/* models/full/kat_latn_full

# kbd_cyrl
if test -f models/full/kbd_cyrl_full/pytorch_model.bin; then
echo "Model already found: kbd_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kbd_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kbd_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=23932 \
--warmup_steps=2393 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kbd_cyrl_full.txt \
--seed=43 \
--override_n_examples=19146 \
--output_dir=models/full/kbd_cyrl_full
cp tokenizers/monolingual/kbd_cyrl_full/* models/full/kbd_cyrl_full

# kbp_latn
if test -f models/full/kbp_latn_full/pytorch_model.bin; then
echo "Model already found: kbp_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kbp_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kbp_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11471 \
--warmup_steps=1147 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kbp_latn_full.txt \
--seed=43 \
--override_n_examples=9177 \
--output_dir=models/full/kbp_latn_full
cp tokenizers/monolingual/kbp_latn_full/* models/full/kbp_latn_full

# kea_latn
if test -f models/full/kea_latn_full/pytorch_model.bin; then
echo "Model already found: kea_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kea_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kea_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7928 \
--warmup_steps=792 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kea_latn_full.txt \
--seed=43 \
--override_n_examples=6343 \
--output_dir=models/full/kea_latn_full
cp tokenizers/monolingual/kea_latn_full/* models/full/kea_latn_full

# kha_latn
if test -f models/full/kha_latn_full/pytorch_model.bin; then
echo "Model already found: kha_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kha_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kha_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=15160 \
--warmup_steps=1516 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kha_latn_full.txt \
--seed=43 \
--override_n_examples=12128 \
--output_dir=models/full/kha_latn_full
cp tokenizers/monolingual/kha_latn_full/* models/full/kha_latn_full

# khk_cyrl
if test -f models/full/khk_cyrl_full/pytorch_model.bin; then
echo "Model already found: khk_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/khk_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/khk_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=14407 \
--warmup_steps=1440 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/khk_cyrl_full.txt \
--seed=43 \
--override_n_examples=46105 \
--output_dir=models/full/khk_cyrl_full
cp tokenizers/monolingual/khk_cyrl_100mb/* models/full/khk_cyrl_full

# khm_khmr
if test -f models/full/khm_khmr_full/pytorch_model.bin; then
echo "Model already found: khm_khmr_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/khm_khmr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/khm_khmr_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=143774 \
--warmup_steps=14377 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/khm_khmr_full.txt \
--seed=43 \
--override_n_examples=460077 \
--output_dir=models/full/khm_khmr_full
cp tokenizers/monolingual/khm_khmr_100mb/* models/full/khm_khmr_full

# kik_latn
if test -f models/full/kik_latn_full/pytorch_model.bin; then
echo "Model already found: kik_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kik_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kik_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11807 \
--warmup_steps=1180 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kik_latn_full.txt \
--seed=43 \
--override_n_examples=4723 \
--output_dir=models/full/kik_latn_full
cp tokenizers/monolingual/kik_latn_full/* models/full/kik_latn_full

# kin_latn
if test -f models/full/kin_latn_full/pytorch_model.bin; then
echo "Model already found: kin_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kin_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kin_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=118140 \
--warmup_steps=11814 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kin_latn_full.txt \
--seed=43 \
--override_n_examples=378049 \
--output_dir=models/full/kin_latn_full
cp tokenizers/monolingual/kin_latn_100mb/* models/full/kin_latn_full

# kjh_cyrl
if test -f models/full/kjh_cyrl_full/pytorch_model.bin; then
echo "Model already found: kjh_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kjh_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kjh_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6210 \
--warmup_steps=621 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kjh_cyrl_full.txt \
--seed=43 \
--override_n_examples=2484 \
--output_dir=models/full/kjh_cyrl_full
cp tokenizers/monolingual/kjh_cyrl_full/* models/full/kjh_cyrl_full

# kmb_latn
if test -f models/full/kmb_latn_full/pytorch_model.bin; then
echo "Model already found: kmb_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kmb_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kmb_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8202 \
--warmup_steps=820 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kmb_latn_full.txt \
--seed=43 \
--override_n_examples=6562 \
--output_dir=models/full/kmb_latn_full
cp tokenizers/monolingual/kmb_latn_full/* models/full/kmb_latn_full

# kmr_latn
if test -f models/full/kmr_latn_full/pytorch_model.bin; then
echo "Model already found: kmr_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kmr_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kmr_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=30027 \
--warmup_steps=3002 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kmr_latn_full.txt \
--seed=43 \
--override_n_examples=24022 \
--output_dir=models/full/kmr_latn_full
cp tokenizers/monolingual/kmr_latn_full/* models/full/kmr_latn_full

# knc_arab
if test -f models/full/knc_arab_full/pytorch_model.bin; then
echo "Model already found: knc_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/knc_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/knc_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=144911 \
--warmup_steps=14491 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/knc_arab_full.txt \
--seed=43 \
--override_n_examples=463716 \
--output_dir=models/full/knc_arab_full
cp tokenizers/monolingual/knc_arab_100mb/* models/full/knc_arab_full

# knc_latn
if test -f models/full/knc_latn_full/pytorch_model.bin; then
echo "Model already found: knc_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/knc_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/knc_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8382 \
--warmup_steps=838 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/knc_latn_full.txt \
--seed=43 \
--override_n_examples=6706 \
--output_dir=models/full/knc_latn_full
cp tokenizers/monolingual/knc_latn_full/* models/full/knc_latn_full

# kom_cyrl
if test -f models/full/kom_cyrl_full/pytorch_model.bin; then
echo "Model already found: kom_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kom_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kom_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11513 \
--warmup_steps=1151 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kom_cyrl_full.txt \
--seed=43 \
--override_n_examples=9211 \
--output_dir=models/full/kom_cyrl_full
cp tokenizers/monolingual/kom_cyrl_full/* models/full/kom_cyrl_full

# kon_latn
if test -f models/full/kon_latn_full/pytorch_model.bin; then
echo "Model already found: kon_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kon_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kon_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8665 \
--warmup_steps=866 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kon_latn_full.txt \
--seed=43 \
--override_n_examples=6932 \
--output_dir=models/full/kon_latn_full
cp tokenizers/monolingual/kon_latn_full/* models/full/kon_latn_full

# kpv_cyrl
if test -f models/full/kpv_cyrl_full/pytorch_model.bin; then
echo "Model already found: kpv_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kpv_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kpv_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6620 \
--warmup_steps=662 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kpv_cyrl_full.txt \
--seed=43 \
--override_n_examples=2648 \
--output_dir=models/full/kpv_cyrl_full
cp tokenizers/monolingual/kpv_cyrl_full/* models/full/kpv_cyrl_full

# krc_cyrl
if test -f models/full/krc_cyrl_full/pytorch_model.bin; then
echo "Model already found: krc_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/krc_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/krc_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11297 \
--warmup_steps=1129 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/krc_cyrl_full.txt \
--seed=43 \
--override_n_examples=9038 \
--output_dir=models/full/krc_cyrl_full
cp tokenizers/monolingual/krc_cyrl_full/* models/full/krc_cyrl_full

# kum_cyrl
if test -f models/full/kum_cyrl_full/pytorch_model.bin; then
echo "Model already found: kum_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kum_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kum_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5392 \
--warmup_steps=539 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kum_cyrl_full.txt \
--seed=43 \
--override_n_examples=4314 \
--output_dir=models/full/kum_cyrl_full
cp tokenizers/monolingual/kum_cyrl_full/* models/full/kum_cyrl_full

# kur_arab
if test -f models/full/kur_arab_full/pytorch_model.bin; then
echo "Model already found: kur_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kur_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kur_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=119924 \
--warmup_steps=11992 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kur_arab_full.txt \
--seed=43 \
--override_n_examples=383757 \
--output_dir=models/full/kur_arab_full
cp tokenizers/monolingual/kur_arab_100mb/* models/full/kur_arab_full

# kur_latn
if test -f models/full/kur_latn_full/pytorch_model.bin; then
echo "Model already found: kur_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kur_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kur_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=115888 \
--warmup_steps=11588 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kur_latn_full.txt \
--seed=43 \
--override_n_examples=370844 \
--output_dir=models/full/kur_latn_full
cp tokenizers/monolingual/kur_latn_100mb/* models/full/kur_latn_full

# lao_laoo
if test -f models/full/lao_laoo_full/pytorch_model.bin; then
echo "Model already found: lao_laoo_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lao_laoo_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lao_laoo_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=75730 \
--warmup_steps=7573 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lao_laoo_full.txt \
--seed=43 \
--override_n_examples=242338 \
--output_dir=models/full/lao_laoo_full
cp tokenizers/monolingual/lao_laoo_100mb/* models/full/lao_laoo_full

# lbe_cyrl
if test -f models/full/lbe_cyrl_full/pytorch_model.bin; then
echo "Model already found: lbe_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lbe_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lbe_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6032 \
--warmup_steps=603 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lbe_cyrl_full.txt \
--seed=43 \
--override_n_examples=4826 \
--output_dir=models/full/lbe_cyrl_full
cp tokenizers/monolingual/lbe_cyrl_full/* models/full/lbe_cyrl_full

# lez_cyrl
if test -f models/full/lez_cyrl_full/pytorch_model.bin; then
echo "Model already found: lez_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lez_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lez_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11517 \
--warmup_steps=1151 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lez_cyrl_full.txt \
--seed=43 \
--override_n_examples=4607 \
--output_dir=models/full/lez_cyrl_full
cp tokenizers/monolingual/lez_cyrl_full/* models/full/lez_cyrl_full

# lfn_latn
if test -f models/full/lfn_latn_full/pytorch_model.bin; then
echo "Model already found: lfn_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lfn_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lfn_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7780 \
--warmup_steps=778 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lfn_latn_full.txt \
--seed=43 \
--override_n_examples=3112 \
--output_dir=models/full/lfn_latn_full
cp tokenizers/monolingual/lfn_latn_full/* models/full/lfn_latn_full

# lij_latn
if test -f models/full/lij_latn_full/pytorch_model.bin; then
echo "Model already found: lij_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lij_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lij_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=20747 \
--warmup_steps=2074 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lij_latn_full.txt \
--seed=43 \
--override_n_examples=16598 \
--output_dir=models/full/lij_latn_full
cp tokenizers/monolingual/lij_latn_full/* models/full/lij_latn_full

# lim_latn
if test -f models/full/lim_latn_full/pytorch_model.bin; then
echo "Model already found: lim_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lim_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lim_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=24231 \
--warmup_steps=2423 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lim_latn_full.txt \
--seed=43 \
--override_n_examples=77540 \
--output_dir=models/full/lim_latn_full
cp tokenizers/monolingual/lim_latn_100mb/* models/full/lim_latn_full

# lin_latn
if test -f models/full/lin_latn_full/pytorch_model.bin; then
echo "Model already found: lin_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lin_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lin_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=32258 \
--warmup_steps=3225 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lin_latn_full.txt \
--seed=43 \
--override_n_examples=25807 \
--output_dir=models/full/lin_latn_full
cp tokenizers/monolingual/lin_latn_full/* models/full/lin_latn_full

# lmo_latn
if test -f models/full/lmo_latn_full/pytorch_model.bin; then
echo "Model already found: lmo_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lmo_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lmo_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=21730 \
--warmup_steps=2173 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lmo_latn_full.txt \
--seed=43 \
--override_n_examples=69538 \
--output_dir=models/full/lmo_latn_full
cp tokenizers/monolingual/lmo_latn_100mb/* models/full/lmo_latn_full

# ltg_latn
if test -f models/full/ltg_latn_full/pytorch_model.bin; then
echo "Model already found: ltg_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ltg_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ltg_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9880 \
--warmup_steps=988 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ltg_latn_full.txt \
--seed=43 \
--override_n_examples=7904 \
--output_dir=models/full/ltg_latn_full
cp tokenizers/monolingual/ltg_latn_full/* models/full/ltg_latn_full

# ltz_latn
if test -f models/full/ltz_latn_full/pytorch_model.bin; then
echo "Model already found: ltz_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ltz_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ltz_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=97778 \
--warmup_steps=9777 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ltz_latn_full.txt \
--seed=43 \
--override_n_examples=312891 \
--output_dir=models/full/ltz_latn_full
cp tokenizers/monolingual/ltz_latn_100mb/* models/full/ltz_latn_full

# lua_latn
if test -f models/full/lua_latn_full/pytorch_model.bin; then
echo "Model already found: lua_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lua_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lua_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=15436 \
--warmup_steps=1543 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lua_latn_full.txt \
--seed=43 \
--override_n_examples=12349 \
--output_dir=models/full/lua_latn_full
cp tokenizers/monolingual/lua_latn_full/* models/full/lua_latn_full

# lub_latn
if test -f models/full/lub_latn_full/pytorch_model.bin; then
echo "Model already found: lub_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lub_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lub_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11080 \
--warmup_steps=1108 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lub_latn_full.txt \
--seed=43 \
--override_n_examples=4432 \
--output_dir=models/full/lub_latn_full
cp tokenizers/monolingual/lub_latn_full/* models/full/lub_latn_full

# lug_latn
if test -f models/full/lug_latn_full/pytorch_model.bin; then
echo "Model already found: lug_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lug_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lug_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=22863 \
--warmup_steps=2286 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lug_latn_full.txt \
--seed=43 \
--override_n_examples=73164 \
--output_dir=models/full/lug_latn_full
cp tokenizers/monolingual/lug_latn_100mb/* models/full/lug_latn_full

# luo_latn
if test -f models/full/luo_latn_full/pytorch_model.bin; then
echo "Model already found: luo_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/luo_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/luo_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=24070 \
--warmup_steps=2407 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/luo_latn_full.txt \
--seed=43 \
--override_n_examples=19256 \
--output_dir=models/full/luo_latn_full
cp tokenizers/monolingual/luo_latn_full/* models/full/luo_latn_full

# lus_latn
if test -f models/full/lus_latn_full/pytorch_model.bin; then
echo "Model already found: lus_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lus_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lus_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=38290 \
--warmup_steps=3829 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lus_latn_full.txt \
--seed=43 \
--override_n_examples=122530 \
--output_dir=models/full/lus_latn_full
cp tokenizers/monolingual/lus_latn_100mb/* models/full/lus_latn_full

# lvs_latn
if test -f models/full/lvs_latn_full/pytorch_model.bin; then
echo "Model already found: lvs_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lvs_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lvs_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=20345 \
--warmup_steps=2034 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lvs_latn_full.txt \
--seed=43 \
--override_n_examples=16276 \
--output_dir=models/full/lvs_latn_full
cp tokenizers/monolingual/lvs_latn_full/* models/full/lvs_latn_full

# lzh_hant
if test -f models/full/lzh_hant_full/pytorch_model.bin; then
echo "Model already found: lzh_hant_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lzh_hant_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lzh_hant_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6757 \
--warmup_steps=675 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lzh_hant_full.txt \
--seed=43 \
--override_n_examples=5406 \
--output_dir=models/full/lzh_hant_full
cp tokenizers/monolingual/lzh_hant_full/* models/full/lzh_hant_full

# mad_latn
if test -f models/full/mad_latn_full/pytorch_model.bin; then
echo "Model already found: mad_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mad_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mad_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9982 \
--warmup_steps=998 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mad_latn_full.txt \
--seed=43 \
--override_n_examples=3993 \
--output_dir=models/full/mad_latn_full
cp tokenizers/monolingual/mad_latn_full/* models/full/mad_latn_full

# mag_deva
if test -f models/full/mag_deva_full/pytorch_model.bin; then
echo "Model already found: mag_deva_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mag_deva_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mag_deva_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12152 \
--warmup_steps=1215 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mag_deva_full.txt \
--seed=43 \
--override_n_examples=4861 \
--output_dir=models/full/mag_deva_full
cp tokenizers/monolingual/mag_deva_full/* models/full/mag_deva_full

# mai_deva
if test -f models/full/mai_deva_full/pytorch_model.bin; then
echo "Model already found: mai_deva_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mai_deva_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mai_deva_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=27243 \
--warmup_steps=2724 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mai_deva_full.txt \
--seed=43 \
--override_n_examples=21795 \
--output_dir=models/full/mai_deva_full
cp tokenizers/monolingual/mai_deva_full/* models/full/mai_deva_full

# mal_latn
if test -f models/full/mal_latn_full/pytorch_model.bin; then
echo "Model already found: mal_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mal_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mal_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7600 \
--warmup_steps=760 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mal_latn_full.txt \
--seed=43 \
--override_n_examples=3040 \
--output_dir=models/full/mal_latn_full
cp tokenizers/monolingual/mal_latn_full/* models/full/mal_latn_full

# mam_latn
if test -f models/full/mam_latn_full/pytorch_model.bin; then
echo "Model already found: mam_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mam_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mam_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=17482 \
--warmup_steps=1748 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mam_latn_full.txt \
--seed=43 \
--override_n_examples=6993 \
--output_dir=models/full/mam_latn_full
cp tokenizers/monolingual/mam_latn_full/* models/full/mam_latn_full

# mdf_cyrl
if test -f models/full/mdf_cyrl_full/pytorch_model.bin; then
echo "Model already found: mdf_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mdf_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mdf_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6357 \
--warmup_steps=635 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mdf_cyrl_full.txt \
--seed=43 \
--override_n_examples=2543 \
--output_dir=models/full/mdf_cyrl_full
cp tokenizers/monolingual/mdf_cyrl_full/* models/full/mdf_cyrl_full

# meo_latn
if test -f models/full/meo_latn_full/pytorch_model.bin; then
echo "Model already found: meo_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/meo_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/meo_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8157 \
--warmup_steps=815 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/meo_latn_full.txt \
--seed=43 \
--override_n_examples=3263 \
--output_dir=models/full/meo_latn_full
cp tokenizers/monolingual/meo_latn_full/* models/full/meo_latn_full

# mgh_latn
if test -f models/full/mgh_latn_full/pytorch_model.bin; then
echo "Model already found: mgh_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mgh_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mgh_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6110 \
--warmup_steps=611 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mgh_latn_full.txt \
--seed=43 \
--override_n_examples=2444 \
--output_dir=models/full/mgh_latn_full
cp tokenizers/monolingual/mgh_latn_full/* models/full/mgh_latn_full

# mhr_cyrl
if test -f models/full/mhr_cyrl_full/pytorch_model.bin; then
echo "Model already found: mhr_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mhr_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mhr_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=16065 \
--warmup_steps=1606 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mhr_cyrl_full.txt \
--seed=43 \
--override_n_examples=12852 \
--output_dir=models/full/mhr_cyrl_full
cp tokenizers/monolingual/mhr_cyrl_full/* models/full/mhr_cyrl_full

# min_latn
if test -f models/full/min_latn_full/pytorch_model.bin; then
echo "Model already found: min_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/min_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/min_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=43292 \
--warmup_steps=4329 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/min_latn_full.txt \
--seed=43 \
--override_n_examples=34634 \
--output_dir=models/full/min_latn_full
cp tokenizers/monolingual/min_latn_full/* models/full/min_latn_full

# mkw_cyrl
if test -f models/full/mkw_cyrl_full/pytorch_model.bin; then
echo "Model already found: mkw_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mkw_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mkw_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11065 \
--warmup_steps=1106 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mkw_cyrl_full.txt \
--seed=43 \
--override_n_examples=4426 \
--output_dir=models/full/mkw_cyrl_full
cp tokenizers/monolingual/mkw_cyrl_full/* models/full/mkw_cyrl_full

# mlg_latn
if test -f models/full/mlg_latn_full/pytorch_model.bin; then
echo "Model already found: mlg_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mlg_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mlg_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=128477 \
--warmup_steps=12847 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mlg_latn_full.txt \
--seed=43 \
--override_n_examples=411127 \
--output_dir=models/full/mlg_latn_full
cp tokenizers/monolingual/mlg_latn_100mb/* models/full/mlg_latn_full

# mon_latn
if test -f models/full/mon_latn_full/pytorch_model.bin; then
echo "Model already found: mon_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mon_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mon_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=30987 \
--warmup_steps=3098 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mon_latn_full.txt \
--seed=43 \
--override_n_examples=24790 \
--output_dir=models/full/mon_latn_full
cp tokenizers/monolingual/mon_latn_full/* models/full/mon_latn_full

# mos_latn
if test -f models/full/mos_latn_full/pytorch_model.bin; then
echo "Model already found: mos_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mos_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mos_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8637 \
--warmup_steps=863 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mos_latn_full.txt \
--seed=43 \
--override_n_examples=6910 \
--output_dir=models/full/mos_latn_full
cp tokenizers/monolingual/mos_latn_full/* models/full/mos_latn_full

# mri_latn
if test -f models/full/mri_latn_full/pytorch_model.bin; then
echo "Model already found: mri_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mri_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mri_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=83015 \
--warmup_steps=8301 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mri_latn_full.txt \
--seed=43 \
--override_n_examples=265648 \
--output_dir=models/full/mri_latn_full
cp tokenizers/monolingual/mri_latn_100mb/* models/full/mri_latn_full

# mrj_cyrl
if test -f models/full/mrj_cyrl_full/pytorch_model.bin; then
echo "Model already found: mrj_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mrj_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mrj_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8852 \
--warmup_steps=885 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mrj_cyrl_full.txt \
--seed=43 \
--override_n_examples=3541 \
--output_dir=models/full/mrj_cyrl_full
cp tokenizers/monolingual/mrj_cyrl_full/* models/full/mrj_cyrl_full

# mwl_latn
if test -f models/full/mwl_latn_full/pytorch_model.bin; then
echo "Model already found: mwl_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mwl_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mwl_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11200 \
--warmup_steps=1120 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mwl_latn_full.txt \
--seed=43 \
--override_n_examples=4480 \
--output_dir=models/full/mwl_latn_full
cp tokenizers/monolingual/mwl_latn_full/* models/full/mwl_latn_full

# mya_mymr
if test -f models/full/mya_mymr_full/pytorch_model.bin; then
echo "Model already found: mya_mymr_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mya_mymr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mya_mymr_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=192489 \
--warmup_steps=19248 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mya_mymr_full.txt \
--seed=43 \
--override_n_examples=615966 \
--output_dir=models/full/mya_mymr_full
cp tokenizers/monolingual/mya_mymr_100mb/* models/full/mya_mymr_full

# myv_cyrl
if test -f models/full/myv_cyrl_full/pytorch_model.bin; then
echo "Model already found: myv_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/myv_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/myv_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9403 \
--warmup_steps=940 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/myv_cyrl_full.txt \
--seed=43 \
--override_n_examples=7523 \
--output_dir=models/full/myv_cyrl_full
cp tokenizers/monolingual/myv_cyrl_full/* models/full/myv_cyrl_full

# nan_latn
if test -f models/full/nan_latn_full/pytorch_model.bin; then
echo "Model already found: nan_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nan_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nan_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=40586 \
--warmup_steps=4058 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nan_latn_full.txt \
--seed=43 \
--override_n_examples=32469 \
--output_dir=models/full/nan_latn_full
cp tokenizers/monolingual/nan_latn_full/* models/full/nan_latn_full

# nap_latn
if test -f models/full/nap_latn_full/pytorch_model.bin; then
echo "Model already found: nap_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nap_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nap_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10370 \
--warmup_steps=1037 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nap_latn_full.txt \
--seed=43 \
--override_n_examples=4148 \
--output_dir=models/full/nap_latn_full
cp tokenizers/monolingual/nap_latn_full/* models/full/nap_latn_full

# nde_latn
if test -f models/full/nde_latn_full/pytorch_model.bin; then
echo "Model already found: nde_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nde_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nde_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4313 \
--warmup_steps=431 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nde_latn_full.txt \
--seed=43 \
--override_n_examples=3451 \
--output_dir=models/full/nde_latn_full
cp tokenizers/monolingual/nde_latn_full/* models/full/nde_latn_full

# nds_latn
if test -f models/full/nds_latn_full/pytorch_model.bin; then
echo "Model already found: nds_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nds_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nds_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=49590 \
--warmup_steps=4959 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nds_latn_full.txt \
--seed=43 \
--override_n_examples=39672 \
--output_dir=models/full/nds_latn_full
cp tokenizers/monolingual/nds_latn_full/* models/full/nds_latn_full

# new_deva
if test -f models/full/new_deva_full/pytorch_model.bin; then
echo "Model already found: new_deva_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/new_deva_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/new_deva_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7147 \
--warmup_steps=714 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/new_deva_full.txt \
--seed=43 \
--override_n_examples=5718 \
--output_dir=models/full/new_deva_full
cp tokenizers/monolingual/new_deva_full/* models/full/new_deva_full

# ngu_latn
if test -f models/full/ngu_latn_full/pytorch_model.bin; then
echo "Model already found: ngu_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ngu_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ngu_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7367 \
--warmup_steps=736 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ngu_latn_full.txt \
--seed=43 \
--override_n_examples=2947 \
--output_dir=models/full/ngu_latn_full
cp tokenizers/monolingual/ngu_latn_full/* models/full/ngu_latn_full

# nhe_latn
if test -f models/full/nhe_latn_full/pytorch_model.bin; then
echo "Model already found: nhe_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nhe_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nhe_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6192 \
--warmup_steps=619 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nhe_latn_full.txt \
--seed=43 \
--override_n_examples=2477 \
--output_dir=models/full/nhe_latn_full
cp tokenizers/monolingual/nhe_latn_full/* models/full/nhe_latn_full

# nnb_latn
if test -f models/full/nnb_latn_full/pytorch_model.bin; then
echo "Model already found: nnb_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nnb_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nnb_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8615 \
--warmup_steps=861 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nnb_latn_full.txt \
--seed=43 \
--override_n_examples=3446 \
--output_dir=models/full/nnb_latn_full
cp tokenizers/monolingual/nnb_latn_full/* models/full/nnb_latn_full

# nno_latn
if test -f models/full/nno_latn_full/pytorch_model.bin; then
echo "Model already found: nno_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nno_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nno_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=70810 \
--warmup_steps=7081 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nno_latn_full.txt \
--seed=43 \
--override_n_examples=226594 \
--output_dir=models/full/nno_latn_full
cp tokenizers/monolingual/nno_latn_100mb/* models/full/nno_latn_full

# nso_latn
if test -f models/full/nso_latn_full/pytorch_model.bin; then
echo "Model already found: nso_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nso_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nso_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=42765 \
--warmup_steps=4276 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nso_latn_full.txt \
--seed=43 \
--override_n_examples=34212 \
--output_dir=models/full/nso_latn_full
cp tokenizers/monolingual/nso_latn_full/* models/full/nso_latn_full

# nya_latn
if test -f models/full/nya_latn_full/pytorch_model.bin; then
echo "Model already found: nya_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nya_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nya_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=68628 \
--warmup_steps=6862 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nya_latn_full.txt \
--seed=43 \
--override_n_examples=219611 \
--output_dir=models/full/nya_latn_full
cp tokenizers/monolingual/nya_latn_100mb/* models/full/nya_latn_full

# nzi_latn
if test -f models/full/nzi_latn_full/pytorch_model.bin; then
echo "Model already found: nzi_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nzi_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nzi_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7395 \
--warmup_steps=739 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nzi_latn_full.txt \
--seed=43 \
--override_n_examples=2958 \
--output_dir=models/full/nzi_latn_full
cp tokenizers/monolingual/nzi_latn_full/* models/full/nzi_latn_full

# oci_latn
if test -f models/full/oci_latn_full/pytorch_model.bin; then
echo "Model already found: oci_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/oci_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/oci_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=60903 \
--warmup_steps=6090 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/oci_latn_full.txt \
--seed=43 \
--override_n_examples=194890 \
--output_dir=models/full/oci_latn_full
cp tokenizers/monolingual/oci_latn_100mb/* models/full/oci_latn_full

# ori_orya
if test -f models/full/ori_orya_full/pytorch_model.bin; then
echo "Model already found: ori_orya_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ori_orya_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ori_orya_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=101030 \
--warmup_steps=10103 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ori_orya_full.txt \
--seed=43 \
--override_n_examples=323298 \
--output_dir=models/full/ori_orya_full
cp tokenizers/monolingual/ori_orya_100mb/* models/full/ori_orya_full

# orm_latn
if test -f models/full/orm_latn_full/pytorch_model.bin; then
echo "Model already found: orm_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/orm_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/orm_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=24257 \
--warmup_steps=2425 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/orm_latn_full.txt \
--seed=43 \
--override_n_examples=77623 \
--output_dir=models/full/orm_latn_full
cp tokenizers/monolingual/orm_latn_100mb/* models/full/orm_latn_full

# oss_cyrl
if test -f models/full/oss_cyrl_full/pytorch_model.bin; then
echo "Model already found: oss_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/oss_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/oss_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=34323 \
--warmup_steps=3432 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/oss_cyrl_full.txt \
--seed=43 \
--override_n_examples=27459 \
--output_dir=models/full/oss_cyrl_full
cp tokenizers/monolingual/oss_cyrl_full/* models/full/oss_cyrl_full

# otq_latn
if test -f models/full/otq_latn_full/pytorch_model.bin; then
echo "Model already found: otq_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/otq_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/otq_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=13922 \
--warmup_steps=1392 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/otq_latn_full.txt \
--seed=43 \
--override_n_examples=11138 \
--output_dir=models/full/otq_latn_full
cp tokenizers/monolingual/otq_latn_full/* models/full/otq_latn_full

# pag_latn
if test -f models/full/pag_latn_full/pytorch_model.bin; then
echo "Model already found: pag_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pag_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pag_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=25492 \
--warmup_steps=2549 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pag_latn_full.txt \
--seed=43 \
--override_n_examples=20394 \
--output_dir=models/full/pag_latn_full
cp tokenizers/monolingual/pag_latn_full/* models/full/pag_latn_full

# pam_latn
if test -f models/full/pam_latn_full/pytorch_model.bin; then
echo "Model already found: pam_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pam_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pam_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=27516 \
--warmup_steps=2751 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pam_latn_full.txt \
--seed=43 \
--override_n_examples=22013 \
--output_dir=models/full/pam_latn_full
cp tokenizers/monolingual/pam_latn_full/* models/full/pam_latn_full

# pap_latn
if test -f models/full/pap_latn_full/pytorch_model.bin; then
echo "Model already found: pap_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pap_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pap_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=36644 \
--warmup_steps=3664 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pap_latn_full.txt \
--seed=43 \
--override_n_examples=117261 \
--output_dir=models/full/pap_latn_full
cp tokenizers/monolingual/pap_latn_100mb/* models/full/pap_latn_full

# pbt_arab
if test -f models/full/pbt_arab_full/pytorch_model.bin; then
echo "Model already found: pbt_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pbt_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pbt_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=36992 \
--warmup_steps=3699 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pbt_arab_full.txt \
--seed=43 \
--override_n_examples=118375 \
--output_dir=models/full/pbt_arab_full
cp tokenizers/monolingual/pbt_arab_100mb/* models/full/pbt_arab_full

# pck_latn
if test -f models/full/pck_latn_full/pytorch_model.bin; then
echo "Model already found: pck_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pck_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pck_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10565 \
--warmup_steps=1056 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pck_latn_full.txt \
--seed=43 \
--override_n_examples=4226 \
--output_dir=models/full/pck_latn_full
cp tokenizers/monolingual/pck_latn_full/* models/full/pck_latn_full

# pcm_latn
if test -f models/full/pcm_latn_full/pytorch_model.bin; then
echo "Model already found: pcm_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pcm_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pcm_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12893 \
--warmup_steps=1289 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pcm_latn_full.txt \
--seed=43 \
--override_n_examples=10315 \
--output_dir=models/full/pcm_latn_full
cp tokenizers/monolingual/pcm_latn_full/* models/full/pcm_latn_full

# plt_latn
if test -f models/full/plt_latn_full/pytorch_model.bin; then
echo "Model already found: plt_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/plt_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/plt_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=59520 \
--warmup_steps=5952 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/plt_latn_full.txt \
--seed=43 \
--override_n_examples=190464 \
--output_dir=models/full/plt_latn_full
cp tokenizers/monolingual/plt_latn_100mb/* models/full/plt_latn_full

# pms_latn
if test -f models/full/pms_latn_full/pytorch_model.bin; then
echo "Model already found: pms_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pms_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pms_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12958 \
--warmup_steps=1295 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pms_latn_full.txt \
--seed=43 \
--override_n_examples=10367 \
--output_dir=models/full/pms_latn_full
cp tokenizers/monolingual/pms_latn_full/* models/full/pms_latn_full

# pnb_arab
if test -f models/full/pnb_arab_full/pytorch_model.bin; then
echo "Model already found: pnb_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pnb_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pnb_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=18377 \
--warmup_steps=1837 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pnb_arab_full.txt \
--seed=43 \
--override_n_examples=58809 \
--output_dir=models/full/pnb_arab_full
cp tokenizers/monolingual/pnb_arab_100mb/* models/full/pnb_arab_full

# pon_latn
if test -f models/full/pon_latn_full/pytorch_model.bin; then
echo "Model already found: pon_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pon_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pon_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6897 \
--warmup_steps=689 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pon_latn_full.txt \
--seed=43 \
--override_n_examples=2759 \
--output_dir=models/full/pon_latn_full
cp tokenizers/monolingual/pon_latn_full/* models/full/pon_latn_full

# prs_arab
if test -f models/full/prs_arab_full/pytorch_model.bin; then
echo "Model already found: prs_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/prs_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/prs_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=22918 \
--warmup_steps=2291 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/prs_arab_full.txt \
--seed=43 \
--override_n_examples=73339 \
--output_dir=models/full/prs_arab_full
cp tokenizers/monolingual/prs_arab_100mb/* models/full/prs_arab_full

# que_latn
if test -f models/full/que_latn_full/pytorch_model.bin; then
echo "Model already found: que_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/que_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/que_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=24777 \
--warmup_steps=2477 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/que_latn_full.txt \
--seed=43 \
--override_n_examples=79289 \
--output_dir=models/full/que_latn_full
cp tokenizers/monolingual/que_latn_100mb/* models/full/que_latn_full

# quy_latn
if test -f models/full/quy_latn_full/pytorch_model.bin; then
echo "Model already found: quy_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/quy_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/quy_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=21271 \
--warmup_steps=2127 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/quy_latn_full.txt \
--seed=43 \
--override_n_examples=68068 \
--output_dir=models/full/quy_latn_full
cp tokenizers/monolingual/quy_latn_100mb/* models/full/quy_latn_full

# quz_latn
if test -f models/full/quz_latn_full/pytorch_model.bin; then
echo "Model already found: quz_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/quz_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/quz_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10110 \
--warmup_steps=1011 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/quz_latn_full.txt \
--seed=43 \
--override_n_examples=4044 \
--output_dir=models/full/quz_latn_full
cp tokenizers/monolingual/quz_latn_full/* models/full/quz_latn_full

# rmc_latn
if test -f models/full/rmc_latn_full/pytorch_model.bin; then
echo "Model already found: rmc_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rmc_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rmc_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6062 \
--warmup_steps=606 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rmc_latn_full.txt \
--seed=43 \
--override_n_examples=2425 \
--output_dir=models/full/rmc_latn_full
cp tokenizers/monolingual/rmc_latn_full/* models/full/rmc_latn_full

# roh_latn
if test -f models/full/roh_latn_full/pytorch_model.bin; then
echo "Model already found: roh_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/roh_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/roh_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=57675 \
--warmup_steps=5767 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/roh_latn_full.txt \
--seed=43 \
--override_n_examples=46140 \
--output_dir=models/full/roh_latn_full
cp tokenizers/monolingual/roh_latn_full/* models/full/roh_latn_full

# rue_cyrl
if test -f models/full/rue_cyrl_full/pytorch_model.bin; then
echo "Model already found: rue_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rue_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rue_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5667 \
--warmup_steps=566 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rue_cyrl_full.txt \
--seed=43 \
--override_n_examples=2267 \
--output_dir=models/full/rue_cyrl_full
cp tokenizers/monolingual/rue_cyrl_full/* models/full/rue_cyrl_full

# run_latn
if test -f models/full/run_latn_full/pytorch_model.bin; then
echo "Model already found: run_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/run_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/run_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=57915 \
--warmup_steps=5791 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/run_latn_full.txt \
--seed=43 \
--override_n_examples=46332 \
--output_dir=models/full/run_latn_full
cp tokenizers/monolingual/run_latn_full/* models/full/run_latn_full

# rus_latn
if test -f models/full/rus_latn_full/pytorch_model.bin; then
echo "Model already found: rus_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rus_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rus_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=18002 \
--warmup_steps=1800 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rus_latn_full.txt \
--seed=43 \
--override_n_examples=14402 \
--output_dir=models/full/rus_latn_full
cp tokenizers/monolingual/rus_latn_full/* models/full/rus_latn_full

# sag_latn
if test -f models/full/sag_latn_full/pytorch_model.bin; then
echo "Model already found: sag_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sag_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sag_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12033 \
--warmup_steps=1203 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sag_latn_full.txt \
--seed=43 \
--override_n_examples=9627 \
--output_dir=models/full/sag_latn_full
cp tokenizers/monolingual/sag_latn_full/* models/full/sag_latn_full

# sah_cyrl
if test -f models/full/sah_cyrl_full/pytorch_model.bin; then
echo "Model already found: sah_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sah_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sah_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=28863 \
--warmup_steps=2886 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sah_cyrl_full.txt \
--seed=43 \
--override_n_examples=92362 \
--output_dir=models/full/sah_cyrl_full
cp tokenizers/monolingual/sah_cyrl_100mb/* models/full/sah_cyrl_full

# san_deva
if test -f models/full/san_deva_full/pytorch_model.bin; then
echo "Model already found: san_deva_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/san_deva_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/san_deva_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=19443 \
--warmup_steps=1944 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/san_deva_full.txt \
--seed=43 \
--override_n_examples=62219 \
--output_dir=models/full/san_deva_full
cp tokenizers/monolingual/san_deva_100mb/* models/full/san_deva_full

# san_latn
if test -f models/full/san_latn_full/pytorch_model.bin; then
echo "Model already found: san_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/san_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/san_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5687 \
--warmup_steps=568 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/san_latn_full.txt \
--seed=43 \
--override_n_examples=2275 \
--output_dir=models/full/san_latn_full
cp tokenizers/monolingual/san_latn_full/* models/full/san_latn_full

# sat_olck
if test -f models/full/sat_olck_full/pytorch_model.bin; then
echo "Model already found: sat_olck_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sat_olck_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sat_olck_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10860 \
--warmup_steps=1086 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sat_olck_full.txt \
--seed=43 \
--override_n_examples=4344 \
--output_dir=models/full/sat_olck_full
cp tokenizers/monolingual/sat_olck_full/* models/full/sat_olck_full

# scn_latn
if test -f models/full/scn_latn_full/pytorch_model.bin; then
echo "Model already found: scn_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/scn_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/scn_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=19537 \
--warmup_steps=1953 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/scn_latn_full.txt \
--seed=43 \
--override_n_examples=62521 \
--output_dir=models/full/scn_latn_full
cp tokenizers/monolingual/scn_latn_100mb/* models/full/scn_latn_full

# sco_latn
if test -f models/full/sco_latn_full/pytorch_model.bin; then
echo "Model already found: sco_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sco_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sco_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=30708 \
--warmup_steps=3070 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sco_latn_full.txt \
--seed=43 \
--override_n_examples=24567 \
--output_dir=models/full/sco_latn_full
cp tokenizers/monolingual/sco_latn_full/* models/full/sco_latn_full

# shn_mymr
if test -f models/full/shn_mymr_full/pytorch_model.bin; then
echo "Model already found: shn_mymr_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/shn_mymr_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/shn_mymr_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5466 \
--warmup_steps=546 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/shn_mymr_full.txt \
--seed=43 \
--override_n_examples=4373 \
--output_dir=models/full/shn_mymr_full
cp tokenizers/monolingual/shn_mymr_full/* models/full/shn_mymr_full

# sme_latn
if test -f models/full/sme_latn_full/pytorch_model.bin; then
echo "Model already found: sme_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sme_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sme_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=38581 \
--warmup_steps=3858 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sme_latn_full.txt \
--seed=43 \
--override_n_examples=30865 \
--output_dir=models/full/sme_latn_full
cp tokenizers/monolingual/sme_latn_full/* models/full/sme_latn_full

# smo_latn
if test -f models/full/smo_latn_full/pytorch_model.bin; then
echo "Model already found: smo_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/smo_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/smo_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=62200 \
--warmup_steps=6220 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/smo_latn_full.txt \
--seed=43 \
--override_n_examples=199043 \
--output_dir=models/full/smo_latn_full
cp tokenizers/monolingual/smo_latn_100mb/* models/full/smo_latn_full

# sna_latn
if test -f models/full/sna_latn_full/pytorch_model.bin; then
echo "Model already found: sna_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sna_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sna_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=92597 \
--warmup_steps=9259 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sna_latn_full.txt \
--seed=43 \
--override_n_examples=296313 \
--output_dir=models/full/sna_latn_full
cp tokenizers/monolingual/sna_latn_100mb/* models/full/sna_latn_full

# snd_arab
if test -f models/full/snd_arab_full/pytorch_model.bin; then
echo "Model already found: snd_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/snd_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/snd_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=69962 \
--warmup_steps=6996 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/snd_arab_full.txt \
--seed=43 \
--override_n_examples=223879 \
--output_dir=models/full/snd_arab_full
cp tokenizers/monolingual/snd_arab_100mb/* models/full/snd_arab_full

# sot_latn
if test -f models/full/sot_latn_full/pytorch_model.bin; then
echo "Model already found: sot_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sot_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sot_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=57460 \
--warmup_steps=5746 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sot_latn_full.txt \
--seed=43 \
--override_n_examples=183875 \
--output_dir=models/full/sot_latn_full
cp tokenizers/monolingual/sot_latn_100mb/* models/full/sot_latn_full

# srd_latn
if test -f models/full/srd_latn_full/pytorch_model.bin; then
echo "Model already found: srd_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srd_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srd_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=16685 \
--warmup_steps=1668 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srd_latn_full.txt \
--seed=43 \
--override_n_examples=13348 \
--output_dir=models/full/srd_latn_full
cp tokenizers/monolingual/srd_latn_full/* models/full/srd_latn_full

# srn_latn
if test -f models/full/srn_latn_full/pytorch_model.bin; then
echo "Model already found: srn_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srn_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srn_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7563 \
--warmup_steps=756 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srn_latn_full.txt \
--seed=43 \
--override_n_examples=6051 \
--output_dir=models/full/srn_latn_full
cp tokenizers/monolingual/srn_latn_full/* models/full/srn_latn_full

# ssw_latn
if test -f models/full/ssw_latn_full/pytorch_model.bin; then
echo "Model already found: ssw_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ssw_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ssw_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=13591 \
--warmup_steps=1359 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ssw_latn_full.txt \
--seed=43 \
--override_n_examples=10873 \
--output_dir=models/full/ssw_latn_full
cp tokenizers/monolingual/ssw_latn_full/* models/full/ssw_latn_full

# sun_latn
if test -f models/full/sun_latn_full/pytorch_model.bin; then
echo "Model already found: sun_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sun_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sun_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=86832 \
--warmup_steps=8683 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sun_latn_full.txt \
--seed=43 \
--override_n_examples=277864 \
--output_dir=models/full/sun_latn_full
cp tokenizers/monolingual/sun_latn_100mb/* models/full/sun_latn_full

# syr_syrc
if test -f models/full/syr_syrc_full/pytorch_model.bin; then
echo "Model already found: syr_syrc_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/syr_syrc_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/syr_syrc_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6448 \
--warmup_steps=644 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/syr_syrc_full.txt \
--seed=43 \
--override_n_examples=5159 \
--output_dir=models/full/syr_syrc_full
cp tokenizers/monolingual/syr_syrc_full/* models/full/syr_syrc_full

# szl_latn
if test -f models/full/szl_latn_full/pytorch_model.bin; then
echo "Model already found: szl_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/szl_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/szl_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=18538 \
--warmup_steps=1853 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/szl_latn_full.txt \
--seed=43 \
--override_n_examples=14831 \
--output_dir=models/full/szl_latn_full
cp tokenizers/monolingual/szl_latn_full/* models/full/szl_latn_full

# tam_latn
if test -f models/full/tam_latn_full/pytorch_model.bin; then
echo "Model already found: tam_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tam_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tam_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11040 \
--warmup_steps=1104 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tam_latn_full.txt \
--seed=43 \
--override_n_examples=4416 \
--output_dir=models/full/tam_latn_full
cp tokenizers/monolingual/tam_latn_full/* models/full/tam_latn_full

# tbz_latn
if test -f models/full/tbz_latn_full/pytorch_model.bin; then
echo "Model already found: tbz_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tbz_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tbz_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=9125 \
--warmup_steps=912 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tbz_latn_full.txt \
--seed=43 \
--override_n_examples=3650 \
--output_dir=models/full/tbz_latn_full
cp tokenizers/monolingual/tbz_latn_full/* models/full/tbz_latn_full

# tcy_knda
if test -f models/full/tcy_knda_full/pytorch_model.bin; then
echo "Model already found: tcy_knda_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tcy_knda_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tcy_knda_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5910 \
--warmup_steps=591 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tcy_knda_full.txt \
--seed=43 \
--override_n_examples=2364 \
--output_dir=models/full/tcy_knda_full
cp tokenizers/monolingual/tcy_knda_full/* models/full/tcy_knda_full

# tdx_latn
if test -f models/full/tdx_latn_full/pytorch_model.bin; then
echo "Model already found: tdx_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tdx_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tdx_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6365 \
--warmup_steps=636 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tdx_latn_full.txt \
--seed=43 \
--override_n_examples=2546 \
--output_dir=models/full/tdx_latn_full
cp tokenizers/monolingual/tdx_latn_full/* models/full/tdx_latn_full

# tel_latn
if test -f models/full/tel_latn_full/pytorch_model.bin; then
echo "Model already found: tel_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tel_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tel_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12857 \
--warmup_steps=1285 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tel_latn_full.txt \
--seed=43 \
--override_n_examples=10286 \
--output_dir=models/full/tel_latn_full
cp tokenizers/monolingual/tel_latn_full/* models/full/tel_latn_full

# tet_latn
if test -f models/full/tet_latn_full/pytorch_model.bin; then
echo "Model already found: tet_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tet_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tet_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=68438 \
--warmup_steps=6843 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tet_latn_full.txt \
--seed=43 \
--override_n_examples=54751 \
--output_dir=models/full/tet_latn_full
cp tokenizers/monolingual/tet_latn_full/* models/full/tet_latn_full

# tir_ethi
if test -f models/full/tir_ethi_full/pytorch_model.bin; then
echo "Model already found: tir_ethi_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tir_ethi_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tir_ethi_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=34494 \
--warmup_steps=3449 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tir_ethi_full.txt \
--seed=43 \
--override_n_examples=110381 \
--output_dir=models/full/tir_ethi_full
cp tokenizers/monolingual/tir_ethi_100mb/* models/full/tir_ethi_full

# tiv_latn
if test -f models/full/tiv_latn_full/pytorch_model.bin; then
echo "Model already found: tiv_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tiv_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tiv_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10347 \
--warmup_steps=1034 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tiv_latn_full.txt \
--seed=43 \
--override_n_examples=4139 \
--output_dir=models/full/tiv_latn_full
cp tokenizers/monolingual/tiv_latn_full/* models/full/tiv_latn_full

# tlh_latn
if test -f models/full/tlh_latn_full/pytorch_model.bin; then
echo "Model already found: tlh_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tlh_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tlh_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8502 \
--warmup_steps=850 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tlh_latn_full.txt \
--seed=43 \
--override_n_examples=3401 \
--output_dir=models/full/tlh_latn_full
cp tokenizers/monolingual/tlh_latn_full/* models/full/tlh_latn_full

# ton_latn
if test -f models/full/ton_latn_full/pytorch_model.bin; then
echo "Model already found: ton_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ton_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ton_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=15227 \
--warmup_steps=1522 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ton_latn_full.txt \
--seed=43 \
--override_n_examples=12182 \
--output_dir=models/full/ton_latn_full
cp tokenizers/monolingual/ton_latn_full/* models/full/ton_latn_full

# tpi_latn
if test -f models/full/tpi_latn_full/pytorch_model.bin; then
echo "Model already found: tpi_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tpi_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tpi_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12457 \
--warmup_steps=1245 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tpi_latn_full.txt \
--seed=43 \
--override_n_examples=9966 \
--output_dir=models/full/tpi_latn_full
cp tokenizers/monolingual/tpi_latn_full/* models/full/tpi_latn_full

# tsn_latn
if test -f models/full/tsn_latn_full/pytorch_model.bin; then
echo "Model already found: tsn_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tsn_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tsn_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=19219 \
--warmup_steps=1921 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tsn_latn_full.txt \
--seed=43 \
--override_n_examples=61501 \
--output_dir=models/full/tsn_latn_full
cp tokenizers/monolingual/tsn_latn_100mb/* models/full/tsn_latn_full

# tso_latn
if test -f models/full/tso_latn_full/pytorch_model.bin; then
echo "Model already found: tso_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tso_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tso_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=52940 \
--warmup_steps=5294 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tso_latn_full.txt \
--seed=43 \
--override_n_examples=42352 \
--output_dir=models/full/tso_latn_full
cp tokenizers/monolingual/tso_latn_full/* models/full/tso_latn_full

# tuk_latn
if test -f models/full/tuk_latn_full/pytorch_model.bin; then
echo "Model already found: tuk_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tuk_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tuk_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=34913 \
--warmup_steps=3491 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tuk_latn_full.txt \
--seed=43 \
--override_n_examples=111722 \
--output_dir=models/full/tuk_latn_full
cp tokenizers/monolingual/tuk_latn_100mb/* models/full/tuk_latn_full

# tum_latn
if test -f models/full/tum_latn_full/pytorch_model.bin; then
echo "Model already found: tum_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tum_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tum_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=24030 \
--warmup_steps=2403 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tum_latn_full.txt \
--seed=43 \
--override_n_examples=19224 \
--output_dir=models/full/tum_latn_full
cp tokenizers/monolingual/tum_latn_full/* models/full/tum_latn_full

# twi_latn
if test -f models/full/twi_latn_full/pytorch_model.bin; then
echo "Model already found: twi_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/twi_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/twi_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=46143 \
--warmup_steps=4614 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/twi_latn_full.txt \
--seed=43 \
--override_n_examples=36915 \
--output_dir=models/full/twi_latn_full
cp tokenizers/monolingual/twi_latn_full/* models/full/twi_latn_full

# tyv_cyrl
if test -f models/full/tyv_cyrl_full/pytorch_model.bin; then
echo "Model already found: tyv_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tyv_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tyv_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=38028 \
--warmup_steps=3802 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tyv_cyrl_full.txt \
--seed=43 \
--override_n_examples=30423 \
--output_dir=models/full/tyv_cyrl_full
cp tokenizers/monolingual/tyv_cyrl_full/* models/full/tyv_cyrl_full

# tzo_latn
if test -f models/full/tzo_latn_full/pytorch_model.bin; then
echo "Model already found: tzo_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tzo_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tzo_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=16912 \
--warmup_steps=1691 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tzo_latn_full.txt \
--seed=43 \
--override_n_examples=6765 \
--output_dir=models/full/tzo_latn_full
cp tokenizers/monolingual/tzo_latn_full/* models/full/tzo_latn_full

# udm_cyrl
if test -f models/full/udm_cyrl_full/pytorch_model.bin; then
echo "Model already found: udm_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/udm_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/udm_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=26691 \
--warmup_steps=2669 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/udm_cyrl_full.txt \
--seed=43 \
--override_n_examples=21353 \
--output_dir=models/full/udm_cyrl_full
cp tokenizers/monolingual/udm_cyrl_full/* models/full/udm_cyrl_full

# uig_arab
if test -f models/full/uig_arab_full/pytorch_model.bin; then
echo "Model already found: uig_arab_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uig_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uig_arab_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=63500 \
--warmup_steps=6350 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uig_arab_full.txt \
--seed=43 \
--override_n_examples=203203 \
--output_dir=models/full/uig_arab_full
cp tokenizers/monolingual/uig_arab_100mb/* models/full/uig_arab_full

# uig_latn
if test -f models/full/uig_latn_full/pytorch_model.bin; then
echo "Model already found: uig_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uig_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uig_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8120 \
--warmup_steps=812 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uig_latn_full.txt \
--seed=43 \
--override_n_examples=3248 \
--output_dir=models/full/uig_latn_full
cp tokenizers/monolingual/uig_latn_full/* models/full/uig_latn_full

# umb_latn
if test -f models/full/umb_latn_full/pytorch_model.bin; then
echo "Model already found: umb_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/umb_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/umb_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11580 \
--warmup_steps=1158 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/umb_latn_full.txt \
--seed=43 \
--override_n_examples=9264 \
--output_dir=models/full/umb_latn_full
cp tokenizers/monolingual/umb_latn_full/* models/full/umb_latn_full

# uzb_cyrl
if test -f models/full/uzb_cyrl_full/pytorch_model.bin; then
echo "Model already found: uzb_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzb_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzb_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=67669 \
--warmup_steps=6766 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzb_cyrl_full.txt \
--seed=43 \
--override_n_examples=216541 \
--output_dir=models/full/uzb_cyrl_full
cp tokenizers/monolingual/uzb_cyrl_100mb/* models/full/uzb_cyrl_full

# uzn_cyrl
if test -f models/full/uzn_cyrl_full/pytorch_model.bin; then
echo "Model already found: uzn_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzn_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzn_cyrl_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=27441 \
--warmup_steps=2744 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzn_cyrl_full.txt \
--seed=43 \
--override_n_examples=87814 \
--output_dir=models/full/uzn_cyrl_full
cp tokenizers/monolingual/uzn_cyrl_100mb/* models/full/uzn_cyrl_full

# uzn_latn
if test -f models/full/uzn_latn_full/pytorch_model.bin; then
echo "Model already found: uzn_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzn_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzn_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=31768 \
--warmup_steps=3176 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzn_latn_full.txt \
--seed=43 \
--override_n_examples=101659 \
--output_dir=models/full/uzn_latn_full
cp tokenizers/monolingual/uzn_latn_100mb/* models/full/uzn_latn_full

# vec_latn
if test -f models/full/vec_latn_full/pytorch_model.bin; then
echo "Model already found: vec_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vec_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vec_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=24733 \
--warmup_steps=2473 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vec_latn_full.txt \
--seed=43 \
--override_n_examples=79148 \
--output_dir=models/full/vec_latn_full
cp tokenizers/monolingual/vec_latn_100mb/* models/full/vec_latn_full

# ven_latn
if test -f models/full/ven_latn_full/pytorch_model.bin; then
echo "Model already found: ven_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ven_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ven_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7980 \
--warmup_steps=798 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ven_latn_full.txt \
--seed=43 \
--override_n_examples=6384 \
--output_dir=models/full/ven_latn_full
cp tokenizers/monolingual/ven_latn_full/* models/full/ven_latn_full

# vep_latn
if test -f models/full/vep_latn_full/pytorch_model.bin; then
echo "Model already found: vep_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vep_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vep_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8552 \
--warmup_steps=855 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vep_latn_full.txt \
--seed=43 \
--override_n_examples=3421 \
--output_dir=models/full/vep_latn_full
cp tokenizers/monolingual/vep_latn_full/* models/full/vep_latn_full

# vls_latn
if test -f models/full/vls_latn_full/pytorch_model.bin; then
echo "Model already found: vls_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vls_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vls_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=12132 \
--warmup_steps=1213 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vls_latn_full.txt \
--seed=43 \
--override_n_examples=4853 \
--output_dir=models/full/vls_latn_full
cp tokenizers/monolingual/vls_latn_full/* models/full/vls_latn_full

# vol_latn
if test -f models/full/vol_latn_full/pytorch_model.bin; then
echo "Model already found: vol_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vol_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vol_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=14722 \
--warmup_steps=1472 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vol_latn_full.txt \
--seed=43 \
--override_n_examples=11778 \
--output_dir=models/full/vol_latn_full
cp tokenizers/monolingual/vol_latn_full/* models/full/vol_latn_full

# war_latn
if test -f models/full/war_latn_full/pytorch_model.bin; then
echo "Model already found: war_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/war_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/war_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=29906 \
--warmup_steps=2990 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/war_latn_full.txt \
--seed=43 \
--override_n_examples=95701 \
--output_dir=models/full/war_latn_full
cp tokenizers/monolingual/war_latn_100mb/* models/full/war_latn_full

# wln_latn
if test -f models/full/wln_latn_full/pytorch_model.bin; then
echo "Model already found: wln_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wln_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wln_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=17755 \
--warmup_steps=1775 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wln_latn_full.txt \
--seed=43 \
--override_n_examples=56819 \
--output_dir=models/full/wln_latn_full
cp tokenizers/monolingual/wln_latn_100mb/* models/full/wln_latn_full

# wol_latn
if test -f models/full/wol_latn_full/pytorch_model.bin; then
echo "Model already found: wol_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wol_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wol_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=29311 \
--warmup_steps=2931 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wol_latn_full.txt \
--seed=43 \
--override_n_examples=23449 \
--output_dir=models/full/wol_latn_full
cp tokenizers/monolingual/wol_latn_full/* models/full/wol_latn_full

# wuu_hani
if test -f models/full/wuu_hani_full/pytorch_model.bin; then
echo "Model already found: wuu_hani_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wuu_hani_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wuu_hani_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=10040 \
--warmup_steps=1004 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wuu_hani_full.txt \
--seed=43 \
--override_n_examples=8032 \
--output_dir=models/full/wuu_hani_full
cp tokenizers/monolingual/wuu_hani_full/* models/full/wuu_hani_full

# xal_cyrl
if test -f models/full/xal_cyrl_full/pytorch_model.bin; then
echo "Model already found: xal_cyrl_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/xal_cyrl_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/xal_cyrl_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7197 \
--warmup_steps=719 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/xal_cyrl_full.txt \
--seed=43 \
--override_n_examples=2879 \
--output_dir=models/full/xal_cyrl_full
cp tokenizers/monolingual/xal_cyrl_full/* models/full/xal_cyrl_full

# xho_latn
if test -f models/full/xho_latn_full/pytorch_model.bin; then
echo "Model already found: xho_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/xho_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/xho_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=78055 \
--warmup_steps=7805 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/xho_latn_full.txt \
--seed=43 \
--override_n_examples=249777 \
--output_dir=models/full/xho_latn_full
cp tokenizers/monolingual/xho_latn_100mb/* models/full/xho_latn_full

# xmf_geor
if test -f models/full/xmf_geor_full/pytorch_model.bin; then
echo "Model already found: xmf_geor_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/xmf_geor_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/xmf_geor_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6675 \
--warmup_steps=667 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/xmf_geor_full.txt \
--seed=43 \
--override_n_examples=2670 \
--output_dir=models/full/xmf_geor_full
cp tokenizers/monolingual/xmf_geor_full/* models/full/xmf_geor_full

# ydd_hebr
if test -f models/full/ydd_hebr_full/pytorch_model.bin; then
echo "Model already found: ydd_hebr_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ydd_hebr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ydd_hebr_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=17276 \
--warmup_steps=1727 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ydd_hebr_full.txt \
--seed=43 \
--override_n_examples=55286 \
--output_dir=models/full/ydd_hebr_full
cp tokenizers/monolingual/ydd_hebr_100mb/* models/full/ydd_hebr_full

# yid_hebr
if test -f models/full/yid_hebr_full/pytorch_model.bin; then
echo "Model already found: yid_hebr_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yid_hebr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yid_hebr_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=52304 \
--warmup_steps=5230 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yid_hebr_full.txt \
--seed=43 \
--override_n_examples=167374 \
--output_dir=models/full/yid_hebr_full
cp tokenizers/monolingual/yid_hebr_100mb/* models/full/yid_hebr_full

# yor_latn
if test -f models/full/yor_latn_full/pytorch_model.bin; then
echo "Model already found: yor_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yor_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yor_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=95110 \
--warmup_steps=9511 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yor_latn_full.txt \
--seed=43 \
--override_n_examples=304354 \
--output_dir=models/full/yor_latn_full
cp tokenizers/monolingual/yor_latn_100mb/* models/full/yor_latn_full

# yua_latn
if test -f models/full/yua_latn_full/pytorch_model.bin; then
echo "Model already found: yua_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yua_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yua_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8900 \
--warmup_steps=890 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yua_latn_full.txt \
--seed=43 \
--override_n_examples=7120 \
--output_dir=models/full/yua_latn_full
cp tokenizers/monolingual/yua_latn_full/* models/full/yua_latn_full

# yue_hant
if test -f models/full/yue_hant_full/pytorch_model.bin; then
echo "Model already found: yue_hant_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yue_hant_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yue_hant_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=39270 \
--warmup_steps=3927 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yue_hant_full.txt \
--seed=43 \
--override_n_examples=31416 \
--output_dir=models/full/yue_hant_full
cp tokenizers/monolingual/yue_hant_full/* models/full/yue_hant_full

# zap_latn
if test -f models/full/zap_latn_full/pytorch_model.bin; then
echo "Model already found: zap_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zap_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zap_latn_full_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=11695 \
--warmup_steps=1169 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zap_latn_full.txt \
--seed=43 \
--override_n_examples=4678 \
--output_dir=models/full/zap_latn_full
cp tokenizers/monolingual/zap_latn_full/* models/full/zap_latn_full

# zho_hant
if test -f models/full/zho_hant_full/pytorch_model.bin; then
echo "Model already found: zho_hant_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zho_hant_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zho_hant_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=26057 \
--warmup_steps=2605 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zho_hant_full.txt \
--seed=43 \
--override_n_examples=83383 \
--output_dir=models/full/zho_hant_full
cp tokenizers/monolingual/zho_hant_100mb/* models/full/zho_hant_full

# zsm_latn
if test -f models/full/zsm_latn_full/pytorch_model.bin; then
echo "Model already found: zsm_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zsm_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zsm_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=113482 \
--warmup_steps=11348 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zsm_latn_full.txt \
--seed=43 \
--override_n_examples=363144 \
--output_dir=models/full/zsm_latn_full
cp tokenizers/monolingual/zsm_latn_100mb/* models/full/zsm_latn_full

# zul_latn
if test -f models/full/zul_latn_full/pytorch_model.bin; then
echo "Model already found: zul_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zul_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zul_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=122049 \
--warmup_steps=12204 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zul_latn_full.txt \
--seed=43 \
--override_n_examples=390558 \
--output_dir=models/full/zul_latn_full
cp tokenizers/monolingual/zul_latn_100mb/* models/full/zul_latn_full

# zza_latn
if test -f models/full/zza_latn_full/pytorch_model.bin; then
echo "Model already found: zza_latn_full."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zza_latn_full \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zza_latn_full_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=36165 \
--warmup_steps=3616 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zza_latn_full.txt \
--seed=43 \
--override_n_examples=28932 \
--output_dir=models/full/zza_latn_full
cp tokenizers/monolingual/zza_latn_full/* models/full/zza_latn_full
