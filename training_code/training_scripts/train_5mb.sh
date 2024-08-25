export CUDA_VISIBLE_DEVICES=0

# abk_cyrl
if test -f models/5mb/abk_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: abk_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/abk_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/abk_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=322 --save_steps=999999999 \
--max_steps=6447 \
--warmup_steps=644 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/abk_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2579 \
--output_dir=models/5mb/abk_cyrl_5mb
cp tokenizers/monolingual/abk_cyrl_5mb/* models/5mb/abk_cyrl_5mb

# ace_latn
if test -f models/5mb/ace_latn_5mb/pytorch_model.bin; then
echo "Model already found: ace_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ace_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ace_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=389 --save_steps=999999999 \
--max_steps=7787 \
--warmup_steps=778 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ace_latn_5mb.txt \
--seed=43 \
--override_n_examples=3115 \
--output_dir=models/5mb/ace_latn_5mb
cp tokenizers/monolingual/ace_latn_5mb/* models/5mb/ace_latn_5mb

# ady_cyrl
if test -f models/5mb/ady_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: ady_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ady_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ady_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=340 --save_steps=999999999 \
--max_steps=6812 \
--warmup_steps=681 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ady_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2725 \
--output_dir=models/5mb/ady_cyrl_5mb
cp tokenizers/monolingual/ady_cyrl_5mb/* models/5mb/ady_cyrl_5mb

# afb_arab
if test -f models/5mb/afb_arab_5mb/pytorch_model.bin; then
echo "Model already found: afb_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/afb_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/afb_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=289 --save_steps=999999999 \
--max_steps=5787 \
--warmup_steps=578 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/afb_arab_5mb.txt \
--seed=43 \
--override_n_examples=2315 \
--output_dir=models/5mb/afb_arab_5mb
cp tokenizers/monolingual/afb_arab_5mb/* models/5mb/afb_arab_5mb

# afr_latn
if test -f models/5mb/afr_latn_5mb/pytorch_model.bin; then
echo "Model already found: afr_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/afr_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/afr_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=305 --save_steps=999999999 \
--max_steps=6107 \
--warmup_steps=610 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/afr_latn_5mb.txt \
--seed=43 \
--override_n_examples=2443 \
--output_dir=models/5mb/afr_latn_5mb
cp tokenizers/monolingual/afr_latn_5mb/* models/5mb/afr_latn_5mb

# aka_latn
if test -f models/5mb/aka_latn_5mb/pytorch_model.bin; then
echo "Model already found: aka_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aka_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aka_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=574 --save_steps=999999999 \
--max_steps=11492 \
--warmup_steps=1149 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aka_latn_5mb.txt \
--seed=43 \
--override_n_examples=4597 \
--output_dir=models/5mb/aka_latn_5mb
cp tokenizers/monolingual/aka_latn_5mb/* models/5mb/aka_latn_5mb

# als_latn
if test -f models/5mb/als_latn_5mb/pytorch_model.bin; then
echo "Model already found: als_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/als_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/als_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=336 --save_steps=999999999 \
--max_steps=6720 \
--warmup_steps=672 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/als_latn_5mb.txt \
--seed=43 \
--override_n_examples=2688 \
--output_dir=models/5mb/als_latn_5mb
cp tokenizers/monolingual/als_latn_5mb/* models/5mb/als_latn_5mb

# alt_cyrl
if test -f models/5mb/alt_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: alt_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/alt_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/alt_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=291 --save_steps=999999999 \
--max_steps=5825 \
--warmup_steps=582 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/alt_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2330 \
--output_dir=models/5mb/alt_cyrl_5mb
cp tokenizers/monolingual/alt_cyrl_5mb/* models/5mb/alt_cyrl_5mb

# amh_ethi
if test -f models/5mb/amh_ethi_5mb/pytorch_model.bin; then
echo "Model already found: amh_ethi_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/amh_ethi_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/amh_ethi_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=277 --save_steps=999999999 \
--max_steps=5547 \
--warmup_steps=554 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/amh_ethi_5mb.txt \
--seed=43 \
--override_n_examples=2219 \
--output_dir=models/5mb/amh_ethi_5mb
cp tokenizers/monolingual/amh_ethi_5mb/* models/5mb/amh_ethi_5mb

# ang_latn
if test -f models/5mb/ang_latn_5mb/pytorch_model.bin; then
echo "Model already found: ang_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ang_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ang_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7562 \
--warmup_steps=756 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ang_latn_5mb.txt \
--seed=43 \
--override_n_examples=3025 \
--output_dir=models/5mb/ang_latn_5mb
cp tokenizers/monolingual/ang_latn_5mb/* models/5mb/ang_latn_5mb

# apc_arab
if test -f models/5mb/apc_arab_5mb/pytorch_model.bin; then
echo "Model already found: apc_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/apc_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/apc_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5952 \
--warmup_steps=595 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/apc_arab_5mb.txt \
--seed=43 \
--override_n_examples=2381 \
--output_dir=models/5mb/apc_arab_5mb
cp tokenizers/monolingual/apc_arab_5mb/* models/5mb/apc_arab_5mb

# arb_arab
if test -f models/5mb/arb_arab_5mb/pytorch_model.bin; then
echo "Model already found: arb_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arb_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arb_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=256 --save_steps=999999999 \
--max_steps=5130 \
--warmup_steps=513 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arb_arab_5mb.txt \
--seed=43 \
--override_n_examples=2052 \
--output_dir=models/5mb/arb_arab_5mb
cp tokenizers/monolingual/arb_arab_5mb/* models/5mb/arb_arab_5mb

# arg_latn
if test -f models/5mb/arg_latn_5mb/pytorch_model.bin; then
echo "Model already found: arg_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arg_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arg_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=370 --save_steps=999999999 \
--max_steps=7400 \
--warmup_steps=740 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arg_latn_5mb.txt \
--seed=43 \
--override_n_examples=2960 \
--output_dir=models/5mb/arg_latn_5mb
cp tokenizers/monolingual/arg_latn_5mb/* models/5mb/arg_latn_5mb

# arz_arab
if test -f models/5mb/arz_arab_5mb/pytorch_model.bin; then
echo "Model already found: arz_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arz_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arz_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=332 --save_steps=999999999 \
--max_steps=6642 \
--warmup_steps=664 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arz_arab_5mb.txt \
--seed=43 \
--override_n_examples=2657 \
--output_dir=models/5mb/arz_arab_5mb
cp tokenizers/monolingual/arz_arab_5mb/* models/5mb/arz_arab_5mb

# asm_beng
if test -f models/5mb/asm_beng_5mb/pytorch_model.bin; then
echo "Model already found: asm_beng_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/asm_beng_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/asm_beng_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=284 --save_steps=999999999 \
--max_steps=5695 \
--warmup_steps=569 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/asm_beng_5mb.txt \
--seed=43 \
--override_n_examples=2278 \
--output_dir=models/5mb/asm_beng_5mb
cp tokenizers/monolingual/asm_beng_5mb/* models/5mb/asm_beng_5mb

# ast_latn
if test -f models/5mb/ast_latn_5mb/pytorch_model.bin; then
echo "Model already found: ast_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ast_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ast_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=525 --save_steps=999999999 \
--max_steps=10500 \
--warmup_steps=1050 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ast_latn_5mb.txt \
--seed=43 \
--override_n_examples=4200 \
--output_dir=models/5mb/ast_latn_5mb
cp tokenizers/monolingual/ast_latn_5mb/* models/5mb/ast_latn_5mb

# ava_cyrl
if test -f models/5mb/ava_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: ava_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ava_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ava_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=327 --save_steps=999999999 \
--max_steps=6545 \
--warmup_steps=654 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ava_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2618 \
--output_dir=models/5mb/ava_cyrl_5mb
cp tokenizers/monolingual/ava_cyrl_5mb/* models/5mb/ava_cyrl_5mb

# aym_latn
if test -f models/5mb/aym_latn_5mb/pytorch_model.bin; then
echo "Model already found: aym_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aym_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aym_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=381 --save_steps=999999999 \
--max_steps=7622 \
--warmup_steps=762 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aym_latn_5mb.txt \
--seed=43 \
--override_n_examples=3049 \
--output_dir=models/5mb/aym_latn_5mb
cp tokenizers/monolingual/aym_latn_5mb/* models/5mb/aym_latn_5mb

# ayr_latn
if test -f models/5mb/ayr_latn_5mb/pytorch_model.bin; then
echo "Model already found: ayr_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ayr_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ayr_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=343 --save_steps=999999999 \
--max_steps=6867 \
--warmup_steps=686 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ayr_latn_5mb.txt \
--seed=43 \
--override_n_examples=2747 \
--output_dir=models/5mb/ayr_latn_5mb
cp tokenizers/monolingual/ayr_latn_5mb/* models/5mb/ayr_latn_5mb

# azb_arab
if test -f models/5mb/azb_arab_5mb/pytorch_model.bin; then
echo "Model already found: azb_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/azb_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/azb_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=320 --save_steps=999999999 \
--max_steps=6405 \
--warmup_steps=640 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/azb_arab_5mb.txt \
--seed=43 \
--override_n_examples=2562 \
--output_dir=models/5mb/azb_arab_5mb
cp tokenizers/monolingual/azb_arab_5mb/* models/5mb/azb_arab_5mb

# aze_arab
if test -f models/5mb/aze_arab_5mb/pytorch_model.bin; then
echo "Model already found: aze_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=275 --save_steps=999999999 \
--max_steps=5500 \
--warmup_steps=550 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_arab_5mb.txt \
--seed=43 \
--override_n_examples=2200 \
--output_dir=models/5mb/aze_arab_5mb
cp tokenizers/monolingual/aze_arab_5mb/* models/5mb/aze_arab_5mb

# aze_cyrl
if test -f models/5mb/aze_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: aze_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=268 --save_steps=999999999 \
--max_steps=5372 \
--warmup_steps=537 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2149 \
--output_dir=models/5mb/aze_cyrl_5mb
cp tokenizers/monolingual/aze_cyrl_5mb/* models/5mb/aze_cyrl_5mb

# aze_latn
if test -f models/5mb/aze_latn_5mb/pytorch_model.bin; then
echo "Model already found: aze_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=300 --save_steps=999999999 \
--max_steps=6005 \
--warmup_steps=600 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_latn_5mb.txt \
--seed=43 \
--override_n_examples=2402 \
--output_dir=models/5mb/aze_latn_5mb
cp tokenizers/monolingual/aze_latn_5mb/* models/5mb/aze_latn_5mb

# azj_latn
if test -f models/5mb/azj_latn_5mb/pytorch_model.bin; then
echo "Model already found: azj_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/azj_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/azj_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=234 --save_steps=999999999 \
--max_steps=4685 \
--warmup_steps=468 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/azj_latn_5mb.txt \
--seed=43 \
--override_n_examples=1874 \
--output_dir=models/5mb/azj_latn_5mb
cp tokenizers/monolingual/azj_latn_5mb/* models/5mb/azj_latn_5mb

# bak_cyrl
if test -f models/5mb/bak_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: bak_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bak_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bak_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=379 --save_steps=999999999 \
--max_steps=7592 \
--warmup_steps=759 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bak_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=3037 \
--output_dir=models/5mb/bak_cyrl_5mb
cp tokenizers/monolingual/bak_cyrl_5mb/* models/5mb/bak_cyrl_5mb

# bak_latn
if test -f models/5mb/bak_latn_5mb/pytorch_model.bin; then
echo "Model already found: bak_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bak_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bak_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5912 \
--warmup_steps=591 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bak_latn_5mb.txt \
--seed=43 \
--override_n_examples=2365 \
--output_dir=models/5mb/bak_latn_5mb
cp tokenizers/monolingual/bak_latn_5mb/* models/5mb/bak_latn_5mb

# bam_latn
if test -f models/5mb/bam_latn_5mb/pytorch_model.bin; then
echo "Model already found: bam_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bam_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bam_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=437 --save_steps=999999999 \
--max_steps=8752 \
--warmup_steps=875 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bam_latn_5mb.txt \
--seed=43 \
--override_n_examples=3501 \
--output_dir=models/5mb/bam_latn_5mb
cp tokenizers/monolingual/bam_latn_5mb/* models/5mb/bam_latn_5mb

# ban_latn
if test -f models/5mb/ban_latn_5mb/pytorch_model.bin; then
echo "Model already found: ban_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ban_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ban_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=367 --save_steps=999999999 \
--max_steps=7347 \
--warmup_steps=734 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ban_latn_5mb.txt \
--seed=43 \
--override_n_examples=2939 \
--output_dir=models/5mb/ban_latn_5mb
cp tokenizers/monolingual/ban_latn_5mb/* models/5mb/ban_latn_5mb

# bar_latn
if test -f models/5mb/bar_latn_5mb/pytorch_model.bin; then
echo "Model already found: bar_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bar_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bar_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=364 --save_steps=999999999 \
--max_steps=7280 \
--warmup_steps=728 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bar_latn_5mb.txt \
--seed=43 \
--override_n_examples=2912 \
--output_dir=models/5mb/bar_latn_5mb
cp tokenizers/monolingual/bar_latn_5mb/* models/5mb/bar_latn_5mb

# bbc_latn
if test -f models/5mb/bbc_latn_5mb/pytorch_model.bin; then
echo "Model already found: bbc_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bbc_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bbc_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7042 \
--warmup_steps=704 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bbc_latn_5mb.txt \
--seed=43 \
--override_n_examples=2817 \
--output_dir=models/5mb/bbc_latn_5mb
cp tokenizers/monolingual/bbc_latn_5mb/* models/5mb/bbc_latn_5mb

# bcl_latn
if test -f models/5mb/bcl_latn_5mb/pytorch_model.bin; then
echo "Model already found: bcl_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bcl_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bcl_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=346 --save_steps=999999999 \
--max_steps=6937 \
--warmup_steps=693 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bcl_latn_5mb.txt \
--seed=43 \
--override_n_examples=2775 \
--output_dir=models/5mb/bcl_latn_5mb
cp tokenizers/monolingual/bcl_latn_5mb/* models/5mb/bcl_latn_5mb

# bel_cyrl
if test -f models/5mb/bel_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: bel_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bel_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bel_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=329 --save_steps=999999999 \
--max_steps=6580 \
--warmup_steps=658 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bel_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2632 \
--output_dir=models/5mb/bel_cyrl_5mb
cp tokenizers/monolingual/bel_cyrl_5mb/* models/5mb/bel_cyrl_5mb

# bem_latn
if test -f models/5mb/bem_latn_5mb/pytorch_model.bin; then
echo "Model already found: bem_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bem_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bem_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=366 --save_steps=999999999 \
--max_steps=7337 \
--warmup_steps=733 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bem_latn_5mb.txt \
--seed=43 \
--override_n_examples=2935 \
--output_dir=models/5mb/bem_latn_5mb
cp tokenizers/monolingual/bem_latn_5mb/* models/5mb/bem_latn_5mb

# ben_beng
if test -f models/5mb/ben_beng_5mb/pytorch_model.bin; then
echo "Model already found: ben_beng_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ben_beng_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ben_beng_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=252 --save_steps=999999999 \
--max_steps=5047 \
--warmup_steps=504 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ben_beng_5mb.txt \
--seed=43 \
--override_n_examples=2019 \
--output_dir=models/5mb/ben_beng_5mb
cp tokenizers/monolingual/ben_beng_5mb/* models/5mb/ben_beng_5mb

# bew_cyrl
if test -f models/5mb/bew_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: bew_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bew_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bew_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=289 --save_steps=999999999 \
--max_steps=5795 \
--warmup_steps=579 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bew_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2318 \
--output_dir=models/5mb/bew_cyrl_5mb
cp tokenizers/monolingual/bew_cyrl_5mb/* models/5mb/bew_cyrl_5mb

# bew_latn
if test -f models/5mb/bew_latn_5mb/pytorch_model.bin; then
echo "Model already found: bew_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bew_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bew_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=336 --save_steps=999999999 \
--max_steps=6732 \
--warmup_steps=673 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bew_latn_5mb.txt \
--seed=43 \
--override_n_examples=2693 \
--output_dir=models/5mb/bew_latn_5mb
cp tokenizers/monolingual/bew_latn_5mb/* models/5mb/bew_latn_5mb

# bgp_latn
if test -f models/5mb/bgp_latn_5mb/pytorch_model.bin; then
echo "Model already found: bgp_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bgp_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bgp_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7652 \
--warmup_steps=765 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bgp_latn_5mb.txt \
--seed=43 \
--override_n_examples=3061 \
--output_dir=models/5mb/bgp_latn_5mb
cp tokenizers/monolingual/bgp_latn_5mb/* models/5mb/bgp_latn_5mb

# bho_deva
if test -f models/5mb/bho_deva_5mb/pytorch_model.bin; then
echo "Model already found: bho_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bho_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bho_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=330 --save_steps=999999999 \
--max_steps=6615 \
--warmup_steps=661 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bho_deva_5mb.txt \
--seed=43 \
--override_n_examples=2646 \
--output_dir=models/5mb/bho_deva_5mb
cp tokenizers/monolingual/bho_deva_5mb/* models/5mb/bho_deva_5mb

# bik_latn
if test -f models/5mb/bik_latn_5mb/pytorch_model.bin; then
echo "Model already found: bik_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bik_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bik_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=358 --save_steps=999999999 \
--max_steps=7167 \
--warmup_steps=716 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bik_latn_5mb.txt \
--seed=43 \
--override_n_examples=2867 \
--output_dir=models/5mb/bik_latn_5mb
cp tokenizers/monolingual/bik_latn_5mb/* models/5mb/bik_latn_5mb

# bjn_latn
if test -f models/5mb/bjn_latn_5mb/pytorch_model.bin; then
echo "Model already found: bjn_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bjn_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bjn_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=340 --save_steps=999999999 \
--max_steps=6802 \
--warmup_steps=680 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bjn_latn_5mb.txt \
--seed=43 \
--override_n_examples=2721 \
--output_dir=models/5mb/bjn_latn_5mb
cp tokenizers/monolingual/bjn_latn_5mb/* models/5mb/bjn_latn_5mb

# bod_tibt
if test -f models/5mb/bod_tibt_5mb/pytorch_model.bin; then
echo "Model already found: bod_tibt_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bod_tibt_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bod_tibt_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=240 --save_steps=999999999 \
--max_steps=4817 \
--warmup_steps=481 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bod_tibt_5mb.txt \
--seed=43 \
--override_n_examples=1927 \
--output_dir=models/5mb/bod_tibt_5mb
cp tokenizers/monolingual/bod_tibt_5mb/* models/5mb/bod_tibt_5mb

# bos_cyrl
if test -f models/5mb/bos_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: bos_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bos_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bos_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=304 --save_steps=999999999 \
--max_steps=6085 \
--warmup_steps=608 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bos_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2434 \
--output_dir=models/5mb/bos_cyrl_5mb
cp tokenizers/monolingual/bos_cyrl_5mb/* models/5mb/bos_cyrl_5mb

# bos_latn
if test -f models/5mb/bos_latn_5mb/pytorch_model.bin; then
echo "Model already found: bos_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bos_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bos_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=293 --save_steps=999999999 \
--max_steps=5860 \
--warmup_steps=586 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bos_latn_5mb.txt \
--seed=43 \
--override_n_examples=2344 \
--output_dir=models/5mb/bos_latn_5mb
cp tokenizers/monolingual/bos_latn_5mb/* models/5mb/bos_latn_5mb

# bpy_beng
if test -f models/5mb/bpy_beng_5mb/pytorch_model.bin; then
echo "Model already found: bpy_beng_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bpy_beng_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bpy_beng_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5505 \
--warmup_steps=550 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bpy_beng_5mb.txt \
--seed=43 \
--override_n_examples=2202 \
--output_dir=models/5mb/bpy_beng_5mb
cp tokenizers/monolingual/bpy_beng_5mb/* models/5mb/bpy_beng_5mb

# bqc_latn
if test -f models/5mb/bqc_latn_5mb/pytorch_model.bin; then
echo "Model already found: bqc_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bqc_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bqc_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6507 \
--warmup_steps=650 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bqc_latn_5mb.txt \
--seed=43 \
--override_n_examples=2603 \
--output_dir=models/5mb/bqc_latn_5mb
cp tokenizers/monolingual/bqc_latn_5mb/* models/5mb/bqc_latn_5mb

# bre_latn
if test -f models/5mb/bre_latn_5mb/pytorch_model.bin; then
echo "Model already found: bre_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bre_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bre_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=337 --save_steps=999999999 \
--max_steps=6755 \
--warmup_steps=675 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bre_latn_5mb.txt \
--seed=43 \
--override_n_examples=2702 \
--output_dir=models/5mb/bre_latn_5mb
cp tokenizers/monolingual/bre_latn_5mb/* models/5mb/bre_latn_5mb

# bsb_latn
if test -f models/5mb/bsb_latn_5mb/pytorch_model.bin; then
echo "Model already found: bsb_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bsb_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bsb_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=352 --save_steps=999999999 \
--max_steps=7055 \
--warmup_steps=705 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bsb_latn_5mb.txt \
--seed=43 \
--override_n_examples=2822 \
--output_dir=models/5mb/bsb_latn_5mb
cp tokenizers/monolingual/bsb_latn_5mb/* models/5mb/bsb_latn_5mb

# bua_cyrl
if test -f models/5mb/bua_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: bua_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bua_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bua_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=292 --save_steps=999999999 \
--max_steps=5855 \
--warmup_steps=585 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bua_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2342 \
--output_dir=models/5mb/bua_cyrl_5mb
cp tokenizers/monolingual/bua_cyrl_5mb/* models/5mb/bua_cyrl_5mb

# bug_latn
if test -f models/5mb/bug_latn_5mb/pytorch_model.bin; then
echo "Model already found: bug_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bug_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bug_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=379 --save_steps=999999999 \
--max_steps=7592 \
--warmup_steps=759 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bug_latn_5mb.txt \
--seed=43 \
--override_n_examples=3037 \
--output_dir=models/5mb/bug_latn_5mb
cp tokenizers/monolingual/bug_latn_5mb/* models/5mb/bug_latn_5mb

# bul_cyrl
if test -f models/5mb/bul_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: bul_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bul_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bul_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=288 --save_steps=999999999 \
--max_steps=5770 \
--warmup_steps=577 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bul_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2308 \
--output_dir=models/5mb/bul_cyrl_5mb
cp tokenizers/monolingual/bul_cyrl_5mb/* models/5mb/bul_cyrl_5mb

# bxr_cyrl
if test -f models/5mb/bxr_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: bxr_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bxr_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bxr_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=309 --save_steps=999999999 \
--max_steps=6185 \
--warmup_steps=618 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bxr_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2474 \
--output_dir=models/5mb/bxr_cyrl_5mb
cp tokenizers/monolingual/bxr_cyrl_5mb/* models/5mb/bxr_cyrl_5mb

# cak_latn
if test -f models/5mb/cak_latn_5mb/pytorch_model.bin; then
echo "Model already found: cak_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cak_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cak_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=599 --save_steps=999999999 \
--max_steps=11982 \
--warmup_steps=1198 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cak_latn_5mb.txt \
--seed=43 \
--override_n_examples=4793 \
--output_dir=models/5mb/cak_latn_5mb
cp tokenizers/monolingual/cak_latn_5mb/* models/5mb/cak_latn_5mb

# cat_latn
if test -f models/5mb/cat_latn_5mb/pytorch_model.bin; then
echo "Model already found: cat_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cat_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cat_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=301 --save_steps=999999999 \
--max_steps=6032 \
--warmup_steps=603 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cat_latn_5mb.txt \
--seed=43 \
--override_n_examples=2413 \
--output_dir=models/5mb/cat_latn_5mb
cp tokenizers/monolingual/cat_latn_5mb/* models/5mb/cat_latn_5mb

# ceb_latn
if test -f models/5mb/ceb_latn_5mb/pytorch_model.bin; then
echo "Model already found: ceb_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ceb_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ceb_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=338 --save_steps=999999999 \
--max_steps=6767 \
--warmup_steps=676 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ceb_latn_5mb.txt \
--seed=43 \
--override_n_examples=2707 \
--output_dir=models/5mb/ceb_latn_5mb
cp tokenizers/monolingual/ceb_latn_5mb/* models/5mb/ceb_latn_5mb

# ces_latn
if test -f models/5mb/ces_latn_5mb/pytorch_model.bin; then
echo "Model already found: ces_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ces_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ces_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=269 --save_steps=999999999 \
--max_steps=5382 \
--warmup_steps=538 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ces_latn_5mb.txt \
--seed=43 \
--override_n_examples=2153 \
--output_dir=models/5mb/ces_latn_5mb
cp tokenizers/monolingual/ces_latn_5mb/* models/5mb/ces_latn_5mb

# cfm_latn
if test -f models/5mb/cfm_latn_5mb/pytorch_model.bin; then
echo "Model already found: cfm_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cfm_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cfm_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=385 --save_steps=999999999 \
--max_steps=7712 \
--warmup_steps=771 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cfm_latn_5mb.txt \
--seed=43 \
--override_n_examples=3085 \
--output_dir=models/5mb/cfm_latn_5mb
cp tokenizers/monolingual/cfm_latn_5mb/* models/5mb/cfm_latn_5mb

# che_cyrl
if test -f models/5mb/che_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: che_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/che_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/che_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=347 --save_steps=999999999 \
--max_steps=6957 \
--warmup_steps=695 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/che_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2783 \
--output_dir=models/5mb/che_cyrl_5mb
cp tokenizers/monolingual/che_cyrl_5mb/* models/5mb/che_cyrl_5mb

# chm_cyrl
if test -f models/5mb/chm_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: chm_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/chm_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/chm_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=291 --save_steps=999999999 \
--max_steps=5832 \
--warmup_steps=583 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/chm_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2333 \
--output_dir=models/5mb/chm_cyrl_5mb
cp tokenizers/monolingual/chm_cyrl_5mb/* models/5mb/chm_cyrl_5mb

# chv_cyrl
if test -f models/5mb/chv_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: chv_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/chv_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/chv_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=410 --save_steps=999999999 \
--max_steps=8205 \
--warmup_steps=820 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/chv_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=3282 \
--output_dir=models/5mb/chv_cyrl_5mb
cp tokenizers/monolingual/chv_cyrl_5mb/* models/5mb/chv_cyrl_5mb

# cjk_latn
if test -f models/5mb/cjk_latn_5mb/pytorch_model.bin; then
echo "Model already found: cjk_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cjk_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cjk_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=375 --save_steps=999999999 \
--max_steps=7512 \
--warmup_steps=751 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cjk_latn_5mb.txt \
--seed=43 \
--override_n_examples=3005 \
--output_dir=models/5mb/cjk_latn_5mb
cp tokenizers/monolingual/cjk_latn_5mb/* models/5mb/cjk_latn_5mb

# ckb_arab
if test -f models/5mb/ckb_arab_5mb/pytorch_model.bin; then
echo "Model already found: ckb_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ckb_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ckb_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=290 --save_steps=999999999 \
--max_steps=5815 \
--warmup_steps=581 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ckb_arab_5mb.txt \
--seed=43 \
--override_n_examples=2326 \
--output_dir=models/5mb/ckb_arab_5mb
cp tokenizers/monolingual/ckb_arab_5mb/* models/5mb/ckb_arab_5mb

# cnh_latn
if test -f models/5mb/cnh_latn_5mb/pytorch_model.bin; then
echo "Model already found: cnh_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cnh_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cnh_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=397 --save_steps=999999999 \
--max_steps=7950 \
--warmup_steps=795 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cnh_latn_5mb.txt \
--seed=43 \
--override_n_examples=3180 \
--output_dir=models/5mb/cnh_latn_5mb
cp tokenizers/monolingual/cnh_latn_5mb/* models/5mb/cnh_latn_5mb

# cor_latn
if test -f models/5mb/cor_latn_5mb/pytorch_model.bin; then
echo "Model already found: cor_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cor_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cor_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7572 \
--warmup_steps=757 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cor_latn_5mb.txt \
--seed=43 \
--override_n_examples=3029 \
--output_dir=models/5mb/cor_latn_5mb
cp tokenizers/monolingual/cor_latn_5mb/* models/5mb/cor_latn_5mb

# cos_latn
if test -f models/5mb/cos_latn_5mb/pytorch_model.bin; then
echo "Model already found: cos_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cos_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cos_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=394 --save_steps=999999999 \
--max_steps=7887 \
--warmup_steps=788 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cos_latn_5mb.txt \
--seed=43 \
--override_n_examples=3155 \
--output_dir=models/5mb/cos_latn_5mb
cp tokenizers/monolingual/cos_latn_5mb/* models/5mb/cos_latn_5mb

# crh_cyrl
if test -f models/5mb/crh_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: crh_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/crh_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/crh_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5322 \
--warmup_steps=532 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/crh_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2129 \
--output_dir=models/5mb/crh_cyrl_5mb
cp tokenizers/monolingual/crh_cyrl_5mb/* models/5mb/crh_cyrl_5mb

# crh_latn
if test -f models/5mb/crh_latn_5mb/pytorch_model.bin; then
echo "Model already found: crh_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/crh_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/crh_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=351 --save_steps=999999999 \
--max_steps=7025 \
--warmup_steps=702 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/crh_latn_5mb.txt \
--seed=43 \
--override_n_examples=2810 \
--output_dir=models/5mb/crh_latn_5mb
cp tokenizers/monolingual/crh_latn_5mb/* models/5mb/crh_latn_5mb

# ctd_latn
if test -f models/5mb/ctd_latn_5mb/pytorch_model.bin; then
echo "Model already found: ctd_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ctd_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ctd_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=388 --save_steps=999999999 \
--max_steps=7765 \
--warmup_steps=776 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ctd_latn_5mb.txt \
--seed=43 \
--override_n_examples=3106 \
--output_dir=models/5mb/ctd_latn_5mb
cp tokenizers/monolingual/ctd_latn_5mb/* models/5mb/ctd_latn_5mb

# cym_latn
if test -f models/5mb/cym_latn_5mb/pytorch_model.bin; then
echo "Model already found: cym_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cym_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cym_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=300 --save_steps=999999999 \
--max_steps=6010 \
--warmup_steps=601 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cym_latn_5mb.txt \
--seed=43 \
--override_n_examples=2404 \
--output_dir=models/5mb/cym_latn_5mb
cp tokenizers/monolingual/cym_latn_5mb/* models/5mb/cym_latn_5mb

# dan_latn
if test -f models/5mb/dan_latn_5mb/pytorch_model.bin; then
echo "Model already found: dan_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dan_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dan_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=265 --save_steps=999999999 \
--max_steps=5300 \
--warmup_steps=530 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dan_latn_5mb.txt \
--seed=43 \
--override_n_examples=2120 \
--output_dir=models/5mb/dan_latn_5mb
cp tokenizers/monolingual/dan_latn_5mb/* models/5mb/dan_latn_5mb

# dar_cyrl
if test -f models/5mb/dar_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: dar_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dar_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dar_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=333 --save_steps=999999999 \
--max_steps=6672 \
--warmup_steps=667 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dar_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2669 \
--output_dir=models/5mb/dar_cyrl_5mb
cp tokenizers/monolingual/dar_cyrl_5mb/* models/5mb/dar_cyrl_5mb

# deu_latn
if test -f models/5mb/deu_latn_5mb/pytorch_model.bin; then
echo "Model already found: deu_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/deu_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/deu_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=269 --save_steps=999999999 \
--max_steps=5392 \
--warmup_steps=539 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/deu_latn_5mb.txt \
--seed=43 \
--override_n_examples=2157 \
--output_dir=models/5mb/deu_latn_5mb
cp tokenizers/monolingual/deu_latn_5mb/* models/5mb/deu_latn_5mb

# dik_latn
if test -f models/5mb/dik_latn_5mb/pytorch_model.bin; then
echo "Model already found: dik_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dik_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dik_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=403 --save_steps=999999999 \
--max_steps=8060 \
--warmup_steps=806 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dik_latn_5mb.txt \
--seed=43 \
--override_n_examples=3224 \
--output_dir=models/5mb/dik_latn_5mb
cp tokenizers/monolingual/dik_latn_5mb/* models/5mb/dik_latn_5mb

# din_latn
if test -f models/5mb/din_latn_5mb/pytorch_model.bin; then
echo "Model already found: din_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/din_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/din_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=440 --save_steps=999999999 \
--max_steps=8805 \
--warmup_steps=880 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/din_latn_5mb.txt \
--seed=43 \
--override_n_examples=3522 \
--output_dir=models/5mb/din_latn_5mb
cp tokenizers/monolingual/din_latn_5mb/* models/5mb/din_latn_5mb

# diq_latn
if test -f models/5mb/diq_latn_5mb/pytorch_model.bin; then
echo "Model already found: diq_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/diq_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/diq_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=334 --save_steps=999999999 \
--max_steps=6685 \
--warmup_steps=668 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/diq_latn_5mb.txt \
--seed=43 \
--override_n_examples=2674 \
--output_dir=models/5mb/diq_latn_5mb
cp tokenizers/monolingual/diq_latn_5mb/* models/5mb/diq_latn_5mb

# div_thaa
if test -f models/5mb/div_thaa_5mb/pytorch_model.bin; then
echo "Model already found: div_thaa_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/div_thaa_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/div_thaa_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=237 --save_steps=999999999 \
--max_steps=4747 \
--warmup_steps=474 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/div_thaa_5mb.txt \
--seed=43 \
--override_n_examples=1899 \
--output_dir=models/5mb/div_thaa_5mb
cp tokenizers/monolingual/div_thaa_5mb/* models/5mb/div_thaa_5mb

# dov_latn
if test -f models/5mb/dov_latn_5mb/pytorch_model.bin; then
echo "Model already found: dov_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dov_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dov_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4680 \
--warmup_steps=468 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dov_latn_5mb.txt \
--seed=43 \
--override_n_examples=1872 \
--output_dir=models/5mb/dov_latn_5mb
cp tokenizers/monolingual/dov_latn_5mb/* models/5mb/dov_latn_5mb

# dyu_latn
if test -f models/5mb/dyu_latn_5mb/pytorch_model.bin; then
echo "Model already found: dyu_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dyu_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dyu_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=402 --save_steps=999999999 \
--max_steps=8040 \
--warmup_steps=804 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dyu_latn_5mb.txt \
--seed=43 \
--override_n_examples=3216 \
--output_dir=models/5mb/dyu_latn_5mb
cp tokenizers/monolingual/dyu_latn_5mb/* models/5mb/dyu_latn_5mb

# dzo_tibt
if test -f models/5mb/dzo_tibt_5mb/pytorch_model.bin; then
echo "Model already found: dzo_tibt_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dzo_tibt_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dzo_tibt_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6820 \
--warmup_steps=682 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dzo_tibt_5mb.txt \
--seed=43 \
--override_n_examples=2728 \
--output_dir=models/5mb/dzo_tibt_5mb
cp tokenizers/monolingual/dzo_tibt_5mb/* models/5mb/dzo_tibt_5mb

# ekk_latn
if test -f models/5mb/ekk_latn_5mb/pytorch_model.bin; then
echo "Model already found: ekk_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ekk_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ekk_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=261 --save_steps=999999999 \
--max_steps=5230 \
--warmup_steps=523 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ekk_latn_5mb.txt \
--seed=43 \
--override_n_examples=2092 \
--output_dir=models/5mb/ekk_latn_5mb
cp tokenizers/monolingual/ekk_latn_5mb/* models/5mb/ekk_latn_5mb

# ell_grek
if test -f models/5mb/ell_grek_5mb/pytorch_model.bin; then
echo "Model already found: ell_grek_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ell_grek_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ell_grek_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=310 --save_steps=999999999 \
--max_steps=6200 \
--warmup_steps=620 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ell_grek_5mb.txt \
--seed=43 \
--override_n_examples=2480 \
--output_dir=models/5mb/ell_grek_5mb
cp tokenizers/monolingual/ell_grek_5mb/* models/5mb/ell_grek_5mb

# ell_latn
if test -f models/5mb/ell_latn_5mb/pytorch_model.bin; then
echo "Model already found: ell_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ell_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ell_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=318 --save_steps=999999999 \
--max_steps=6375 \
--warmup_steps=637 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ell_latn_5mb.txt \
--seed=43 \
--override_n_examples=2550 \
--output_dir=models/5mb/ell_latn_5mb
cp tokenizers/monolingual/ell_latn_5mb/* models/5mb/ell_latn_5mb

# eng_latn
if test -f models/5mb/eng_latn_5mb/pytorch_model.bin; then
echo "Model already found: eng_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/eng_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/eng_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=272 --save_steps=999999999 \
--max_steps=5447 \
--warmup_steps=544 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/eng_latn_5mb.txt \
--seed=43 \
--override_n_examples=2179 \
--output_dir=models/5mb/eng_latn_5mb
cp tokenizers/monolingual/eng_latn_5mb/* models/5mb/eng_latn_5mb

# epo_latn
if test -f models/5mb/epo_latn_5mb/pytorch_model.bin; then
echo "Model already found: epo_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/epo_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/epo_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=294 --save_steps=999999999 \
--max_steps=5887 \
--warmup_steps=588 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/epo_latn_5mb.txt \
--seed=43 \
--override_n_examples=2355 \
--output_dir=models/5mb/epo_latn_5mb
cp tokenizers/monolingual/epo_latn_5mb/* models/5mb/epo_latn_5mb

# est_latn
if test -f models/5mb/est_latn_5mb/pytorch_model.bin; then
echo "Model already found: est_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/est_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/est_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=243 --save_steps=999999999 \
--max_steps=4862 \
--warmup_steps=486 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/est_latn_5mb.txt \
--seed=43 \
--override_n_examples=1945 \
--output_dir=models/5mb/est_latn_5mb
cp tokenizers/monolingual/est_latn_5mb/* models/5mb/est_latn_5mb

# eus_latn
if test -f models/5mb/eus_latn_5mb/pytorch_model.bin; then
echo "Model already found: eus_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/eus_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/eus_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=267 --save_steps=999999999 \
--max_steps=5357 \
--warmup_steps=535 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/eus_latn_5mb.txt \
--seed=43 \
--override_n_examples=2143 \
--output_dir=models/5mb/eus_latn_5mb
cp tokenizers/monolingual/eus_latn_5mb/* models/5mb/eus_latn_5mb

# ewe_latn
if test -f models/5mb/ewe_latn_5mb/pytorch_model.bin; then
echo "Model already found: ewe_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ewe_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ewe_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=370 --save_steps=999999999 \
--max_steps=7417 \
--warmup_steps=741 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ewe_latn_5mb.txt \
--seed=43 \
--override_n_examples=2967 \
--output_dir=models/5mb/ewe_latn_5mb
cp tokenizers/monolingual/ewe_latn_5mb/* models/5mb/ewe_latn_5mb

# fao_latn
if test -f models/5mb/fao_latn_5mb/pytorch_model.bin; then
echo "Model already found: fao_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fao_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fao_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=307 --save_steps=999999999 \
--max_steps=6150 \
--warmup_steps=615 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fao_latn_5mb.txt \
--seed=43 \
--override_n_examples=2460 \
--output_dir=models/5mb/fao_latn_5mb
cp tokenizers/monolingual/fao_latn_5mb/* models/5mb/fao_latn_5mb

# fas_arab
if test -f models/5mb/fas_arab_5mb/pytorch_model.bin; then
echo "Model already found: fas_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fas_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fas_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=309 --save_steps=999999999 \
--max_steps=6197 \
--warmup_steps=619 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fas_arab_5mb.txt \
--seed=43 \
--override_n_examples=2479 \
--output_dir=models/5mb/fas_arab_5mb
cp tokenizers/monolingual/fas_arab_5mb/* models/5mb/fas_arab_5mb

# fij_latn
if test -f models/5mb/fij_latn_5mb/pytorch_model.bin; then
echo "Model already found: fij_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fij_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fij_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=376 --save_steps=999999999 \
--max_steps=7535 \
--warmup_steps=753 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fij_latn_5mb.txt \
--seed=43 \
--override_n_examples=3014 \
--output_dir=models/5mb/fij_latn_5mb
cp tokenizers/monolingual/fij_latn_5mb/* models/5mb/fij_latn_5mb

# fil_latn
if test -f models/5mb/fil_latn_5mb/pytorch_model.bin; then
echo "Model already found: fil_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fil_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fil_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=351 --save_steps=999999999 \
--max_steps=7037 \
--warmup_steps=703 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fil_latn_5mb.txt \
--seed=43 \
--override_n_examples=2815 \
--output_dir=models/5mb/fil_latn_5mb
cp tokenizers/monolingual/fil_latn_5mb/* models/5mb/fil_latn_5mb

# fin_latn
if test -f models/5mb/fin_latn_5mb/pytorch_model.bin; then
echo "Model already found: fin_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fin_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fin_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=245 --save_steps=999999999 \
--max_steps=4910 \
--warmup_steps=491 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fin_latn_5mb.txt \
--seed=43 \
--override_n_examples=1964 \
--output_dir=models/5mb/fin_latn_5mb
cp tokenizers/monolingual/fin_latn_5mb/* models/5mb/fin_latn_5mb

# fon_latn
if test -f models/5mb/fon_latn_5mb/pytorch_model.bin; then
echo "Model already found: fon_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fon_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fon_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=435 --save_steps=999999999 \
--max_steps=8700 \
--warmup_steps=870 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fon_latn_5mb.txt \
--seed=43 \
--override_n_examples=3480 \
--output_dir=models/5mb/fon_latn_5mb
cp tokenizers/monolingual/fon_latn_5mb/* models/5mb/fon_latn_5mb

# fra_latn
if test -f models/5mb/fra_latn_5mb/pytorch_model.bin; then
echo "Model already found: fra_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fra_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fra_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=317 --save_steps=999999999 \
--max_steps=6347 \
--warmup_steps=634 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fra_latn_5mb.txt \
--seed=43 \
--override_n_examples=2539 \
--output_dir=models/5mb/fra_latn_5mb
cp tokenizers/monolingual/fra_latn_5mb/* models/5mb/fra_latn_5mb

# frr_latn
if test -f models/5mb/frr_latn_5mb/pytorch_model.bin; then
echo "Model already found: frr_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/frr_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/frr_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6902 \
--warmup_steps=690 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/frr_latn_5mb.txt \
--seed=43 \
--override_n_examples=2761 \
--output_dir=models/5mb/frr_latn_5mb
cp tokenizers/monolingual/frr_latn_5mb/* models/5mb/frr_latn_5mb

# fry_latn
if test -f models/5mb/fry_latn_5mb/pytorch_model.bin; then
echo "Model already found: fry_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fry_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fry_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=356 --save_steps=999999999 \
--max_steps=7125 \
--warmup_steps=712 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fry_latn_5mb.txt \
--seed=43 \
--override_n_examples=2850 \
--output_dir=models/5mb/fry_latn_5mb
cp tokenizers/monolingual/fry_latn_5mb/* models/5mb/fry_latn_5mb

# ful_latn
if test -f models/5mb/ful_latn_5mb/pytorch_model.bin; then
echo "Model already found: ful_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ful_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ful_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=409 --save_steps=999999999 \
--max_steps=8195 \
--warmup_steps=819 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ful_latn_5mb.txt \
--seed=43 \
--override_n_examples=3278 \
--output_dir=models/5mb/ful_latn_5mb
cp tokenizers/monolingual/ful_latn_5mb/* models/5mb/ful_latn_5mb

# fur_latn
if test -f models/5mb/fur_latn_5mb/pytorch_model.bin; then
echo "Model already found: fur_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fur_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fur_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=331 --save_steps=999999999 \
--max_steps=6625 \
--warmup_steps=662 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fur_latn_5mb.txt \
--seed=43 \
--override_n_examples=2650 \
--output_dir=models/5mb/fur_latn_5mb
cp tokenizers/monolingual/fur_latn_5mb/* models/5mb/fur_latn_5mb

# fuv_latn
if test -f models/5mb/fuv_latn_5mb/pytorch_model.bin; then
echo "Model already found: fuv_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fuv_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fuv_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=365 --save_steps=999999999 \
--max_steps=7317 \
--warmup_steps=731 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fuv_latn_5mb.txt \
--seed=43 \
--override_n_examples=2927 \
--output_dir=models/5mb/fuv_latn_5mb
cp tokenizers/monolingual/fuv_latn_5mb/* models/5mb/fuv_latn_5mb

# gaz_latn
if test -f models/5mb/gaz_latn_5mb/pytorch_model.bin; then
echo "Model already found: gaz_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gaz_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gaz_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=410 --save_steps=999999999 \
--max_steps=8205 \
--warmup_steps=820 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gaz_latn_5mb.txt \
--seed=43 \
--override_n_examples=3282 \
--output_dir=models/5mb/gaz_latn_5mb
cp tokenizers/monolingual/gaz_latn_5mb/* models/5mb/gaz_latn_5mb

# gla_latn
if test -f models/5mb/gla_latn_5mb/pytorch_model.bin; then
echo "Model already found: gla_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gla_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gla_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=281 --save_steps=999999999 \
--max_steps=5625 \
--warmup_steps=562 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gla_latn_5mb.txt \
--seed=43 \
--override_n_examples=2250 \
--output_dir=models/5mb/gla_latn_5mb
cp tokenizers/monolingual/gla_latn_5mb/* models/5mb/gla_latn_5mb

# gle_latn
if test -f models/5mb/gle_latn_5mb/pytorch_model.bin; then
echo "Model already found: gle_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gle_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gle_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=526 --save_steps=999999999 \
--max_steps=10520 \
--warmup_steps=1052 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gle_latn_5mb.txt \
--seed=43 \
--override_n_examples=4208 \
--output_dir=models/5mb/gle_latn_5mb
cp tokenizers/monolingual/gle_latn_5mb/* models/5mb/gle_latn_5mb

# glg_latn
if test -f models/5mb/glg_latn_5mb/pytorch_model.bin; then
echo "Model already found: glg_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/glg_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/glg_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=283 --save_steps=999999999 \
--max_steps=5662 \
--warmup_steps=566 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/glg_latn_5mb.txt \
--seed=43 \
--override_n_examples=2265 \
--output_dir=models/5mb/glg_latn_5mb
cp tokenizers/monolingual/glg_latn_5mb/* models/5mb/glg_latn_5mb

# glk_arab
if test -f models/5mb/glk_arab_5mb/pytorch_model.bin; then
echo "Model already found: glk_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/glk_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/glk_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=327 --save_steps=999999999 \
--max_steps=6540 \
--warmup_steps=654 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/glk_arab_5mb.txt \
--seed=43 \
--override_n_examples=2616 \
--output_dir=models/5mb/glk_arab_5mb
cp tokenizers/monolingual/glk_arab_5mb/* models/5mb/glk_arab_5mb

# glv_latn
if test -f models/5mb/glv_latn_5mb/pytorch_model.bin; then
echo "Model already found: glv_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/glv_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/glv_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7237 \
--warmup_steps=723 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/glv_latn_5mb.txt \
--seed=43 \
--override_n_examples=2895 \
--output_dir=models/5mb/glv_latn_5mb
cp tokenizers/monolingual/glv_latn_5mb/* models/5mb/glv_latn_5mb

# gom_deva
if test -f models/5mb/gom_deva_5mb/pytorch_model.bin; then
echo "Model already found: gom_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gom_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gom_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=245 --save_steps=999999999 \
--max_steps=4910 \
--warmup_steps=491 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gom_deva_5mb.txt \
--seed=43 \
--override_n_examples=1964 \
--output_dir=models/5mb/gom_deva_5mb
cp tokenizers/monolingual/gom_deva_5mb/* models/5mb/gom_deva_5mb

# gom_latn
if test -f models/5mb/gom_latn_5mb/pytorch_model.bin; then
echo "Model already found: gom_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gom_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gom_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=337 --save_steps=999999999 \
--max_steps=6750 \
--warmup_steps=675 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gom_latn_5mb.txt \
--seed=43 \
--override_n_examples=2700 \
--output_dir=models/5mb/gom_latn_5mb
cp tokenizers/monolingual/gom_latn_5mb/* models/5mb/gom_latn_5mb

# grc_grek
if test -f models/5mb/grc_grek_5mb/pytorch_model.bin; then
echo "Model already found: grc_grek_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/grc_grek_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/grc_grek_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=302 --save_steps=999999999 \
--max_steps=6040 \
--warmup_steps=604 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/grc_grek_5mb.txt \
--seed=43 \
--override_n_examples=2416 \
--output_dir=models/5mb/grc_grek_5mb
cp tokenizers/monolingual/grc_grek_5mb/* models/5mb/grc_grek_5mb

# grn_latn
if test -f models/5mb/grn_latn_5mb/pytorch_model.bin; then
echo "Model already found: grn_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/grn_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/grn_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=324 --save_steps=999999999 \
--max_steps=6485 \
--warmup_steps=648 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/grn_latn_5mb.txt \
--seed=43 \
--override_n_examples=2594 \
--output_dir=models/5mb/grn_latn_5mb
cp tokenizers/monolingual/grn_latn_5mb/* models/5mb/grn_latn_5mb

# gsw_latn
if test -f models/5mb/gsw_latn_5mb/pytorch_model.bin; then
echo "Model already found: gsw_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gsw_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gsw_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=386 --save_steps=999999999 \
--max_steps=7730 \
--warmup_steps=773 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gsw_latn_5mb.txt \
--seed=43 \
--override_n_examples=3092 \
--output_dir=models/5mb/gsw_latn_5mb
cp tokenizers/monolingual/gsw_latn_5mb/* models/5mb/gsw_latn_5mb

# guj_gujr
if test -f models/5mb/guj_gujr_5mb/pytorch_model.bin; then
echo "Model already found: guj_gujr_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/guj_gujr_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/guj_gujr_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=253 --save_steps=999999999 \
--max_steps=5065 \
--warmup_steps=506 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/guj_gujr_5mb.txt \
--seed=43 \
--override_n_examples=2026 \
--output_dir=models/5mb/guj_gujr_5mb
cp tokenizers/monolingual/guj_gujr_5mb/* models/5mb/guj_gujr_5mb

# guj_latn
if test -f models/5mb/guj_latn_5mb/pytorch_model.bin; then
echo "Model already found: guj_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/guj_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/guj_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=312 --save_steps=999999999 \
--max_steps=6255 \
--warmup_steps=625 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/guj_latn_5mb.txt \
--seed=43 \
--override_n_examples=2502 \
--output_dir=models/5mb/guj_latn_5mb
cp tokenizers/monolingual/guj_latn_5mb/* models/5mb/guj_latn_5mb

# hat_latn
if test -f models/5mb/hat_latn_5mb/pytorch_model.bin; then
echo "Model already found: hat_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hat_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hat_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=303 --save_steps=999999999 \
--max_steps=6075 \
--warmup_steps=607 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hat_latn_5mb.txt \
--seed=43 \
--override_n_examples=2430 \
--output_dir=models/5mb/hat_latn_5mb
cp tokenizers/monolingual/hat_latn_5mb/* models/5mb/hat_latn_5mb

# hau_latn
if test -f models/5mb/hau_latn_5mb/pytorch_model.bin; then
echo "Model already found: hau_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hau_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hau_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=350 --save_steps=999999999 \
--max_steps=7017 \
--warmup_steps=701 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hau_latn_5mb.txt \
--seed=43 \
--override_n_examples=2807 \
--output_dir=models/5mb/hau_latn_5mb
cp tokenizers/monolingual/hau_latn_5mb/* models/5mb/hau_latn_5mb

# haw_latn
if test -f models/5mb/haw_latn_5mb/pytorch_model.bin; then
echo "Model already found: haw_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/haw_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/haw_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=415 --save_steps=999999999 \
--max_steps=8307 \
--warmup_steps=830 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/haw_latn_5mb.txt \
--seed=43 \
--override_n_examples=3323 \
--output_dir=models/5mb/haw_latn_5mb
cp tokenizers/monolingual/haw_latn_5mb/* models/5mb/haw_latn_5mb

# heb_hebr
if test -f models/5mb/heb_hebr_5mb/pytorch_model.bin; then
echo "Model already found: heb_hebr_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/heb_hebr_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/heb_hebr_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=251 --save_steps=999999999 \
--max_steps=5025 \
--warmup_steps=502 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/heb_hebr_5mb.txt \
--seed=43 \
--override_n_examples=2010 \
--output_dir=models/5mb/heb_hebr_5mb
cp tokenizers/monolingual/heb_hebr_5mb/* models/5mb/heb_hebr_5mb

# hif_latn
if test -f models/5mb/hif_latn_5mb/pytorch_model.bin; then
echo "Model already found: hif_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hif_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hif_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=404 --save_steps=999999999 \
--max_steps=8087 \
--warmup_steps=808 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hif_latn_5mb.txt \
--seed=43 \
--override_n_examples=3235 \
--output_dir=models/5mb/hif_latn_5mb
cp tokenizers/monolingual/hif_latn_5mb/* models/5mb/hif_latn_5mb

# hil_latn
if test -f models/5mb/hil_latn_5mb/pytorch_model.bin; then
echo "Model already found: hil_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hil_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hil_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=360 --save_steps=999999999 \
--max_steps=7200 \
--warmup_steps=720 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hil_latn_5mb.txt \
--seed=43 \
--override_n_examples=2880 \
--output_dir=models/5mb/hil_latn_5mb
cp tokenizers/monolingual/hil_latn_5mb/* models/5mb/hil_latn_5mb

# hin_deva
if test -f models/5mb/hin_deva_5mb/pytorch_model.bin; then
echo "Model already found: hin_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hin_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hin_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=297 --save_steps=999999999 \
--max_steps=5957 \
--warmup_steps=595 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hin_deva_5mb.txt \
--seed=43 \
--override_n_examples=2383 \
--output_dir=models/5mb/hin_deva_5mb
cp tokenizers/monolingual/hin_deva_5mb/* models/5mb/hin_deva_5mb

# hin_latn
if test -f models/5mb/hin_latn_5mb/pytorch_model.bin; then
echo "Model already found: hin_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hin_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hin_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=370 --save_steps=999999999 \
--max_steps=7410 \
--warmup_steps=741 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hin_latn_5mb.txt \
--seed=43 \
--override_n_examples=2964 \
--output_dir=models/5mb/hin_latn_5mb
cp tokenizers/monolingual/hin_latn_5mb/* models/5mb/hin_latn_5mb

# hmn_latn
if test -f models/5mb/hmn_latn_5mb/pytorch_model.bin; then
echo "Model already found: hmn_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hmn_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hmn_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=364 --save_steps=999999999 \
--max_steps=7287 \
--warmup_steps=728 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hmn_latn_5mb.txt \
--seed=43 \
--override_n_examples=2915 \
--output_dir=models/5mb/hmn_latn_5mb
cp tokenizers/monolingual/hmn_latn_5mb/* models/5mb/hmn_latn_5mb

# hne_deva
if test -f models/5mb/hne_deva_5mb/pytorch_model.bin; then
echo "Model already found: hne_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hne_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hne_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6100 \
--warmup_steps=610 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hne_deva_5mb.txt \
--seed=43 \
--override_n_examples=2440 \
--output_dir=models/5mb/hne_deva_5mb
cp tokenizers/monolingual/hne_deva_5mb/* models/5mb/hne_deva_5mb

# hrv_latn
if test -f models/5mb/hrv_latn_5mb/pytorch_model.bin; then
echo "Model already found: hrv_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hrv_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hrv_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=285 --save_steps=999999999 \
--max_steps=5702 \
--warmup_steps=570 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hrv_latn_5mb.txt \
--seed=43 \
--override_n_examples=2281 \
--output_dir=models/5mb/hrv_latn_5mb
cp tokenizers/monolingual/hrv_latn_5mb/* models/5mb/hrv_latn_5mb

# hsb_latn
if test -f models/5mb/hsb_latn_5mb/pytorch_model.bin; then
echo "Model already found: hsb_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hsb_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hsb_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=305 --save_steps=999999999 \
--max_steps=6107 \
--warmup_steps=610 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hsb_latn_5mb.txt \
--seed=43 \
--override_n_examples=2443 \
--output_dir=models/5mb/hsb_latn_5mb
cp tokenizers/monolingual/hsb_latn_5mb/* models/5mb/hsb_latn_5mb

# hun_latn
if test -f models/5mb/hun_latn_5mb/pytorch_model.bin; then
echo "Model already found: hun_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hun_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hun_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=246 --save_steps=999999999 \
--max_steps=4937 \
--warmup_steps=493 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hun_latn_5mb.txt \
--seed=43 \
--override_n_examples=1975 \
--output_dir=models/5mb/hun_latn_5mb
cp tokenizers/monolingual/hun_latn_5mb/* models/5mb/hun_latn_5mb

# hye_armn
if test -f models/5mb/hye_armn_5mb/pytorch_model.bin; then
echo "Model already found: hye_armn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hye_armn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hye_armn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=262 --save_steps=999999999 \
--max_steps=5250 \
--warmup_steps=525 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hye_armn_5mb.txt \
--seed=43 \
--override_n_examples=2100 \
--output_dir=models/5mb/hye_armn_5mb
cp tokenizers/monolingual/hye_armn_5mb/* models/5mb/hye_armn_5mb

# iba_latn
if test -f models/5mb/iba_latn_5mb/pytorch_model.bin; then
echo "Model already found: iba_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/iba_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/iba_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=337 --save_steps=999999999 \
--max_steps=6740 \
--warmup_steps=674 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/iba_latn_5mb.txt \
--seed=43 \
--override_n_examples=2696 \
--output_dir=models/5mb/iba_latn_5mb
cp tokenizers/monolingual/iba_latn_5mb/* models/5mb/iba_latn_5mb

# ibo_latn
if test -f models/5mb/ibo_latn_5mb/pytorch_model.bin; then
echo "Model already found: ibo_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ibo_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ibo_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=389 --save_steps=999999999 \
--max_steps=7792 \
--warmup_steps=779 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ibo_latn_5mb.txt \
--seed=43 \
--override_n_examples=3117 \
--output_dir=models/5mb/ibo_latn_5mb
cp tokenizers/monolingual/ibo_latn_5mb/* models/5mb/ibo_latn_5mb

# ido_latn
if test -f models/5mb/ido_latn_5mb/pytorch_model.bin; then
echo "Model already found: ido_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ido_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ido_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=357 --save_steps=999999999 \
--max_steps=7147 \
--warmup_steps=714 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ido_latn_5mb.txt \
--seed=43 \
--override_n_examples=2859 \
--output_dir=models/5mb/ido_latn_5mb
cp tokenizers/monolingual/ido_latn_5mb/* models/5mb/ido_latn_5mb

# iku_cans
if test -f models/5mb/iku_cans_5mb/pytorch_model.bin; then
echo "Model already found: iku_cans_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/iku_cans_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/iku_cans_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=239 --save_steps=999999999 \
--max_steps=4780 \
--warmup_steps=478 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/iku_cans_5mb.txt \
--seed=43 \
--override_n_examples=1912 \
--output_dir=models/5mb/iku_cans_5mb
cp tokenizers/monolingual/iku_cans_5mb/* models/5mb/iku_cans_5mb

# ilo_latn
if test -f models/5mb/ilo_latn_5mb/pytorch_model.bin; then
echo "Model already found: ilo_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ilo_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ilo_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=330 --save_steps=999999999 \
--max_steps=6607 \
--warmup_steps=660 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ilo_latn_5mb.txt \
--seed=43 \
--override_n_examples=2643 \
--output_dir=models/5mb/ilo_latn_5mb
cp tokenizers/monolingual/ilo_latn_5mb/* models/5mb/ilo_latn_5mb

# ina_latn
if test -f models/5mb/ina_latn_5mb/pytorch_model.bin; then
echo "Model already found: ina_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ina_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ina_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=369 --save_steps=999999999 \
--max_steps=7382 \
--warmup_steps=738 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ina_latn_5mb.txt \
--seed=43 \
--override_n_examples=2953 \
--output_dir=models/5mb/ina_latn_5mb
cp tokenizers/monolingual/ina_latn_5mb/* models/5mb/ina_latn_5mb

# ind_latn
if test -f models/5mb/ind_latn_5mb/pytorch_model.bin; then
echo "Model already found: ind_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ind_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ind_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=270 --save_steps=999999999 \
--max_steps=5407 \
--warmup_steps=540 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ind_latn_5mb.txt \
--seed=43 \
--override_n_examples=2163 \
--output_dir=models/5mb/ind_latn_5mb
cp tokenizers/monolingual/ind_latn_5mb/* models/5mb/ind_latn_5mb

# inh_cyrl
if test -f models/5mb/inh_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: inh_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/inh_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/inh_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=357 --save_steps=999999999 \
--max_steps=7145 \
--warmup_steps=714 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/inh_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2858 \
--output_dir=models/5mb/inh_cyrl_5mb
cp tokenizers/monolingual/inh_cyrl_5mb/* models/5mb/inh_cyrl_5mb

# isl_latn
if test -f models/5mb/isl_latn_5mb/pytorch_model.bin; then
echo "Model already found: isl_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/isl_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/isl_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=303 --save_steps=999999999 \
--max_steps=6067 \
--warmup_steps=606 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/isl_latn_5mb.txt \
--seed=43 \
--override_n_examples=2427 \
--output_dir=models/5mb/isl_latn_5mb
cp tokenizers/monolingual/isl_latn_5mb/* models/5mb/isl_latn_5mb

# iso_latn
if test -f models/5mb/iso_latn_5mb/pytorch_model.bin; then
echo "Model already found: iso_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/iso_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/iso_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8085 \
--warmup_steps=808 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/iso_latn_5mb.txt \
--seed=43 \
--override_n_examples=3234 \
--output_dir=models/5mb/iso_latn_5mb
cp tokenizers/monolingual/iso_latn_5mb/* models/5mb/iso_latn_5mb

# ita_latn
if test -f models/5mb/ita_latn_5mb/pytorch_model.bin; then
echo "Model already found: ita_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ita_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ita_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=276 --save_steps=999999999 \
--max_steps=5537 \
--warmup_steps=553 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ita_latn_5mb.txt \
--seed=43 \
--override_n_examples=2215 \
--output_dir=models/5mb/ita_latn_5mb
cp tokenizers/monolingual/ita_latn_5mb/* models/5mb/ita_latn_5mb

# jav_latn
if test -f models/5mb/jav_latn_5mb/pytorch_model.bin; then
echo "Model already found: jav_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/jav_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/jav_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=316 --save_steps=999999999 \
--max_steps=6325 \
--warmup_steps=632 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/jav_latn_5mb.txt \
--seed=43 \
--override_n_examples=2530 \
--output_dir=models/5mb/jav_latn_5mb
cp tokenizers/monolingual/jav_latn_5mb/* models/5mb/jav_latn_5mb

# jpn_jpan
if test -f models/5mb/jpn_jpan_5mb/pytorch_model.bin; then
echo "Model already found: jpn_jpan_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/jpn_jpan_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/jpn_jpan_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=282 --save_steps=999999999 \
--max_steps=5657 \
--warmup_steps=565 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/jpn_jpan_5mb.txt \
--seed=43 \
--override_n_examples=2263 \
--output_dir=models/5mb/jpn_jpan_5mb
cp tokenizers/monolingual/jpn_jpan_5mb/* models/5mb/jpn_jpan_5mb

# kaa_cyrl
if test -f models/5mb/kaa_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: kaa_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaa_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaa_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=274 --save_steps=999999999 \
--max_steps=5487 \
--warmup_steps=548 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaa_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2195 \
--output_dir=models/5mb/kaa_cyrl_5mb
cp tokenizers/monolingual/kaa_cyrl_5mb/* models/5mb/kaa_cyrl_5mb

# kaa_latn
if test -f models/5mb/kaa_latn_5mb/pytorch_model.bin; then
echo "Model already found: kaa_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaa_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaa_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=304 --save_steps=999999999 \
--max_steps=6080 \
--warmup_steps=608 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaa_latn_5mb.txt \
--seed=43 \
--override_n_examples=2432 \
--output_dir=models/5mb/kaa_latn_5mb
cp tokenizers/monolingual/kaa_latn_5mb/* models/5mb/kaa_latn_5mb

# kab_latn
if test -f models/5mb/kab_latn_5mb/pytorch_model.bin; then
echo "Model already found: kab_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kab_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kab_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=393 --save_steps=999999999 \
--max_steps=7862 \
--warmup_steps=786 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kab_latn_5mb.txt \
--seed=43 \
--override_n_examples=3145 \
--output_dir=models/5mb/kab_latn_5mb
cp tokenizers/monolingual/kab_latn_5mb/* models/5mb/kab_latn_5mb

# kac_latn
if test -f models/5mb/kac_latn_5mb/pytorch_model.bin; then
echo "Model already found: kac_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kac_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kac_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=437 --save_steps=999999999 \
--max_steps=8745 \
--warmup_steps=874 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kac_latn_5mb.txt \
--seed=43 \
--override_n_examples=3498 \
--output_dir=models/5mb/kac_latn_5mb
cp tokenizers/monolingual/kac_latn_5mb/* models/5mb/kac_latn_5mb

# kal_latn
if test -f models/5mb/kal_latn_5mb/pytorch_model.bin; then
echo "Model already found: kal_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kal_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kal_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=282 --save_steps=999999999 \
--max_steps=5655 \
--warmup_steps=565 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kal_latn_5mb.txt \
--seed=43 \
--override_n_examples=2262 \
--output_dir=models/5mb/kal_latn_5mb
cp tokenizers/monolingual/kal_latn_5mb/* models/5mb/kal_latn_5mb

# kan_knda
if test -f models/5mb/kan_knda_5mb/pytorch_model.bin; then
echo "Model already found: kan_knda_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kan_knda_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kan_knda_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=276 --save_steps=999999999 \
--max_steps=5530 \
--warmup_steps=553 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kan_knda_5mb.txt \
--seed=43 \
--override_n_examples=2212 \
--output_dir=models/5mb/kan_knda_5mb
cp tokenizers/monolingual/kan_knda_5mb/* models/5mb/kan_knda_5mb

# kas_deva
if test -f models/5mb/kas_deva_5mb/pytorch_model.bin; then
echo "Model already found: kas_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kas_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kas_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7050 \
--warmup_steps=705 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kas_deva_5mb.txt \
--seed=43 \
--override_n_examples=2820 \
--output_dir=models/5mb/kas_deva_5mb
cp tokenizers/monolingual/kas_deva_5mb/* models/5mb/kas_deva_5mb

# kat_geor
if test -f models/5mb/kat_geor_5mb/pytorch_model.bin; then
echo "Model already found: kat_geor_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kat_geor_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kat_geor_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=451 --save_steps=999999999 \
--max_steps=9030 \
--warmup_steps=903 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kat_geor_5mb.txt \
--seed=43 \
--override_n_examples=3612 \
--output_dir=models/5mb/kat_geor_5mb
cp tokenizers/monolingual/kat_geor_5mb/* models/5mb/kat_geor_5mb

# kat_latn
if test -f models/5mb/kat_latn_5mb/pytorch_model.bin; then
echo "Model already found: kat_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kat_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kat_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6300 \
--warmup_steps=630 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kat_latn_5mb.txt \
--seed=43 \
--override_n_examples=2520 \
--output_dir=models/5mb/kat_latn_5mb
cp tokenizers/monolingual/kat_latn_5mb/* models/5mb/kat_latn_5mb

# kaz_cyrl
if test -f models/5mb/kaz_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: kaz_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaz_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaz_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=259 --save_steps=999999999 \
--max_steps=5197 \
--warmup_steps=519 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaz_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2079 \
--output_dir=models/5mb/kaz_cyrl_5mb
cp tokenizers/monolingual/kaz_cyrl_5mb/* models/5mb/kaz_cyrl_5mb

# kbd_cyrl
if test -f models/5mb/kbd_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: kbd_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kbd_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kbd_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=354 --save_steps=999999999 \
--max_steps=7092 \
--warmup_steps=709 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kbd_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2837 \
--output_dir=models/5mb/kbd_cyrl_5mb
cp tokenizers/monolingual/kbd_cyrl_5mb/* models/5mb/kbd_cyrl_5mb

# kbp_latn
if test -f models/5mb/kbp_latn_5mb/pytorch_model.bin; then
echo "Model already found: kbp_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kbp_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kbp_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=424 --save_steps=999999999 \
--max_steps=8480 \
--warmup_steps=848 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kbp_latn_5mb.txt \
--seed=43 \
--override_n_examples=3392 \
--output_dir=models/5mb/kbp_latn_5mb
cp tokenizers/monolingual/kbp_latn_5mb/* models/5mb/kbp_latn_5mb

# kea_latn
if test -f models/5mb/kea_latn_5mb/pytorch_model.bin; then
echo "Model already found: kea_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kea_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kea_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=259 --save_steps=999999999 \
--max_steps=5180 \
--warmup_steps=518 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kea_latn_5mb.txt \
--seed=43 \
--override_n_examples=2072 \
--output_dir=models/5mb/kea_latn_5mb
cp tokenizers/monolingual/kea_latn_5mb/* models/5mb/kea_latn_5mb

# kha_latn
if test -f models/5mb/kha_latn_5mb/pytorch_model.bin; then
echo "Model already found: kha_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kha_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kha_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=380 --save_steps=999999999 \
--max_steps=7610 \
--warmup_steps=761 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kha_latn_5mb.txt \
--seed=43 \
--override_n_examples=3044 \
--output_dir=models/5mb/kha_latn_5mb
cp tokenizers/monolingual/kha_latn_5mb/* models/5mb/kha_latn_5mb

# khk_cyrl
if test -f models/5mb/khk_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: khk_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/khk_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/khk_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=281 --save_steps=999999999 \
--max_steps=5627 \
--warmup_steps=562 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/khk_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2251 \
--output_dir=models/5mb/khk_cyrl_5mb
cp tokenizers/monolingual/khk_cyrl_5mb/* models/5mb/khk_cyrl_5mb

# khm_khmr
if test -f models/5mb/khm_khmr_5mb/pytorch_model.bin; then
echo "Model already found: khm_khmr_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/khm_khmr_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/khm_khmr_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=400 --save_steps=999999999 \
--max_steps=8005 \
--warmup_steps=800 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/khm_khmr_5mb.txt \
--seed=43 \
--override_n_examples=3202 \
--output_dir=models/5mb/khm_khmr_5mb
cp tokenizers/monolingual/khm_khmr_5mb/* models/5mb/khm_khmr_5mb

# kik_latn
if test -f models/5mb/kik_latn_5mb/pytorch_model.bin; then
echo "Model already found: kik_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kik_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kik_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7167 \
--warmup_steps=716 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kik_latn_5mb.txt \
--seed=43 \
--override_n_examples=2867 \
--output_dir=models/5mb/kik_latn_5mb
cp tokenizers/monolingual/kik_latn_5mb/* models/5mb/kik_latn_5mb

# kin_latn
if test -f models/5mb/kin_latn_5mb/pytorch_model.bin; then
echo "Model already found: kin_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kin_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kin_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=311 --save_steps=999999999 \
--max_steps=6220 \
--warmup_steps=622 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kin_latn_5mb.txt \
--seed=43 \
--override_n_examples=2488 \
--output_dir=models/5mb/kin_latn_5mb
cp tokenizers/monolingual/kin_latn_5mb/* models/5mb/kin_latn_5mb

# kir_cyrl
if test -f models/5mb/kir_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: kir_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kir_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kir_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=287 --save_steps=999999999 \
--max_steps=5742 \
--warmup_steps=574 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kir_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2297 \
--output_dir=models/5mb/kir_cyrl_5mb
cp tokenizers/monolingual/kir_cyrl_5mb/* models/5mb/kir_cyrl_5mb

# kjh_cyrl
if test -f models/5mb/kjh_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: kjh_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kjh_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kjh_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5105 \
--warmup_steps=510 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kjh_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2042 \
--output_dir=models/5mb/kjh_cyrl_5mb
cp tokenizers/monolingual/kjh_cyrl_5mb/* models/5mb/kjh_cyrl_5mb

# kmb_latn
if test -f models/5mb/kmb_latn_5mb/pytorch_model.bin; then
echo "Model already found: kmb_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kmb_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kmb_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=377 --save_steps=999999999 \
--max_steps=7555 \
--warmup_steps=755 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kmb_latn_5mb.txt \
--seed=43 \
--override_n_examples=3022 \
--output_dir=models/5mb/kmb_latn_5mb
cp tokenizers/monolingual/kmb_latn_5mb/* models/5mb/kmb_latn_5mb

# kmr_latn
if test -f models/5mb/kmr_latn_5mb/pytorch_model.bin; then
echo "Model already found: kmr_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kmr_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kmr_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=294 --save_steps=999999999 \
--max_steps=5882 \
--warmup_steps=588 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kmr_latn_5mb.txt \
--seed=43 \
--override_n_examples=2353 \
--output_dir=models/5mb/kmr_latn_5mb
cp tokenizers/monolingual/kmr_latn_5mb/* models/5mb/kmr_latn_5mb

# knc_arab
if test -f models/5mb/knc_arab_5mb/pytorch_model.bin; then
echo "Model already found: knc_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/knc_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/knc_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=1304 --save_steps=999999999 \
--max_steps=26080 \
--warmup_steps=2608 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/knc_arab_5mb.txt \
--seed=43 \
--override_n_examples=10432 \
--output_dir=models/5mb/knc_arab_5mb
cp tokenizers/monolingual/knc_arab_5mb/* models/5mb/knc_arab_5mb

# knc_latn
if test -f models/5mb/knc_latn_5mb/pytorch_model.bin; then
echo "Model already found: knc_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/knc_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/knc_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=425 --save_steps=999999999 \
--max_steps=8505 \
--warmup_steps=850 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/knc_latn_5mb.txt \
--seed=43 \
--override_n_examples=3402 \
--output_dir=models/5mb/knc_latn_5mb
cp tokenizers/monolingual/knc_latn_5mb/* models/5mb/knc_latn_5mb

# kom_cyrl
if test -f models/5mb/kom_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: kom_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kom_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kom_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=326 --save_steps=999999999 \
--max_steps=6522 \
--warmup_steps=652 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kom_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2609 \
--output_dir=models/5mb/kom_cyrl_5mb
cp tokenizers/monolingual/kom_cyrl_5mb/* models/5mb/kom_cyrl_5mb

# kon_latn
if test -f models/5mb/kon_latn_5mb/pytorch_model.bin; then
echo "Model already found: kon_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kon_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kon_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=391 --save_steps=999999999 \
--max_steps=7827 \
--warmup_steps=782 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kon_latn_5mb.txt \
--seed=43 \
--override_n_examples=3131 \
--output_dir=models/5mb/kon_latn_5mb
cp tokenizers/monolingual/kon_latn_5mb/* models/5mb/kon_latn_5mb

# kor_hang
if test -f models/5mb/kor_hang_5mb/pytorch_model.bin; then
echo "Model already found: kor_hang_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kor_hang_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kor_hang_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=298 --save_steps=999999999 \
--max_steps=5977 \
--warmup_steps=597 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kor_hang_5mb.txt \
--seed=43 \
--override_n_examples=2391 \
--output_dir=models/5mb/kor_hang_5mb
cp tokenizers/monolingual/kor_hang_5mb/* models/5mb/kor_hang_5mb

# kpv_cyrl
if test -f models/5mb/kpv_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: kpv_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kpv_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kpv_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6445 \
--warmup_steps=644 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kpv_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2578 \
--output_dir=models/5mb/kpv_cyrl_5mb
cp tokenizers/monolingual/kpv_cyrl_5mb/* models/5mb/kpv_cyrl_5mb

# krc_cyrl
if test -f models/5mb/krc_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: krc_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/krc_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/krc_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=276 --save_steps=999999999 \
--max_steps=5532 \
--warmup_steps=553 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/krc_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2213 \
--output_dir=models/5mb/krc_cyrl_5mb
cp tokenizers/monolingual/krc_cyrl_5mb/* models/5mb/krc_cyrl_5mb

# kum_cyrl
if test -f models/5mb/kum_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: kum_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kum_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kum_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=252 --save_steps=999999999 \
--max_steps=5050 \
--warmup_steps=505 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kum_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2020 \
--output_dir=models/5mb/kum_cyrl_5mb
cp tokenizers/monolingual/kum_cyrl_5mb/* models/5mb/kum_cyrl_5mb

# kur_arab
if test -f models/5mb/kur_arab_5mb/pytorch_model.bin; then
echo "Model already found: kur_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kur_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kur_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=277 --save_steps=999999999 \
--max_steps=5555 \
--warmup_steps=555 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kur_arab_5mb.txt \
--seed=43 \
--override_n_examples=2222 \
--output_dir=models/5mb/kur_arab_5mb
cp tokenizers/monolingual/kur_arab_5mb/* models/5mb/kur_arab_5mb

# kur_latn
if test -f models/5mb/kur_latn_5mb/pytorch_model.bin; then
echo "Model already found: kur_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kur_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kur_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=357 --save_steps=999999999 \
--max_steps=7155 \
--warmup_steps=715 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kur_latn_5mb.txt \
--seed=43 \
--override_n_examples=2862 \
--output_dir=models/5mb/kur_latn_5mb
cp tokenizers/monolingual/kur_latn_5mb/* models/5mb/kur_latn_5mb

# lao_laoo
if test -f models/5mb/lao_laoo_5mb/pytorch_model.bin; then
echo "Model already found: lao_laoo_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lao_laoo_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lao_laoo_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=295 --save_steps=999999999 \
--max_steps=5910 \
--warmup_steps=591 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lao_laoo_5mb.txt \
--seed=43 \
--override_n_examples=2364 \
--output_dir=models/5mb/lao_laoo_5mb
cp tokenizers/monolingual/lao_laoo_5mb/* models/5mb/lao_laoo_5mb

# lat_latn
if test -f models/5mb/lat_latn_5mb/pytorch_model.bin; then
echo "Model already found: lat_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lat_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lat_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=243 --save_steps=999999999 \
--max_steps=4865 \
--warmup_steps=486 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lat_latn_5mb.txt \
--seed=43 \
--override_n_examples=1946 \
--output_dir=models/5mb/lat_latn_5mb
cp tokenizers/monolingual/lat_latn_5mb/* models/5mb/lat_latn_5mb

# lav_latn
if test -f models/5mb/lav_latn_5mb/pytorch_model.bin; then
echo "Model already found: lav_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lav_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lav_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=316 --save_steps=999999999 \
--max_steps=6335 \
--warmup_steps=633 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lav_latn_5mb.txt \
--seed=43 \
--override_n_examples=2534 \
--output_dir=models/5mb/lav_latn_5mb
cp tokenizers/monolingual/lav_latn_5mb/* models/5mb/lav_latn_5mb

# lbe_cyrl
if test -f models/5mb/lbe_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: lbe_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lbe_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lbe_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=300 --save_steps=999999999 \
--max_steps=6000 \
--warmup_steps=600 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lbe_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2400 \
--output_dir=models/5mb/lbe_cyrl_5mb
cp tokenizers/monolingual/lbe_cyrl_5mb/* models/5mb/lbe_cyrl_5mb

# lez_cyrl
if test -f models/5mb/lez_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: lez_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lez_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lez_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=312 --save_steps=999999999 \
--max_steps=6252 \
--warmup_steps=625 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lez_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2501 \
--output_dir=models/5mb/lez_cyrl_5mb
cp tokenizers/monolingual/lez_cyrl_5mb/* models/5mb/lez_cyrl_5mb

# lfn_latn
if test -f models/5mb/lfn_latn_5mb/pytorch_model.bin; then
echo "Model already found: lfn_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lfn_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lfn_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7627 \
--warmup_steps=762 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lfn_latn_5mb.txt \
--seed=43 \
--override_n_examples=3051 \
--output_dir=models/5mb/lfn_latn_5mb
cp tokenizers/monolingual/lfn_latn_5mb/* models/5mb/lfn_latn_5mb

# lij_latn
if test -f models/5mb/lij_latn_5mb/pytorch_model.bin; then
echo "Model already found: lij_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lij_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lij_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=387 --save_steps=999999999 \
--max_steps=7750 \
--warmup_steps=775 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lij_latn_5mb.txt \
--seed=43 \
--override_n_examples=3100 \
--output_dir=models/5mb/lij_latn_5mb
cp tokenizers/monolingual/lij_latn_5mb/* models/5mb/lij_latn_5mb

# lim_latn
if test -f models/5mb/lim_latn_5mb/pytorch_model.bin; then
echo "Model already found: lim_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lim_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lim_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=355 --save_steps=999999999 \
--max_steps=7115 \
--warmup_steps=711 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lim_latn_5mb.txt \
--seed=43 \
--override_n_examples=2846 \
--output_dir=models/5mb/lim_latn_5mb
cp tokenizers/monolingual/lim_latn_5mb/* models/5mb/lim_latn_5mb

# lin_latn
if test -f models/5mb/lin_latn_5mb/pytorch_model.bin; then
echo "Model already found: lin_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lin_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lin_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=354 --save_steps=999999999 \
--max_steps=7095 \
--warmup_steps=709 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lin_latn_5mb.txt \
--seed=43 \
--override_n_examples=2838 \
--output_dir=models/5mb/lin_latn_5mb
cp tokenizers/monolingual/lin_latn_5mb/* models/5mb/lin_latn_5mb

# lit_latn
if test -f models/5mb/lit_latn_5mb/pytorch_model.bin; then
echo "Model already found: lit_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lit_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lit_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=262 --save_steps=999999999 \
--max_steps=5245 \
--warmup_steps=524 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lit_latn_5mb.txt \
--seed=43 \
--override_n_examples=2098 \
--output_dir=models/5mb/lit_latn_5mb
cp tokenizers/monolingual/lit_latn_5mb/* models/5mb/lit_latn_5mb

# lmo_latn
if test -f models/5mb/lmo_latn_5mb/pytorch_model.bin; then
echo "Model already found: lmo_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lmo_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lmo_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=366 --save_steps=999999999 \
--max_steps=7327 \
--warmup_steps=732 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lmo_latn_5mb.txt \
--seed=43 \
--override_n_examples=2931 \
--output_dir=models/5mb/lmo_latn_5mb
cp tokenizers/monolingual/lmo_latn_5mb/* models/5mb/lmo_latn_5mb

# ltg_latn
if test -f models/5mb/ltg_latn_5mb/pytorch_model.bin; then
echo "Model already found: ltg_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ltg_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ltg_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=274 --save_steps=999999999 \
--max_steps=5487 \
--warmup_steps=548 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ltg_latn_5mb.txt \
--seed=43 \
--override_n_examples=2195 \
--output_dir=models/5mb/ltg_latn_5mb
cp tokenizers/monolingual/ltg_latn_5mb/* models/5mb/ltg_latn_5mb

# ltz_latn
if test -f models/5mb/ltz_latn_5mb/pytorch_model.bin; then
echo "Model already found: ltz_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ltz_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ltz_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=351 --save_steps=999999999 \
--max_steps=7025 \
--warmup_steps=702 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ltz_latn_5mb.txt \
--seed=43 \
--override_n_examples=2810 \
--output_dir=models/5mb/ltz_latn_5mb
cp tokenizers/monolingual/ltz_latn_5mb/* models/5mb/ltz_latn_5mb

# lua_latn
if test -f models/5mb/lua_latn_5mb/pytorch_model.bin; then
echo "Model already found: lua_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lua_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lua_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=383 --save_steps=999999999 \
--max_steps=7677 \
--warmup_steps=767 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lua_latn_5mb.txt \
--seed=43 \
--override_n_examples=3071 \
--output_dir=models/5mb/lua_latn_5mb
cp tokenizers/monolingual/lua_latn_5mb/* models/5mb/lua_latn_5mb

# lub_latn
if test -f models/5mb/lub_latn_5mb/pytorch_model.bin; then
echo "Model already found: lub_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lub_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lub_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6610 \
--warmup_steps=661 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lub_latn_5mb.txt \
--seed=43 \
--override_n_examples=2644 \
--output_dir=models/5mb/lub_latn_5mb
cp tokenizers/monolingual/lub_latn_5mb/* models/5mb/lub_latn_5mb

# lug_latn
if test -f models/5mb/lug_latn_5mb/pytorch_model.bin; then
echo "Model already found: lug_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lug_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lug_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=364 --save_steps=999999999 \
--max_steps=7292 \
--warmup_steps=729 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lug_latn_5mb.txt \
--seed=43 \
--override_n_examples=2917 \
--output_dir=models/5mb/lug_latn_5mb
cp tokenizers/monolingual/lug_latn_5mb/* models/5mb/lug_latn_5mb

# luo_latn
if test -f models/5mb/luo_latn_5mb/pytorch_model.bin; then
echo "Model already found: luo_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/luo_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/luo_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=361 --save_steps=999999999 \
--max_steps=7220 \
--warmup_steps=722 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/luo_latn_5mb.txt \
--seed=43 \
--override_n_examples=2888 \
--output_dir=models/5mb/luo_latn_5mb
cp tokenizers/monolingual/luo_latn_5mb/* models/5mb/luo_latn_5mb

# lus_latn
if test -f models/5mb/lus_latn_5mb/pytorch_model.bin; then
echo "Model already found: lus_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lus_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lus_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=373 --save_steps=999999999 \
--max_steps=7462 \
--warmup_steps=746 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lus_latn_5mb.txt \
--seed=43 \
--override_n_examples=2985 \
--output_dir=models/5mb/lus_latn_5mb
cp tokenizers/monolingual/lus_latn_5mb/* models/5mb/lus_latn_5mb

# lvs_latn
if test -f models/5mb/lvs_latn_5mb/pytorch_model.bin; then
echo "Model already found: lvs_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lvs_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lvs_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=305 --save_steps=999999999 \
--max_steps=6107 \
--warmup_steps=610 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lvs_latn_5mb.txt \
--seed=43 \
--override_n_examples=2443 \
--output_dir=models/5mb/lvs_latn_5mb
cp tokenizers/monolingual/lvs_latn_5mb/* models/5mb/lvs_latn_5mb

# lzh_hant
if test -f models/5mb/lzh_hant_5mb/pytorch_model.bin; then
echo "Model already found: lzh_hant_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lzh_hant_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lzh_hant_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=228 --save_steps=999999999 \
--max_steps=4562 \
--warmup_steps=456 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lzh_hant_5mb.txt \
--seed=43 \
--override_n_examples=1825 \
--output_dir=models/5mb/lzh_hant_5mb
cp tokenizers/monolingual/lzh_hant_5mb/* models/5mb/lzh_hant_5mb

# mad_latn
if test -f models/5mb/mad_latn_5mb/pytorch_model.bin; then
echo "Model already found: mad_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mad_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mad_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6952 \
--warmup_steps=695 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mad_latn_5mb.txt \
--seed=43 \
--override_n_examples=2781 \
--output_dir=models/5mb/mad_latn_5mb
cp tokenizers/monolingual/mad_latn_5mb/* models/5mb/mad_latn_5mb

# mag_deva
if test -f models/5mb/mag_deva_5mb/pytorch_model.bin; then
echo "Model already found: mag_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mag_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mag_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=342 --save_steps=999999999 \
--max_steps=6840 \
--warmup_steps=684 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mag_deva_5mb.txt \
--seed=43 \
--override_n_examples=2736 \
--output_dir=models/5mb/mag_deva_5mb
cp tokenizers/monolingual/mag_deva_5mb/* models/5mb/mag_deva_5mb

# mai_deva
if test -f models/5mb/mai_deva_5mb/pytorch_model.bin; then
echo "Model already found: mai_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mai_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mai_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=338 --save_steps=999999999 \
--max_steps=6777 \
--warmup_steps=677 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mai_deva_5mb.txt \
--seed=43 \
--override_n_examples=2711 \
--output_dir=models/5mb/mai_deva_5mb
cp tokenizers/monolingual/mai_deva_5mb/* models/5mb/mai_deva_5mb

# mal_latn
if test -f models/5mb/mal_latn_5mb/pytorch_model.bin; then
echo "Model already found: mal_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mal_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mal_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6032 \
--warmup_steps=603 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mal_latn_5mb.txt \
--seed=43 \
--override_n_examples=2413 \
--output_dir=models/5mb/mal_latn_5mb
cp tokenizers/monolingual/mal_latn_5mb/* models/5mb/mal_latn_5mb

# mal_mlym
if test -f models/5mb/mal_mlym_5mb/pytorch_model.bin; then
echo "Model already found: mal_mlym_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mal_mlym_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mal_mlym_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=315 --save_steps=999999999 \
--max_steps=6305 \
--warmup_steps=630 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mal_mlym_5mb.txt \
--seed=43 \
--override_n_examples=2522 \
--output_dir=models/5mb/mal_mlym_5mb
cp tokenizers/monolingual/mal_mlym_5mb/* models/5mb/mal_mlym_5mb

# mam_latn
if test -f models/5mb/mam_latn_5mb/pytorch_model.bin; then
echo "Model already found: mam_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mam_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mam_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=478 --save_steps=999999999 \
--max_steps=9572 \
--warmup_steps=957 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mam_latn_5mb.txt \
--seed=43 \
--override_n_examples=3829 \
--output_dir=models/5mb/mam_latn_5mb
cp tokenizers/monolingual/mam_latn_5mb/* models/5mb/mam_latn_5mb

# mar_deva
if test -f models/5mb/mar_deva_5mb/pytorch_model.bin; then
echo "Model already found: mar_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mar_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mar_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=271 --save_steps=999999999 \
--max_steps=5420 \
--warmup_steps=542 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mar_deva_5mb.txt \
--seed=43 \
--override_n_examples=2168 \
--output_dir=models/5mb/mar_deva_5mb
cp tokenizers/monolingual/mar_deva_5mb/* models/5mb/mar_deva_5mb

# mdf_cyrl
if test -f models/5mb/mdf_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: mdf_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mdf_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mdf_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5227 \
--warmup_steps=522 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mdf_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2091 \
--output_dir=models/5mb/mdf_cyrl_5mb
cp tokenizers/monolingual/mdf_cyrl_5mb/* models/5mb/mdf_cyrl_5mb

# meo_latn
if test -f models/5mb/meo_latn_5mb/pytorch_model.bin; then
echo "Model already found: meo_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/meo_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/meo_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7532 \
--warmup_steps=753 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/meo_latn_5mb.txt \
--seed=43 \
--override_n_examples=3013 \
--output_dir=models/5mb/meo_latn_5mb
cp tokenizers/monolingual/meo_latn_5mb/* models/5mb/meo_latn_5mb

# mgh_latn
if test -f models/5mb/mgh_latn_5mb/pytorch_model.bin; then
echo "Model already found: mgh_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mgh_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mgh_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5360 \
--warmup_steps=536 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mgh_latn_5mb.txt \
--seed=43 \
--override_n_examples=2144 \
--output_dir=models/5mb/mgh_latn_5mb
cp tokenizers/monolingual/mgh_latn_5mb/* models/5mb/mgh_latn_5mb

# mhr_cyrl
if test -f models/5mb/mhr_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: mhr_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mhr_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mhr_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=298 --save_steps=999999999 \
--max_steps=5975 \
--warmup_steps=597 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mhr_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2390 \
--output_dir=models/5mb/mhr_cyrl_5mb
cp tokenizers/monolingual/mhr_cyrl_5mb/* models/5mb/mhr_cyrl_5mb

# min_latn
if test -f models/5mb/min_latn_5mb/pytorch_model.bin; then
echo "Model already found: min_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/min_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/min_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=304 --save_steps=999999999 \
--max_steps=6097 \
--warmup_steps=609 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/min_latn_5mb.txt \
--seed=43 \
--override_n_examples=2439 \
--output_dir=models/5mb/min_latn_5mb
cp tokenizers/monolingual/min_latn_5mb/* models/5mb/min_latn_5mb

# mkd_cyrl
if test -f models/5mb/mkd_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: mkd_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mkd_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mkd_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=285 --save_steps=999999999 \
--max_steps=5702 \
--warmup_steps=570 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mkd_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2281 \
--output_dir=models/5mb/mkd_cyrl_5mb
cp tokenizers/monolingual/mkd_cyrl_5mb/* models/5mb/mkd_cyrl_5mb

# mkw_cyrl
if test -f models/5mb/mkw_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: mkw_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mkw_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mkw_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=301 --save_steps=999999999 \
--max_steps=6037 \
--warmup_steps=603 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mkw_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2415 \
--output_dir=models/5mb/mkw_cyrl_5mb
cp tokenizers/monolingual/mkw_cyrl_5mb/* models/5mb/mkw_cyrl_5mb

# mlg_latn
if test -f models/5mb/mlg_latn_5mb/pytorch_model.bin; then
echo "Model already found: mlg_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mlg_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mlg_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=372 --save_steps=999999999 \
--max_steps=7450 \
--warmup_steps=745 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mlg_latn_5mb.txt \
--seed=43 \
--override_n_examples=2980 \
--output_dir=models/5mb/mlg_latn_5mb
cp tokenizers/monolingual/mlg_latn_5mb/* models/5mb/mlg_latn_5mb

# mlt_latn
if test -f models/5mb/mlt_latn_5mb/pytorch_model.bin; then
echo "Model already found: mlt_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mlt_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mlt_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=363 --save_steps=999999999 \
--max_steps=7267 \
--warmup_steps=726 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mlt_latn_5mb.txt \
--seed=43 \
--override_n_examples=2907 \
--output_dir=models/5mb/mlt_latn_5mb
cp tokenizers/monolingual/mlt_latn_5mb/* models/5mb/mlt_latn_5mb

# mon_cyrl
if test -f models/5mb/mon_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: mon_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mon_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mon_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=269 --save_steps=999999999 \
--max_steps=5382 \
--warmup_steps=538 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mon_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2153 \
--output_dir=models/5mb/mon_cyrl_5mb
cp tokenizers/monolingual/mon_cyrl_5mb/* models/5mb/mon_cyrl_5mb

# mon_latn
if test -f models/5mb/mon_latn_5mb/pytorch_model.bin; then
echo "Model already found: mon_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mon_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mon_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=331 --save_steps=999999999 \
--max_steps=6627 \
--warmup_steps=662 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mon_latn_5mb.txt \
--seed=43 \
--override_n_examples=2651 \
--output_dir=models/5mb/mon_latn_5mb
cp tokenizers/monolingual/mon_latn_5mb/* models/5mb/mon_latn_5mb

# mos_latn
if test -f models/5mb/mos_latn_5mb/pytorch_model.bin; then
echo "Model already found: mos_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mos_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mos_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=423 --save_steps=999999999 \
--max_steps=8460 \
--warmup_steps=846 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mos_latn_5mb.txt \
--seed=43 \
--override_n_examples=3384 \
--output_dir=models/5mb/mos_latn_5mb
cp tokenizers/monolingual/mos_latn_5mb/* models/5mb/mos_latn_5mb

# mri_latn
if test -f models/5mb/mri_latn_5mb/pytorch_model.bin; then
echo "Model already found: mri_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mri_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mri_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=381 --save_steps=999999999 \
--max_steps=7635 \
--warmup_steps=763 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mri_latn_5mb.txt \
--seed=43 \
--override_n_examples=3054 \
--output_dir=models/5mb/mri_latn_5mb
cp tokenizers/monolingual/mri_latn_5mb/* models/5mb/mri_latn_5mb

# mrj_cyrl
if test -f models/5mb/mrj_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: mrj_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mrj_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mrj_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5177 \
--warmup_steps=517 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mrj_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2071 \
--output_dir=models/5mb/mrj_cyrl_5mb
cp tokenizers/monolingual/mrj_cyrl_5mb/* models/5mb/mrj_cyrl_5mb

# msa_latn
if test -f models/5mb/msa_latn_5mb/pytorch_model.bin; then
echo "Model already found: msa_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/msa_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/msa_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=302 --save_steps=999999999 \
--max_steps=6055 \
--warmup_steps=605 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/msa_latn_5mb.txt \
--seed=43 \
--override_n_examples=2422 \
--output_dir=models/5mb/msa_latn_5mb
cp tokenizers/monolingual/msa_latn_5mb/* models/5mb/msa_latn_5mb

# mwl_latn
if test -f models/5mb/mwl_latn_5mb/pytorch_model.bin; then
echo "Model already found: mwl_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mwl_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mwl_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6977 \
--warmup_steps=697 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mwl_latn_5mb.txt \
--seed=43 \
--override_n_examples=2791 \
--output_dir=models/5mb/mwl_latn_5mb
cp tokenizers/monolingual/mwl_latn_5mb/* models/5mb/mwl_latn_5mb

# mya_mymr
if test -f models/5mb/mya_mymr_5mb/pytorch_model.bin; then
echo "Model already found: mya_mymr_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mya_mymr_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mya_mymr_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=521 --save_steps=999999999 \
--max_steps=10420 \
--warmup_steps=1042 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mya_mymr_5mb.txt \
--seed=43 \
--override_n_examples=4168 \
--output_dir=models/5mb/mya_mymr_5mb
cp tokenizers/monolingual/mya_mymr_5mb/* models/5mb/mya_mymr_5mb

# myv_cyrl
if test -f models/5mb/myv_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: myv_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/myv_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/myv_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=289 --save_steps=999999999 \
--max_steps=5792 \
--warmup_steps=579 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/myv_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2317 \
--output_dir=models/5mb/myv_cyrl_5mb
cp tokenizers/monolingual/myv_cyrl_5mb/* models/5mb/myv_cyrl_5mb

# nan_latn
if test -f models/5mb/nan_latn_5mb/pytorch_model.bin; then
echo "Model already found: nan_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nan_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nan_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=467 --save_steps=999999999 \
--max_steps=9357 \
--warmup_steps=935 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nan_latn_5mb.txt \
--seed=43 \
--override_n_examples=3743 \
--output_dir=models/5mb/nan_latn_5mb
cp tokenizers/monolingual/nan_latn_5mb/* models/5mb/nan_latn_5mb

# nap_latn
if test -f models/5mb/nap_latn_5mb/pytorch_model.bin; then
echo "Model already found: nap_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nap_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nap_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7767 \
--warmup_steps=776 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nap_latn_5mb.txt \
--seed=43 \
--override_n_examples=3107 \
--output_dir=models/5mb/nap_latn_5mb
cp tokenizers/monolingual/nap_latn_5mb/* models/5mb/nap_latn_5mb

# nde_latn
if test -f models/5mb/nde_latn_5mb/pytorch_model.bin; then
echo "Model already found: nde_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nde_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nde_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4410 \
--warmup_steps=441 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nde_latn_5mb.txt \
--seed=43 \
--override_n_examples=1764 \
--output_dir=models/5mb/nde_latn_5mb
cp tokenizers/monolingual/nde_latn_5mb/* models/5mb/nde_latn_5mb

# nds_latn
if test -f models/5mb/nds_latn_5mb/pytorch_model.bin; then
echo "Model already found: nds_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nds_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nds_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=344 --save_steps=999999999 \
--max_steps=6890 \
--warmup_steps=689 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nds_latn_5mb.txt \
--seed=43 \
--override_n_examples=2756 \
--output_dir=models/5mb/nds_latn_5mb
cp tokenizers/monolingual/nds_latn_5mb/* models/5mb/nds_latn_5mb

# nep_deva
if test -f models/5mb/nep_deva_5mb/pytorch_model.bin; then
echo "Model already found: nep_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nep_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nep_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=283 --save_steps=999999999 \
--max_steps=5665 \
--warmup_steps=566 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nep_deva_5mb.txt \
--seed=43 \
--override_n_examples=2266 \
--output_dir=models/5mb/nep_deva_5mb
cp tokenizers/monolingual/nep_deva_5mb/* models/5mb/nep_deva_5mb

# new_deva
if test -f models/5mb/new_deva_5mb/pytorch_model.bin; then
echo "Model already found: new_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/new_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/new_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=298 --save_steps=999999999 \
--max_steps=5967 \
--warmup_steps=596 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/new_deva_5mb.txt \
--seed=43 \
--override_n_examples=2387 \
--output_dir=models/5mb/new_deva_5mb
cp tokenizers/monolingual/new_deva_5mb/* models/5mb/new_deva_5mb

# ngu_latn
if test -f models/5mb/ngu_latn_5mb/pytorch_model.bin; then
echo "Model already found: ngu_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ngu_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ngu_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6220 \
--warmup_steps=622 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ngu_latn_5mb.txt \
--seed=43 \
--override_n_examples=2488 \
--output_dir=models/5mb/ngu_latn_5mb
cp tokenizers/monolingual/ngu_latn_5mb/* models/5mb/ngu_latn_5mb

# nhe_latn
if test -f models/5mb/nhe_latn_5mb/pytorch_model.bin; then
echo "Model already found: nhe_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nhe_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nhe_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6182 \
--warmup_steps=618 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nhe_latn_5mb.txt \
--seed=43 \
--override_n_examples=2473 \
--output_dir=models/5mb/nhe_latn_5mb
cp tokenizers/monolingual/nhe_latn_5mb/* models/5mb/nhe_latn_5mb

# nld_latn
if test -f models/5mb/nld_latn_5mb/pytorch_model.bin; then
echo "Model already found: nld_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nld_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nld_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=276 --save_steps=999999999 \
--max_steps=5520 \
--warmup_steps=552 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nld_latn_5mb.txt \
--seed=43 \
--override_n_examples=2208 \
--output_dir=models/5mb/nld_latn_5mb
cp tokenizers/monolingual/nld_latn_5mb/* models/5mb/nld_latn_5mb

# nnb_latn
if test -f models/5mb/nnb_latn_5mb/pytorch_model.bin; then
echo "Model already found: nnb_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nnb_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nnb_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6732 \
--warmup_steps=673 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nnb_latn_5mb.txt \
--seed=43 \
--override_n_examples=2693 \
--output_dir=models/5mb/nnb_latn_5mb
cp tokenizers/monolingual/nnb_latn_5mb/* models/5mb/nnb_latn_5mb

# nno_latn
if test -f models/5mb/nno_latn_5mb/pytorch_model.bin; then
echo "Model already found: nno_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nno_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nno_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=297 --save_steps=999999999 \
--max_steps=5950 \
--warmup_steps=595 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nno_latn_5mb.txt \
--seed=43 \
--override_n_examples=2380 \
--output_dir=models/5mb/nno_latn_5mb
cp tokenizers/monolingual/nno_latn_5mb/* models/5mb/nno_latn_5mb

# nob_latn
if test -f models/5mb/nob_latn_5mb/pytorch_model.bin; then
echo "Model already found: nob_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nob_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nob_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=261 --save_steps=999999999 \
--max_steps=5227 \
--warmup_steps=522 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nob_latn_5mb.txt \
--seed=43 \
--override_n_examples=2091 \
--output_dir=models/5mb/nob_latn_5mb
cp tokenizers/monolingual/nob_latn_5mb/* models/5mb/nob_latn_5mb

# nor_latn
if test -f models/5mb/nor_latn_5mb/pytorch_model.bin; then
echo "Model already found: nor_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nor_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nor_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=322 --save_steps=999999999 \
--max_steps=6455 \
--warmup_steps=645 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nor_latn_5mb.txt \
--seed=43 \
--override_n_examples=2582 \
--output_dir=models/5mb/nor_latn_5mb
cp tokenizers/monolingual/nor_latn_5mb/* models/5mb/nor_latn_5mb

# nso_latn
if test -f models/5mb/nso_latn_5mb/pytorch_model.bin; then
echo "Model already found: nso_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nso_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nso_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=373 --save_steps=999999999 \
--max_steps=7477 \
--warmup_steps=747 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nso_latn_5mb.txt \
--seed=43 \
--override_n_examples=2991 \
--output_dir=models/5mb/nso_latn_5mb
cp tokenizers/monolingual/nso_latn_5mb/* models/5mb/nso_latn_5mb

# nya_latn
if test -f models/5mb/nya_latn_5mb/pytorch_model.bin; then
echo "Model already found: nya_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nya_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nya_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=324 --save_steps=999999999 \
--max_steps=6487 \
--warmup_steps=648 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nya_latn_5mb.txt \
--seed=43 \
--override_n_examples=2595 \
--output_dir=models/5mb/nya_latn_5mb
cp tokenizers/monolingual/nya_latn_5mb/* models/5mb/nya_latn_5mb

# nzi_latn
if test -f models/5mb/nzi_latn_5mb/pytorch_model.bin; then
echo "Model already found: nzi_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nzi_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nzi_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7367 \
--warmup_steps=736 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nzi_latn_5mb.txt \
--seed=43 \
--override_n_examples=2947 \
--output_dir=models/5mb/nzi_latn_5mb
cp tokenizers/monolingual/nzi_latn_5mb/* models/5mb/nzi_latn_5mb

# oci_latn
if test -f models/5mb/oci_latn_5mb/pytorch_model.bin; then
echo "Model already found: oci_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/oci_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/oci_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=340 --save_steps=999999999 \
--max_steps=6800 \
--warmup_steps=680 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/oci_latn_5mb.txt \
--seed=43 \
--override_n_examples=2720 \
--output_dir=models/5mb/oci_latn_5mb
cp tokenizers/monolingual/oci_latn_5mb/* models/5mb/oci_latn_5mb

# ori_orya
if test -f models/5mb/ori_orya_5mb/pytorch_model.bin; then
echo "Model already found: ori_orya_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ori_orya_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ori_orya_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=275 --save_steps=999999999 \
--max_steps=5507 \
--warmup_steps=550 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ori_orya_5mb.txt \
--seed=43 \
--override_n_examples=2203 \
--output_dir=models/5mb/ori_orya_5mb
cp tokenizers/monolingual/ori_orya_5mb/* models/5mb/ori_orya_5mb

# orm_latn
if test -f models/5mb/orm_latn_5mb/pytorch_model.bin; then
echo "Model already found: orm_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/orm_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/orm_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=366 --save_steps=999999999 \
--max_steps=7335 \
--warmup_steps=733 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/orm_latn_5mb.txt \
--seed=43 \
--override_n_examples=2934 \
--output_dir=models/5mb/orm_latn_5mb
cp tokenizers/monolingual/orm_latn_5mb/* models/5mb/orm_latn_5mb

# oss_cyrl
if test -f models/5mb/oss_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: oss_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/oss_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/oss_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=455 --save_steps=999999999 \
--max_steps=9100 \
--warmup_steps=910 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/oss_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=3640 \
--output_dir=models/5mb/oss_cyrl_5mb
cp tokenizers/monolingual/oss_cyrl_5mb/* models/5mb/oss_cyrl_5mb

# otq_latn
if test -f models/5mb/otq_latn_5mb/pytorch_model.bin; then
echo "Model already found: otq_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/otq_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/otq_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=409 --save_steps=999999999 \
--max_steps=8195 \
--warmup_steps=819 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/otq_latn_5mb.txt \
--seed=43 \
--override_n_examples=3278 \
--output_dir=models/5mb/otq_latn_5mb
cp tokenizers/monolingual/otq_latn_5mb/* models/5mb/otq_latn_5mb

# pag_latn
if test -f models/5mb/pag_latn_5mb/pytorch_model.bin; then
echo "Model already found: pag_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pag_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pag_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=363 --save_steps=999999999 \
--max_steps=7262 \
--warmup_steps=726 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pag_latn_5mb.txt \
--seed=43 \
--override_n_examples=2905 \
--output_dir=models/5mb/pag_latn_5mb
cp tokenizers/monolingual/pag_latn_5mb/* models/5mb/pag_latn_5mb

# pam_latn
if test -f models/5mb/pam_latn_5mb/pytorch_model.bin; then
echo "Model already found: pam_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pam_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pam_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=381 --save_steps=999999999 \
--max_steps=7635 \
--warmup_steps=763 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pam_latn_5mb.txt \
--seed=43 \
--override_n_examples=3054 \
--output_dir=models/5mb/pam_latn_5mb
cp tokenizers/monolingual/pam_latn_5mb/* models/5mb/pam_latn_5mb

# pan_guru
if test -f models/5mb/pan_guru_5mb/pytorch_model.bin; then
echo "Model already found: pan_guru_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pan_guru_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pan_guru_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=280 --save_steps=999999999 \
--max_steps=5612 \
--warmup_steps=561 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pan_guru_5mb.txt \
--seed=43 \
--override_n_examples=2245 \
--output_dir=models/5mb/pan_guru_5mb
cp tokenizers/monolingual/pan_guru_5mb/* models/5mb/pan_guru_5mb

# pap_latn
if test -f models/5mb/pap_latn_5mb/pytorch_model.bin; then
echo "Model already found: pap_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pap_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pap_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=298 --save_steps=999999999 \
--max_steps=5970 \
--warmup_steps=597 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pap_latn_5mb.txt \
--seed=43 \
--override_n_examples=2388 \
--output_dir=models/5mb/pap_latn_5mb
cp tokenizers/monolingual/pap_latn_5mb/* models/5mb/pap_latn_5mb

# pbt_arab
if test -f models/5mb/pbt_arab_5mb/pytorch_model.bin; then
echo "Model already found: pbt_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pbt_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pbt_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=349 --save_steps=999999999 \
--max_steps=6990 \
--warmup_steps=699 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pbt_arab_5mb.txt \
--seed=43 \
--override_n_examples=2796 \
--output_dir=models/5mb/pbt_arab_5mb
cp tokenizers/monolingual/pbt_arab_5mb/* models/5mb/pbt_arab_5mb

# pck_latn
if test -f models/5mb/pck_latn_5mb/pytorch_model.bin; then
echo "Model already found: pck_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pck_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pck_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7747 \
--warmup_steps=774 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pck_latn_5mb.txt \
--seed=43 \
--override_n_examples=3099 \
--output_dir=models/5mb/pck_latn_5mb
cp tokenizers/monolingual/pck_latn_5mb/* models/5mb/pck_latn_5mb

# pcm_latn
if test -f models/5mb/pcm_latn_5mb/pytorch_model.bin; then
echo "Model already found: pcm_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pcm_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pcm_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=273 --save_steps=999999999 \
--max_steps=5467 \
--warmup_steps=546 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pcm_latn_5mb.txt \
--seed=43 \
--override_n_examples=2187 \
--output_dir=models/5mb/pcm_latn_5mb
cp tokenizers/monolingual/pcm_latn_5mb/* models/5mb/pcm_latn_5mb

# pes_arab
if test -f models/5mb/pes_arab_5mb/pytorch_model.bin; then
echo "Model already found: pes_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pes_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pes_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=278 --save_steps=999999999 \
--max_steps=5575 \
--warmup_steps=557 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pes_arab_5mb.txt \
--seed=43 \
--override_n_examples=2230 \
--output_dir=models/5mb/pes_arab_5mb
cp tokenizers/monolingual/pes_arab_5mb/* models/5mb/pes_arab_5mb

# plt_latn
if test -f models/5mb/plt_latn_5mb/pytorch_model.bin; then
echo "Model already found: plt_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/plt_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/plt_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=334 --save_steps=999999999 \
--max_steps=6692 \
--warmup_steps=669 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/plt_latn_5mb.txt \
--seed=43 \
--override_n_examples=2677 \
--output_dir=models/5mb/plt_latn_5mb
cp tokenizers/monolingual/plt_latn_5mb/* models/5mb/plt_latn_5mb

# pms_latn
if test -f models/5mb/pms_latn_5mb/pytorch_model.bin; then
echo "Model already found: pms_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pms_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pms_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=397 --save_steps=999999999 \
--max_steps=7952 \
--warmup_steps=795 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pms_latn_5mb.txt \
--seed=43 \
--override_n_examples=3181 \
--output_dir=models/5mb/pms_latn_5mb
cp tokenizers/monolingual/pms_latn_5mb/* models/5mb/pms_latn_5mb

# pnb_arab
if test -f models/5mb/pnb_arab_5mb/pytorch_model.bin; then
echo "Model already found: pnb_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pnb_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pnb_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=321 --save_steps=999999999 \
--max_steps=6432 \
--warmup_steps=643 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pnb_arab_5mb.txt \
--seed=43 \
--override_n_examples=2573 \
--output_dir=models/5mb/pnb_arab_5mb
cp tokenizers/monolingual/pnb_arab_5mb/* models/5mb/pnb_arab_5mb

# pol_latn
if test -f models/5mb/pol_latn_5mb/pytorch_model.bin; then
echo "Model already found: pol_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pol_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pol_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=280 --save_steps=999999999 \
--max_steps=5612 \
--warmup_steps=561 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pol_latn_5mb.txt \
--seed=43 \
--override_n_examples=2245 \
--output_dir=models/5mb/pol_latn_5mb
cp tokenizers/monolingual/pol_latn_5mb/* models/5mb/pol_latn_5mb

# pon_latn
if test -f models/5mb/pon_latn_5mb/pytorch_model.bin; then
echo "Model already found: pon_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pon_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pon_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4960 \
--warmup_steps=496 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pon_latn_5mb.txt \
--seed=43 \
--override_n_examples=1984 \
--output_dir=models/5mb/pon_latn_5mb
cp tokenizers/monolingual/pon_latn_5mb/* models/5mb/pon_latn_5mb

# por_latn
if test -f models/5mb/por_latn_5mb/pytorch_model.bin; then
echo "Model already found: por_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/por_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/por_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=285 --save_steps=999999999 \
--max_steps=5700 \
--warmup_steps=570 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/por_latn_5mb.txt \
--seed=43 \
--override_n_examples=2280 \
--output_dir=models/5mb/por_latn_5mb
cp tokenizers/monolingual/por_latn_5mb/* models/5mb/por_latn_5mb

# prs_arab
if test -f models/5mb/prs_arab_5mb/pytorch_model.bin; then
echo "Model already found: prs_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/prs_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/prs_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=294 --save_steps=999999999 \
--max_steps=5897 \
--warmup_steps=589 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/prs_arab_5mb.txt \
--seed=43 \
--override_n_examples=2359 \
--output_dir=models/5mb/prs_arab_5mb
cp tokenizers/monolingual/prs_arab_5mb/* models/5mb/prs_arab_5mb

# pus_arab
if test -f models/5mb/pus_arab_5mb/pytorch_model.bin; then
echo "Model already found: pus_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pus_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pus_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=310 --save_steps=999999999 \
--max_steps=6207 \
--warmup_steps=620 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pus_arab_5mb.txt \
--seed=43 \
--override_n_examples=2483 \
--output_dir=models/5mb/pus_arab_5mb
cp tokenizers/monolingual/pus_arab_5mb/* models/5mb/pus_arab_5mb

# que_latn
if test -f models/5mb/que_latn_5mb/pytorch_model.bin; then
echo "Model already found: que_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/que_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/que_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=373 --save_steps=999999999 \
--max_steps=7472 \
--warmup_steps=747 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/que_latn_5mb.txt \
--seed=43 \
--override_n_examples=2989 \
--output_dir=models/5mb/que_latn_5mb
cp tokenizers/monolingual/que_latn_5mb/* models/5mb/que_latn_5mb

# quy_latn
if test -f models/5mb/quy_latn_5mb/pytorch_model.bin; then
echo "Model already found: quy_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/quy_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/quy_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=363 --save_steps=999999999 \
--max_steps=7262 \
--warmup_steps=726 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/quy_latn_5mb.txt \
--seed=43 \
--override_n_examples=2905 \
--output_dir=models/5mb/quy_latn_5mb
cp tokenizers/monolingual/quy_latn_5mb/* models/5mb/quy_latn_5mb

# quz_latn
if test -f models/5mb/quz_latn_5mb/pytorch_model.bin; then
echo "Model already found: quz_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/quz_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/quz_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5585 \
--warmup_steps=558 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/quz_latn_5mb.txt \
--seed=43 \
--override_n_examples=2234 \
--output_dir=models/5mb/quz_latn_5mb
cp tokenizers/monolingual/quz_latn_5mb/* models/5mb/quz_latn_5mb

# rmc_latn
if test -f models/5mb/rmc_latn_5mb/pytorch_model.bin; then
echo "Model already found: rmc_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rmc_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rmc_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5432 \
--warmup_steps=543 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rmc_latn_5mb.txt \
--seed=43 \
--override_n_examples=2173 \
--output_dir=models/5mb/rmc_latn_5mb
cp tokenizers/monolingual/rmc_latn_5mb/* models/5mb/rmc_latn_5mb

# roh_latn
if test -f models/5mb/roh_latn_5mb/pytorch_model.bin; then
echo "Model already found: roh_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/roh_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/roh_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=348 --save_steps=999999999 \
--max_steps=6972 \
--warmup_steps=697 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/roh_latn_5mb.txt \
--seed=43 \
--override_n_examples=2789 \
--output_dir=models/5mb/roh_latn_5mb
cp tokenizers/monolingual/roh_latn_5mb/* models/5mb/roh_latn_5mb

# ron_latn
if test -f models/5mb/ron_latn_5mb/pytorch_model.bin; then
echo "Model already found: ron_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ron_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ron_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=297 --save_steps=999999999 \
--max_steps=5942 \
--warmup_steps=594 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ron_latn_5mb.txt \
--seed=43 \
--override_n_examples=2377 \
--output_dir=models/5mb/ron_latn_5mb
cp tokenizers/monolingual/ron_latn_5mb/* models/5mb/ron_latn_5mb

# rue_cyrl
if test -f models/5mb/rue_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: rue_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rue_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rue_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5655 \
--warmup_steps=565 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rue_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2262 \
--output_dir=models/5mb/rue_cyrl_5mb
cp tokenizers/monolingual/rue_cyrl_5mb/* models/5mb/rue_cyrl_5mb

# run_latn
if test -f models/5mb/run_latn_5mb/pytorch_model.bin; then
echo "Model already found: run_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/run_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/run_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=339 --save_steps=999999999 \
--max_steps=6787 \
--warmup_steps=678 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/run_latn_5mb.txt \
--seed=43 \
--override_n_examples=2715 \
--output_dir=models/5mb/run_latn_5mb
cp tokenizers/monolingual/run_latn_5mb/* models/5mb/run_latn_5mb

# rus_cyrl
if test -f models/5mb/rus_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: rus_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rus_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rus_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=287 --save_steps=999999999 \
--max_steps=5740 \
--warmup_steps=574 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rus_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2296 \
--output_dir=models/5mb/rus_cyrl_5mb
cp tokenizers/monolingual/rus_cyrl_5mb/* models/5mb/rus_cyrl_5mb

# rus_latn
if test -f models/5mb/rus_latn_5mb/pytorch_model.bin; then
echo "Model already found: rus_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rus_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rus_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=352 --save_steps=999999999 \
--max_steps=7040 \
--warmup_steps=704 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rus_latn_5mb.txt \
--seed=43 \
--override_n_examples=2816 \
--output_dir=models/5mb/rus_latn_5mb
cp tokenizers/monolingual/rus_latn_5mb/* models/5mb/rus_latn_5mb

# sag_latn
if test -f models/5mb/sag_latn_5mb/pytorch_model.bin; then
echo "Model already found: sag_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sag_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sag_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=390 --save_steps=999999999 \
--max_steps=7812 \
--warmup_steps=781 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sag_latn_5mb.txt \
--seed=43 \
--override_n_examples=3125 \
--output_dir=models/5mb/sag_latn_5mb
cp tokenizers/monolingual/sag_latn_5mb/* models/5mb/sag_latn_5mb

# sah_cyrl
if test -f models/5mb/sah_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: sah_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sah_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sah_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=298 --save_steps=999999999 \
--max_steps=5962 \
--warmup_steps=596 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sah_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2385 \
--output_dir=models/5mb/sah_cyrl_5mb
cp tokenizers/monolingual/sah_cyrl_5mb/* models/5mb/sah_cyrl_5mb

# san_deva
if test -f models/5mb/san_deva_5mb/pytorch_model.bin; then
echo "Model already found: san_deva_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/san_deva_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/san_deva_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=345 --save_steps=999999999 \
--max_steps=6907 \
--warmup_steps=690 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/san_deva_5mb.txt \
--seed=43 \
--override_n_examples=2763 \
--output_dir=models/5mb/san_deva_5mb
cp tokenizers/monolingual/san_deva_5mb/* models/5mb/san_deva_5mb

# san_latn
if test -f models/5mb/san_latn_5mb/pytorch_model.bin; then
echo "Model already found: san_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/san_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/san_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4992 \
--warmup_steps=499 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/san_latn_5mb.txt \
--seed=43 \
--override_n_examples=1997 \
--output_dir=models/5mb/san_latn_5mb
cp tokenizers/monolingual/san_latn_5mb/* models/5mb/san_latn_5mb

# sat_olck
if test -f models/5mb/sat_olck_5mb/pytorch_model.bin; then
echo "Model already found: sat_olck_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sat_olck_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sat_olck_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6520 \
--warmup_steps=652 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sat_olck_5mb.txt \
--seed=43 \
--override_n_examples=2608 \
--output_dir=models/5mb/sat_olck_5mb
cp tokenizers/monolingual/sat_olck_5mb/* models/5mb/sat_olck_5mb

# scn_latn
if test -f models/5mb/scn_latn_5mb/pytorch_model.bin; then
echo "Model already found: scn_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/scn_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/scn_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=357 --save_steps=999999999 \
--max_steps=7157 \
--warmup_steps=715 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/scn_latn_5mb.txt \
--seed=43 \
--override_n_examples=2863 \
--output_dir=models/5mb/scn_latn_5mb
cp tokenizers/monolingual/scn_latn_5mb/* models/5mb/scn_latn_5mb

# sco_latn
if test -f models/5mb/sco_latn_5mb/pytorch_model.bin; then
echo "Model already found: sco_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sco_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sco_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=374 --save_steps=999999999 \
--max_steps=7480 \
--warmup_steps=748 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sco_latn_5mb.txt \
--seed=43 \
--override_n_examples=2992 \
--output_dir=models/5mb/sco_latn_5mb
cp tokenizers/monolingual/sco_latn_5mb/* models/5mb/sco_latn_5mb

# shn_mymr
if test -f models/5mb/shn_mymr_5mb/pytorch_model.bin; then
echo "Model already found: shn_mymr_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/shn_mymr_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/shn_mymr_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=242 --save_steps=999999999 \
--max_steps=4842 \
--warmup_steps=484 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/shn_mymr_5mb.txt \
--seed=43 \
--override_n_examples=1937 \
--output_dir=models/5mb/shn_mymr_5mb
cp tokenizers/monolingual/shn_mymr_5mb/* models/5mb/shn_mymr_5mb

# sin_sinh
if test -f models/5mb/sin_sinh_5mb/pytorch_model.bin; then
echo "Model already found: sin_sinh_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sin_sinh_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sin_sinh_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=297 --save_steps=999999999 \
--max_steps=5955 \
--warmup_steps=595 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sin_sinh_5mb.txt \
--seed=43 \
--override_n_examples=2382 \
--output_dir=models/5mb/sin_sinh_5mb
cp tokenizers/monolingual/sin_sinh_5mb/* models/5mb/sin_sinh_5mb

# slk_latn
if test -f models/5mb/slk_latn_5mb/pytorch_model.bin; then
echo "Model already found: slk_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/slk_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/slk_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=275 --save_steps=999999999 \
--max_steps=5502 \
--warmup_steps=550 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/slk_latn_5mb.txt \
--seed=43 \
--override_n_examples=2201 \
--output_dir=models/5mb/slk_latn_5mb
cp tokenizers/monolingual/slk_latn_5mb/* models/5mb/slk_latn_5mb

# slv_latn
if test -f models/5mb/slv_latn_5mb/pytorch_model.bin; then
echo "Model already found: slv_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/slv_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/slv_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=256 --save_steps=999999999 \
--max_steps=5132 \
--warmup_steps=513 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/slv_latn_5mb.txt \
--seed=43 \
--override_n_examples=2053 \
--output_dir=models/5mb/slv_latn_5mb
cp tokenizers/monolingual/slv_latn_5mb/* models/5mb/slv_latn_5mb

# sme_latn
if test -f models/5mb/sme_latn_5mb/pytorch_model.bin; then
echo "Model already found: sme_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sme_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sme_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=305 --save_steps=999999999 \
--max_steps=6105 \
--warmup_steps=610 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sme_latn_5mb.txt \
--seed=43 \
--override_n_examples=2442 \
--output_dir=models/5mb/sme_latn_5mb
cp tokenizers/monolingual/sme_latn_5mb/* models/5mb/sme_latn_5mb

# smo_latn
if test -f models/5mb/smo_latn_5mb/pytorch_model.bin; then
echo "Model already found: smo_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/smo_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/smo_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=404 --save_steps=999999999 \
--max_steps=8090 \
--warmup_steps=809 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/smo_latn_5mb.txt \
--seed=43 \
--override_n_examples=3236 \
--output_dir=models/5mb/smo_latn_5mb
cp tokenizers/monolingual/smo_latn_5mb/* models/5mb/smo_latn_5mb

# sna_latn
if test -f models/5mb/sna_latn_5mb/pytorch_model.bin; then
echo "Model already found: sna_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sna_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sna_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=323 --save_steps=999999999 \
--max_steps=6475 \
--warmup_steps=647 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sna_latn_5mb.txt \
--seed=43 \
--override_n_examples=2590 \
--output_dir=models/5mb/sna_latn_5mb
cp tokenizers/monolingual/sna_latn_5mb/* models/5mb/sna_latn_5mb

# snd_arab
if test -f models/5mb/snd_arab_5mb/pytorch_model.bin; then
echo "Model already found: snd_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/snd_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/snd_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=320 --save_steps=999999999 \
--max_steps=6412 \
--warmup_steps=641 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/snd_arab_5mb.txt \
--seed=43 \
--override_n_examples=2565 \
--output_dir=models/5mb/snd_arab_5mb
cp tokenizers/monolingual/snd_arab_5mb/* models/5mb/snd_arab_5mb

# som_latn
if test -f models/5mb/som_latn_5mb/pytorch_model.bin; then
echo "Model already found: som_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/som_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/som_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=386 --save_steps=999999999 \
--max_steps=7727 \
--warmup_steps=772 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/som_latn_5mb.txt \
--seed=43 \
--override_n_examples=3091 \
--output_dir=models/5mb/som_latn_5mb
cp tokenizers/monolingual/som_latn_5mb/* models/5mb/som_latn_5mb

# sot_latn
if test -f models/5mb/sot_latn_5mb/pytorch_model.bin; then
echo "Model already found: sot_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sot_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sot_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=358 --save_steps=999999999 \
--max_steps=7172 \
--warmup_steps=717 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sot_latn_5mb.txt \
--seed=43 \
--override_n_examples=2869 \
--output_dir=models/5mb/sot_latn_5mb
cp tokenizers/monolingual/sot_latn_5mb/* models/5mb/sot_latn_5mb

# spa_latn
if test -f models/5mb/spa_latn_5mb/pytorch_model.bin; then
echo "Model already found: spa_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/spa_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/spa_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=281 --save_steps=999999999 \
--max_steps=5637 \
--warmup_steps=563 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/spa_latn_5mb.txt \
--seed=43 \
--override_n_examples=2255 \
--output_dir=models/5mb/spa_latn_5mb
cp tokenizers/monolingual/spa_latn_5mb/* models/5mb/spa_latn_5mb

# sqi_latn
if test -f models/5mb/sqi_latn_5mb/pytorch_model.bin; then
echo "Model already found: sqi_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sqi_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sqi_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=351 --save_steps=999999999 \
--max_steps=7035 \
--warmup_steps=703 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sqi_latn_5mb.txt \
--seed=43 \
--override_n_examples=2814 \
--output_dir=models/5mb/sqi_latn_5mb
cp tokenizers/monolingual/sqi_latn_5mb/* models/5mb/sqi_latn_5mb

# srd_latn
if test -f models/5mb/srd_latn_5mb/pytorch_model.bin; then
echo "Model already found: srd_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srd_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srd_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=350 --save_steps=999999999 \
--max_steps=7012 \
--warmup_steps=701 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srd_latn_5mb.txt \
--seed=43 \
--override_n_examples=2805 \
--output_dir=models/5mb/srd_latn_5mb
cp tokenizers/monolingual/srd_latn_5mb/* models/5mb/srd_latn_5mb

# srn_latn
if test -f models/5mb/srn_latn_5mb/pytorch_model.bin; then
echo "Model already found: srn_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srn_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srn_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=334 --save_steps=999999999 \
--max_steps=6697 \
--warmup_steps=669 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srn_latn_5mb.txt \
--seed=43 \
--override_n_examples=2679 \
--output_dir=models/5mb/srn_latn_5mb
cp tokenizers/monolingual/srn_latn_5mb/* models/5mb/srn_latn_5mb

# srp_cyrl
if test -f models/5mb/srp_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: srp_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srp_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srp_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=240 --save_steps=999999999 \
--max_steps=4800 \
--warmup_steps=480 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srp_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=1920 \
--output_dir=models/5mb/srp_cyrl_5mb
cp tokenizers/monolingual/srp_cyrl_5mb/* models/5mb/srp_cyrl_5mb

# srp_latn
if test -f models/5mb/srp_latn_5mb/pytorch_model.bin; then
echo "Model already found: srp_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srp_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srp_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=262 --save_steps=999999999 \
--max_steps=5250 \
--warmup_steps=525 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srp_latn_5mb.txt \
--seed=43 \
--override_n_examples=2100 \
--output_dir=models/5mb/srp_latn_5mb
cp tokenizers/monolingual/srp_latn_5mb/* models/5mb/srp_latn_5mb

# ssw_latn
if test -f models/5mb/ssw_latn_5mb/pytorch_model.bin; then
echo "Model already found: ssw_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ssw_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ssw_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=337 --save_steps=999999999 \
--max_steps=6747 \
--warmup_steps=674 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ssw_latn_5mb.txt \
--seed=43 \
--override_n_examples=2699 \
--output_dir=models/5mb/ssw_latn_5mb
cp tokenizers/monolingual/ssw_latn_5mb/* models/5mb/ssw_latn_5mb

# sun_latn
if test -f models/5mb/sun_latn_5mb/pytorch_model.bin; then
echo "Model already found: sun_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sun_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sun_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=315 --save_steps=999999999 \
--max_steps=6310 \
--warmup_steps=631 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sun_latn_5mb.txt \
--seed=43 \
--override_n_examples=2524 \
--output_dir=models/5mb/sun_latn_5mb
cp tokenizers/monolingual/sun_latn_5mb/* models/5mb/sun_latn_5mb

# swa_latn
if test -f models/5mb/swa_latn_5mb/pytorch_model.bin; then
echo "Model already found: swa_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/swa_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/swa_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=336 --save_steps=999999999 \
--max_steps=6730 \
--warmup_steps=673 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/swa_latn_5mb.txt \
--seed=43 \
--override_n_examples=2692 \
--output_dir=models/5mb/swa_latn_5mb
cp tokenizers/monolingual/swa_latn_5mb/* models/5mb/swa_latn_5mb

# swe_latn
if test -f models/5mb/swe_latn_5mb/pytorch_model.bin; then
echo "Model already found: swe_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/swe_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/swe_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=264 --save_steps=999999999 \
--max_steps=5282 \
--warmup_steps=528 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/swe_latn_5mb.txt \
--seed=43 \
--override_n_examples=2113 \
--output_dir=models/5mb/swe_latn_5mb
cp tokenizers/monolingual/swe_latn_5mb/* models/5mb/swe_latn_5mb

# syr_syrc
if test -f models/5mb/syr_syrc_5mb/pytorch_model.bin; then
echo "Model already found: syr_syrc_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/syr_syrc_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/syr_syrc_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=277 --save_steps=999999999 \
--max_steps=5540 \
--warmup_steps=554 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/syr_syrc_5mb.txt \
--seed=43 \
--override_n_examples=2216 \
--output_dir=models/5mb/syr_syrc_5mb
cp tokenizers/monolingual/syr_syrc_5mb/* models/5mb/syr_syrc_5mb

# szl_latn
if test -f models/5mb/szl_latn_5mb/pytorch_model.bin; then
echo "Model already found: szl_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/szl_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/szl_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=360 --save_steps=999999999 \
--max_steps=7212 \
--warmup_steps=721 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/szl_latn_5mb.txt \
--seed=43 \
--override_n_examples=2885 \
--output_dir=models/5mb/szl_latn_5mb
cp tokenizers/monolingual/szl_latn_5mb/* models/5mb/szl_latn_5mb

# tam_latn
if test -f models/5mb/tam_latn_5mb/pytorch_model.bin; then
echo "Model already found: tam_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tam_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tam_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=317 --save_steps=999999999 \
--max_steps=6342 \
--warmup_steps=634 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tam_latn_5mb.txt \
--seed=43 \
--override_n_examples=2537 \
--output_dir=models/5mb/tam_latn_5mb
cp tokenizers/monolingual/tam_latn_5mb/* models/5mb/tam_latn_5mb

# tam_taml
if test -f models/5mb/tam_taml_5mb/pytorch_model.bin; then
echo "Model already found: tam_taml_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tam_taml_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tam_taml_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=260 --save_steps=999999999 \
--max_steps=5210 \
--warmup_steps=521 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tam_taml_5mb.txt \
--seed=43 \
--override_n_examples=2084 \
--output_dir=models/5mb/tam_taml_5mb
cp tokenizers/monolingual/tam_taml_5mb/* models/5mb/tam_taml_5mb

# tat_cyrl
if test -f models/5mb/tat_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: tat_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tat_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tat_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=298 --save_steps=999999999 \
--max_steps=5977 \
--warmup_steps=597 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tat_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2391 \
--output_dir=models/5mb/tat_cyrl_5mb
cp tokenizers/monolingual/tat_cyrl_5mb/* models/5mb/tat_cyrl_5mb

# tbz_latn
if test -f models/5mb/tbz_latn_5mb/pytorch_model.bin; then
echo "Model already found: tbz_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tbz_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tbz_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8960 \
--warmup_steps=896 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tbz_latn_5mb.txt \
--seed=43 \
--override_n_examples=3584 \
--output_dir=models/5mb/tbz_latn_5mb
cp tokenizers/monolingual/tbz_latn_5mb/* models/5mb/tbz_latn_5mb

# tcy_knda
if test -f models/5mb/tcy_knda_5mb/pytorch_model.bin; then
echo "Model already found: tcy_knda_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tcy_knda_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tcy_knda_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5630 \
--warmup_steps=563 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tcy_knda_5mb.txt \
--seed=43 \
--override_n_examples=2252 \
--output_dir=models/5mb/tcy_knda_5mb
cp tokenizers/monolingual/tcy_knda_5mb/* models/5mb/tcy_knda_5mb

# tdx_latn
if test -f models/5mb/tdx_latn_5mb/pytorch_model.bin; then
echo "Model already found: tdx_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tdx_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tdx_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6105 \
--warmup_steps=610 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tdx_latn_5mb.txt \
--seed=43 \
--override_n_examples=2442 \
--output_dir=models/5mb/tdx_latn_5mb
cp tokenizers/monolingual/tdx_latn_5mb/* models/5mb/tdx_latn_5mb

# tel_latn
if test -f models/5mb/tel_latn_5mb/pytorch_model.bin; then
echo "Model already found: tel_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tel_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tel_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=334 --save_steps=999999999 \
--max_steps=6682 \
--warmup_steps=668 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tel_latn_5mb.txt \
--seed=43 \
--override_n_examples=2673 \
--output_dir=models/5mb/tel_latn_5mb
cp tokenizers/monolingual/tel_latn_5mb/* models/5mb/tel_latn_5mb

# tel_telu
if test -f models/5mb/tel_telu_5mb/pytorch_model.bin; then
echo "Model already found: tel_telu_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tel_telu_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tel_telu_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=269 --save_steps=999999999 \
--max_steps=5392 \
--warmup_steps=539 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tel_telu_5mb.txt \
--seed=43 \
--override_n_examples=2157 \
--output_dir=models/5mb/tel_telu_5mb
cp tokenizers/monolingual/tel_telu_5mb/* models/5mb/tel_telu_5mb

# tet_latn
if test -f models/5mb/tet_latn_5mb/pytorch_model.bin; then
echo "Model already found: tet_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tet_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tet_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=371 --save_steps=999999999 \
--max_steps=7420 \
--warmup_steps=742 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tet_latn_5mb.txt \
--seed=43 \
--override_n_examples=2968 \
--output_dir=models/5mb/tet_latn_5mb
cp tokenizers/monolingual/tet_latn_5mb/* models/5mb/tet_latn_5mb

# tgk_cyrl
if test -f models/5mb/tgk_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: tgk_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tgk_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tgk_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=286 --save_steps=999999999 \
--max_steps=5730 \
--warmup_steps=573 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tgk_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2292 \
--output_dir=models/5mb/tgk_cyrl_5mb
cp tokenizers/monolingual/tgk_cyrl_5mb/* models/5mb/tgk_cyrl_5mb

# tgl_latn
if test -f models/5mb/tgl_latn_5mb/pytorch_model.bin; then
echo "Model already found: tgl_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tgl_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tgl_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=314 --save_steps=999999999 \
--max_steps=6282 \
--warmup_steps=628 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tgl_latn_5mb.txt \
--seed=43 \
--override_n_examples=2513 \
--output_dir=models/5mb/tgl_latn_5mb
cp tokenizers/monolingual/tgl_latn_5mb/* models/5mb/tgl_latn_5mb

# tha_thai
if test -f models/5mb/tha_thai_5mb/pytorch_model.bin; then
echo "Model already found: tha_thai_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tha_thai_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tha_thai_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=264 --save_steps=999999999 \
--max_steps=5280 \
--warmup_steps=528 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tha_thai_5mb.txt \
--seed=43 \
--override_n_examples=2112 \
--output_dir=models/5mb/tha_thai_5mb
cp tokenizers/monolingual/tha_thai_5mb/* models/5mb/tha_thai_5mb

# tir_ethi
if test -f models/5mb/tir_ethi_5mb/pytorch_model.bin; then
echo "Model already found: tir_ethi_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tir_ethi_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tir_ethi_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=289 --save_steps=999999999 \
--max_steps=5797 \
--warmup_steps=579 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tir_ethi_5mb.txt \
--seed=43 \
--override_n_examples=2319 \
--output_dir=models/5mb/tir_ethi_5mb
cp tokenizers/monolingual/tir_ethi_5mb/* models/5mb/tir_ethi_5mb

# tiv_latn
if test -f models/5mb/tiv_latn_5mb/pytorch_model.bin; then
echo "Model already found: tiv_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tiv_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tiv_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8270 \
--warmup_steps=827 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tiv_latn_5mb.txt \
--seed=43 \
--override_n_examples=3308 \
--output_dir=models/5mb/tiv_latn_5mb
cp tokenizers/monolingual/tiv_latn_5mb/* models/5mb/tiv_latn_5mb

# tlh_latn
if test -f models/5mb/tlh_latn_5mb/pytorch_model.bin; then
echo "Model already found: tlh_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tlh_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tlh_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7257 \
--warmup_steps=725 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tlh_latn_5mb.txt \
--seed=43 \
--override_n_examples=2903 \
--output_dir=models/5mb/tlh_latn_5mb
cp tokenizers/monolingual/tlh_latn_5mb/* models/5mb/tlh_latn_5mb

# ton_latn
if test -f models/5mb/ton_latn_5mb/pytorch_model.bin; then
echo "Model already found: ton_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ton_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ton_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=445 --save_steps=999999999 \
--max_steps=8912 \
--warmup_steps=891 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ton_latn_5mb.txt \
--seed=43 \
--override_n_examples=3565 \
--output_dir=models/5mb/ton_latn_5mb
cp tokenizers/monolingual/ton_latn_5mb/* models/5mb/ton_latn_5mb

# tpi_latn
if test -f models/5mb/tpi_latn_5mb/pytorch_model.bin; then
echo "Model already found: tpi_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tpi_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tpi_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=382 --save_steps=999999999 \
--max_steps=7655 \
--warmup_steps=765 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tpi_latn_5mb.txt \
--seed=43 \
--override_n_examples=3062 \
--output_dir=models/5mb/tpi_latn_5mb
cp tokenizers/monolingual/tpi_latn_5mb/* models/5mb/tpi_latn_5mb

# tsn_latn
if test -f models/5mb/tsn_latn_5mb/pytorch_model.bin; then
echo "Model already found: tsn_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tsn_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tsn_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=392 --save_steps=999999999 \
--max_steps=7845 \
--warmup_steps=784 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tsn_latn_5mb.txt \
--seed=43 \
--override_n_examples=3138 \
--output_dir=models/5mb/tsn_latn_5mb
cp tokenizers/monolingual/tsn_latn_5mb/* models/5mb/tsn_latn_5mb

# tso_latn
if test -f models/5mb/tso_latn_5mb/pytorch_model.bin; then
echo "Model already found: tso_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tso_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tso_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=383 --save_steps=999999999 \
--max_steps=7670 \
--warmup_steps=767 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tso_latn_5mb.txt \
--seed=43 \
--override_n_examples=3068 \
--output_dir=models/5mb/tso_latn_5mb
cp tokenizers/monolingual/tso_latn_5mb/* models/5mb/tso_latn_5mb

# tuk_latn
if test -f models/5mb/tuk_latn_5mb/pytorch_model.bin; then
echo "Model already found: tuk_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tuk_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tuk_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=394 --save_steps=999999999 \
--max_steps=7890 \
--warmup_steps=789 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tuk_latn_5mb.txt \
--seed=43 \
--override_n_examples=3156 \
--output_dir=models/5mb/tuk_latn_5mb
cp tokenizers/monolingual/tuk_latn_5mb/* models/5mb/tuk_latn_5mb

# tum_latn
if test -f models/5mb/tum_latn_5mb/pytorch_model.bin; then
echo "Model already found: tum_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tum_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tum_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=343 --save_steps=999999999 \
--max_steps=6877 \
--warmup_steps=687 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tum_latn_5mb.txt \
--seed=43 \
--override_n_examples=2751 \
--output_dir=models/5mb/tum_latn_5mb
cp tokenizers/monolingual/tum_latn_5mb/* models/5mb/tum_latn_5mb

# tur_latn
if test -f models/5mb/tur_latn_5mb/pytorch_model.bin; then
echo "Model already found: tur_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tur_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tur_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=242 --save_steps=999999999 \
--max_steps=4840 \
--warmup_steps=484 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tur_latn_5mb.txt \
--seed=43 \
--override_n_examples=1936 \
--output_dir=models/5mb/tur_latn_5mb
cp tokenizers/monolingual/tur_latn_5mb/* models/5mb/tur_latn_5mb

# twi_latn
if test -f models/5mb/twi_latn_5mb/pytorch_model.bin; then
echo "Model already found: twi_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/twi_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/twi_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=381 --save_steps=999999999 \
--max_steps=7620 \
--warmup_steps=762 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/twi_latn_5mb.txt \
--seed=43 \
--override_n_examples=3048 \
--output_dir=models/5mb/twi_latn_5mb
cp tokenizers/monolingual/twi_latn_5mb/* models/5mb/twi_latn_5mb

# tyv_cyrl
if test -f models/5mb/tyv_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: tyv_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tyv_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tyv_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=292 --save_steps=999999999 \
--max_steps=5840 \
--warmup_steps=584 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tyv_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2336 \
--output_dir=models/5mb/tyv_cyrl_5mb
cp tokenizers/monolingual/tyv_cyrl_5mb/* models/5mb/tyv_cyrl_5mb

# tzo_latn
if test -f models/5mb/tzo_latn_5mb/pytorch_model.bin; then
echo "Model already found: tzo_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tzo_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tzo_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=474 --save_steps=999999999 \
--max_steps=9495 \
--warmup_steps=949 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tzo_latn_5mb.txt \
--seed=43 \
--override_n_examples=3798 \
--output_dir=models/5mb/tzo_latn_5mb
cp tokenizers/monolingual/tzo_latn_5mb/* models/5mb/tzo_latn_5mb

# udm_cyrl
if test -f models/5mb/udm_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: udm_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/udm_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/udm_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=270 --save_steps=999999999 \
--max_steps=5402 \
--warmup_steps=540 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/udm_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2161 \
--output_dir=models/5mb/udm_cyrl_5mb
cp tokenizers/monolingual/udm_cyrl_5mb/* models/5mb/udm_cyrl_5mb

# uig_arab
if test -f models/5mb/uig_arab_5mb/pytorch_model.bin; then
echo "Model already found: uig_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uig_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uig_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=339 --save_steps=999999999 \
--max_steps=6780 \
--warmup_steps=678 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uig_arab_5mb.txt \
--seed=43 \
--override_n_examples=2712 \
--output_dir=models/5mb/uig_arab_5mb
cp tokenizers/monolingual/uig_arab_5mb/* models/5mb/uig_arab_5mb

# uig_latn
if test -f models/5mb/uig_latn_5mb/pytorch_model.bin; then
echo "Model already found: uig_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uig_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uig_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5252 \
--warmup_steps=525 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uig_latn_5mb.txt \
--seed=43 \
--override_n_examples=2101 \
--output_dir=models/5mb/uig_latn_5mb
cp tokenizers/monolingual/uig_latn_5mb/* models/5mb/uig_latn_5mb

# ukr_cyrl
if test -f models/5mb/ukr_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: ukr_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ukr_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ukr_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=279 --save_steps=999999999 \
--max_steps=5582 \
--warmup_steps=558 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ukr_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2233 \
--output_dir=models/5mb/ukr_cyrl_5mb
cp tokenizers/monolingual/ukr_cyrl_5mb/* models/5mb/ukr_cyrl_5mb

# umb_latn
if test -f models/5mb/umb_latn_5mb/pytorch_model.bin; then
echo "Model already found: umb_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/umb_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/umb_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=371 --save_steps=999999999 \
--max_steps=7427 \
--warmup_steps=742 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/umb_latn_5mb.txt \
--seed=43 \
--override_n_examples=2971 \
--output_dir=models/5mb/umb_latn_5mb
cp tokenizers/monolingual/umb_latn_5mb/* models/5mb/umb_latn_5mb

# urd_arab
if test -f models/5mb/urd_arab_5mb/pytorch_model.bin; then
echo "Model already found: urd_arab_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/urd_arab_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/urd_arab_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=320 --save_steps=999999999 \
--max_steps=6417 \
--warmup_steps=641 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/urd_arab_5mb.txt \
--seed=43 \
--override_n_examples=2567 \
--output_dir=models/5mb/urd_arab_5mb
cp tokenizers/monolingual/urd_arab_5mb/* models/5mb/urd_arab_5mb

# uzb_cyrl
if test -f models/5mb/uzb_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: uzb_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzb_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzb_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=272 --save_steps=999999999 \
--max_steps=5450 \
--warmup_steps=545 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzb_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2180 \
--output_dir=models/5mb/uzb_cyrl_5mb
cp tokenizers/monolingual/uzb_cyrl_5mb/* models/5mb/uzb_cyrl_5mb

# uzb_latn
if test -f models/5mb/uzb_latn_5mb/pytorch_model.bin; then
echo "Model already found: uzb_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzb_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzb_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=333 --save_steps=999999999 \
--max_steps=6665 \
--warmup_steps=666 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzb_latn_5mb.txt \
--seed=43 \
--override_n_examples=2666 \
--output_dir=models/5mb/uzb_latn_5mb
cp tokenizers/monolingual/uzb_latn_5mb/* models/5mb/uzb_latn_5mb

# uzn_cyrl
if test -f models/5mb/uzn_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: uzn_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzn_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzn_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=277 --save_steps=999999999 \
--max_steps=5545 \
--warmup_steps=554 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzn_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2218 \
--output_dir=models/5mb/uzn_cyrl_5mb
cp tokenizers/monolingual/uzn_cyrl_5mb/* models/5mb/uzn_cyrl_5mb

# uzn_latn
if test -f models/5mb/uzn_latn_5mb/pytorch_model.bin; then
echo "Model already found: uzn_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzn_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzn_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=453 --save_steps=999999999 \
--max_steps=9065 \
--warmup_steps=906 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzn_latn_5mb.txt \
--seed=43 \
--override_n_examples=3626 \
--output_dir=models/5mb/uzn_latn_5mb
cp tokenizers/monolingual/uzn_latn_5mb/* models/5mb/uzn_latn_5mb

# vec_latn
if test -f models/5mb/vec_latn_5mb/pytorch_model.bin; then
echo "Model already found: vec_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vec_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vec_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=342 --save_steps=999999999 \
--max_steps=6842 \
--warmup_steps=684 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vec_latn_5mb.txt \
--seed=43 \
--override_n_examples=2737 \
--output_dir=models/5mb/vec_latn_5mb
cp tokenizers/monolingual/vec_latn_5mb/* models/5mb/vec_latn_5mb

# ven_latn
if test -f models/5mb/ven_latn_5mb/pytorch_model.bin; then
echo "Model already found: ven_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ven_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ven_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=346 --save_steps=999999999 \
--max_steps=6930 \
--warmup_steps=693 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ven_latn_5mb.txt \
--seed=43 \
--override_n_examples=2772 \
--output_dir=models/5mb/ven_latn_5mb
cp tokenizers/monolingual/ven_latn_5mb/* models/5mb/ven_latn_5mb

# vep_latn
if test -f models/5mb/vep_latn_5mb/pytorch_model.bin; then
echo "Model already found: vep_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vep_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vep_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6302 \
--warmup_steps=630 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vep_latn_5mb.txt \
--seed=43 \
--override_n_examples=2521 \
--output_dir=models/5mb/vep_latn_5mb
cp tokenizers/monolingual/vep_latn_5mb/* models/5mb/vep_latn_5mb

# vie_latn
if test -f models/5mb/vie_latn_5mb/pytorch_model.bin; then
echo "Model already found: vie_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vie_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vie_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=331 --save_steps=999999999 \
--max_steps=6632 \
--warmup_steps=663 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vie_latn_5mb.txt \
--seed=43 \
--override_n_examples=2653 \
--output_dir=models/5mb/vie_latn_5mb
cp tokenizers/monolingual/vie_latn_5mb/* models/5mb/vie_latn_5mb

# vls_latn
if test -f models/5mb/vls_latn_5mb/pytorch_model.bin; then
echo "Model already found: vls_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vls_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vls_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=361 --save_steps=999999999 \
--max_steps=7237 \
--warmup_steps=723 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vls_latn_5mb.txt \
--seed=43 \
--override_n_examples=2895 \
--output_dir=models/5mb/vls_latn_5mb
cp tokenizers/monolingual/vls_latn_5mb/* models/5mb/vls_latn_5mb

# vol_latn
if test -f models/5mb/vol_latn_5mb/pytorch_model.bin; then
echo "Model already found: vol_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vol_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vol_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=356 --save_steps=999999999 \
--max_steps=7120 \
--warmup_steps=712 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vol_latn_5mb.txt \
--seed=43 \
--override_n_examples=2848 \
--output_dir=models/5mb/vol_latn_5mb
cp tokenizers/monolingual/vol_latn_5mb/* models/5mb/vol_latn_5mb

# war_latn
if test -f models/5mb/war_latn_5mb/pytorch_model.bin; then
echo "Model already found: war_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/war_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/war_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=364 --save_steps=999999999 \
--max_steps=7292 \
--warmup_steps=729 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/war_latn_5mb.txt \
--seed=43 \
--override_n_examples=2917 \
--output_dir=models/5mb/war_latn_5mb
cp tokenizers/monolingual/war_latn_5mb/* models/5mb/war_latn_5mb

# wln_latn
if test -f models/5mb/wln_latn_5mb/pytorch_model.bin; then
echo "Model already found: wln_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wln_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wln_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=407 --save_steps=999999999 \
--max_steps=8157 \
--warmup_steps=815 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wln_latn_5mb.txt \
--seed=43 \
--override_n_examples=3263 \
--output_dir=models/5mb/wln_latn_5mb
cp tokenizers/monolingual/wln_latn_5mb/* models/5mb/wln_latn_5mb

# wol_latn
if test -f models/5mb/wol_latn_5mb/pytorch_model.bin; then
echo "Model already found: wol_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wol_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wol_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=404 --save_steps=999999999 \
--max_steps=8095 \
--warmup_steps=809 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wol_latn_5mb.txt \
--seed=43 \
--override_n_examples=3238 \
--output_dir=models/5mb/wol_latn_5mb
cp tokenizers/monolingual/wol_latn_5mb/* models/5mb/wol_latn_5mb

# wuu_hani
if test -f models/5mb/wuu_hani_5mb/pytorch_model.bin; then
echo "Model already found: wuu_hani_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wuu_hani_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wuu_hani_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=211 --save_steps=999999999 \
--max_steps=4237 \
--warmup_steps=423 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wuu_hani_5mb.txt \
--seed=43 \
--override_n_examples=1695 \
--output_dir=models/5mb/wuu_hani_5mb
cp tokenizers/monolingual/wuu_hani_5mb/* models/5mb/wuu_hani_5mb

# xal_cyrl
if test -f models/5mb/xal_cyrl_5mb/pytorch_model.bin; then
echo "Model already found: xal_cyrl_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/xal_cyrl_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/xal_cyrl_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6040 \
--warmup_steps=604 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/xal_cyrl_5mb.txt \
--seed=43 \
--override_n_examples=2416 \
--output_dir=models/5mb/xal_cyrl_5mb
cp tokenizers/monolingual/xal_cyrl_5mb/* models/5mb/xal_cyrl_5mb

# xho_latn
if test -f models/5mb/xho_latn_5mb/pytorch_model.bin; then
echo "Model already found: xho_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/xho_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/xho_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=345 --save_steps=999999999 \
--max_steps=6917 \
--warmup_steps=691 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/xho_latn_5mb.txt \
--seed=43 \
--override_n_examples=2767 \
--output_dir=models/5mb/xho_latn_5mb
cp tokenizers/monolingual/xho_latn_5mb/* models/5mb/xho_latn_5mb

# xmf_geor
if test -f models/5mb/xmf_geor_5mb/pytorch_model.bin; then
echo "Model already found: xmf_geor_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/xmf_geor_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/xmf_geor_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6187 \
--warmup_steps=618 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/xmf_geor_5mb.txt \
--seed=43 \
--override_n_examples=2475 \
--output_dir=models/5mb/xmf_geor_5mb
cp tokenizers/monolingual/xmf_geor_5mb/* models/5mb/xmf_geor_5mb

# ydd_hebr
if test -f models/5mb/ydd_hebr_5mb/pytorch_model.bin; then
echo "Model already found: ydd_hebr_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ydd_hebr_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ydd_hebr_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=297 --save_steps=999999999 \
--max_steps=5955 \
--warmup_steps=595 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ydd_hebr_5mb.txt \
--seed=43 \
--override_n_examples=2382 \
--output_dir=models/5mb/ydd_hebr_5mb
cp tokenizers/monolingual/ydd_hebr_5mb/* models/5mb/ydd_hebr_5mb

# yid_hebr
if test -f models/5mb/yid_hebr_5mb/pytorch_model.bin; then
echo "Model already found: yid_hebr_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yid_hebr_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yid_hebr_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=245 --save_steps=999999999 \
--max_steps=4907 \
--warmup_steps=490 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yid_hebr_5mb.txt \
--seed=43 \
--override_n_examples=1963 \
--output_dir=models/5mb/yid_hebr_5mb
cp tokenizers/monolingual/yid_hebr_5mb/* models/5mb/yid_hebr_5mb

# yor_latn
if test -f models/5mb/yor_latn_5mb/pytorch_model.bin; then
echo "Model already found: yor_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yor_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yor_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=394 --save_steps=999999999 \
--max_steps=7890 \
--warmup_steps=789 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yor_latn_5mb.txt \
--seed=43 \
--override_n_examples=3156 \
--output_dir=models/5mb/yor_latn_5mb
cp tokenizers/monolingual/yor_latn_5mb/* models/5mb/yor_latn_5mb

# yua_latn
if test -f models/5mb/yua_latn_5mb/pytorch_model.bin; then
echo "Model already found: yua_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yua_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yua_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=397 --save_steps=999999999 \
--max_steps=7957 \
--warmup_steps=795 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yua_latn_5mb.txt \
--seed=43 \
--override_n_examples=3183 \
--output_dir=models/5mb/yua_latn_5mb
cp tokenizers/monolingual/yua_latn_5mb/* models/5mb/yua_latn_5mb

# yue_hant
if test -f models/5mb/yue_hant_5mb/pytorch_model.bin; then
echo "Model already found: yue_hant_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yue_hant_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yue_hant_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=259 --save_steps=999999999 \
--max_steps=5180 \
--warmup_steps=518 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yue_hant_5mb.txt \
--seed=43 \
--override_n_examples=2072 \
--output_dir=models/5mb/yue_hant_5mb
cp tokenizers/monolingual/yue_hant_5mb/* models/5mb/yue_hant_5mb

# zap_latn
if test -f models/5mb/zap_latn_5mb/pytorch_model.bin; then
echo "Model already found: zap_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zap_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zap_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=313 --save_steps=999999999 \
--max_steps=6272 \
--warmup_steps=627 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zap_latn_5mb.txt \
--seed=43 \
--override_n_examples=2509 \
--output_dir=models/5mb/zap_latn_5mb
cp tokenizers/monolingual/zap_latn_5mb/* models/5mb/zap_latn_5mb

# zho_hans
if test -f models/5mb/zho_hans_5mb/pytorch_model.bin; then
echo "Model already found: zho_hans_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zho_hans_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zho_hans_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=265 --save_steps=999999999 \
--max_steps=5302 \
--warmup_steps=530 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zho_hans_5mb.txt \
--seed=43 \
--override_n_examples=2121 \
--output_dir=models/5mb/zho_hans_5mb
cp tokenizers/monolingual/zho_hans_5mb/* models/5mb/zho_hans_5mb

# zho_hant
if test -f models/5mb/zho_hant_5mb/pytorch_model.bin; then
echo "Model already found: zho_hant_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zho_hant_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zho_hant_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=302 --save_steps=999999999 \
--max_steps=6052 \
--warmup_steps=605 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zho_hant_5mb.txt \
--seed=43 \
--override_n_examples=2421 \
--output_dir=models/5mb/zho_hant_5mb
cp tokenizers/monolingual/zho_hant_5mb/* models/5mb/zho_hant_5mb

# zsm_latn
if test -f models/5mb/zsm_latn_5mb/pytorch_model.bin; then
echo "Model already found: zsm_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zsm_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zsm_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=277 --save_steps=999999999 \
--max_steps=5550 \
--warmup_steps=555 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zsm_latn_5mb.txt \
--seed=43 \
--override_n_examples=2220 \
--output_dir=models/5mb/zsm_latn_5mb
cp tokenizers/monolingual/zsm_latn_5mb/* models/5mb/zsm_latn_5mb

# zul_latn
if test -f models/5mb/zul_latn_5mb/pytorch_model.bin; then
echo "Model already found: zul_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zul_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zul_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=337 --save_steps=999999999 \
--max_steps=6742 \
--warmup_steps=674 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zul_latn_5mb.txt \
--seed=43 \
--override_n_examples=2697 \
--output_dir=models/5mb/zul_latn_5mb
cp tokenizers/monolingual/zul_latn_5mb/* models/5mb/zul_latn_5mb

# zza_latn
if test -f models/5mb/zza_latn_5mb/pytorch_model.bin; then
echo "Model already found: zza_latn_5mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zza_latn_5mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zza_latn_5mb_eval2k.txt \
--per_device_train_batch_size=4 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=405 --save_steps=999999999 \
--max_steps=8102 \
--warmup_steps=810 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zza_latn_5mb.txt \
--seed=43 \
--override_n_examples=3241 \
--output_dir=models/5mb/zza_latn_5mb
cp tokenizers/monolingual/zza_latn_5mb/* models/5mb/zza_latn_5mb
