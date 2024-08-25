export CUDA_VISIBLE_DEVICES=0

# abk_cyrl
if test -f models/10mb/abk_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: abk_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/abk_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/abk_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=316 --save_steps=999999999 \
--max_steps=6322 \
--warmup_steps=632 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/abk_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=5058 \
--output_dir=models/10mb/abk_cyrl_10mb
cp tokenizers/monolingual/abk_cyrl_10mb/* models/10mb/abk_cyrl_10mb

# ace_latn
if test -f models/10mb/ace_latn_10mb/pytorch_model.bin; then
echo "Model already found: ace_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ace_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ace_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=382 --save_steps=999999999 \
--max_steps=7642 \
--warmup_steps=764 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ace_latn_10mb.txt \
--seed=43 \
--override_n_examples=6114 \
--output_dir=models/10mb/ace_latn_10mb
cp tokenizers/monolingual/ace_latn_10mb/* models/10mb/ace_latn_10mb

# ady_cyrl
if test -f models/10mb/ady_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: ady_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ady_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ady_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=335 --save_steps=999999999 \
--max_steps=6712 \
--warmup_steps=671 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ady_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=5370 \
--output_dir=models/10mb/ady_cyrl_10mb
cp tokenizers/monolingual/ady_cyrl_10mb/* models/10mb/ady_cyrl_10mb

# afb_arab
if test -f models/10mb/afb_arab_10mb/pytorch_model.bin; then
echo "Model already found: afb_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/afb_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/afb_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5630 \
--warmup_steps=563 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/afb_arab_10mb.txt \
--seed=43 \
--override_n_examples=4504 \
--output_dir=models/10mb/afb_arab_10mb
cp tokenizers/monolingual/afb_arab_10mb/* models/10mb/afb_arab_10mb

# afr_latn
if test -f models/10mb/afr_latn_10mb/pytorch_model.bin; then
echo "Model already found: afr_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/afr_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/afr_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=300 --save_steps=999999999 \
--max_steps=6007 \
--warmup_steps=600 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/afr_latn_10mb.txt \
--seed=43 \
--override_n_examples=4806 \
--output_dir=models/10mb/afr_latn_10mb
cp tokenizers/monolingual/afr_latn_10mb/* models/10mb/afr_latn_10mb

# aka_latn
if test -f models/10mb/aka_latn_10mb/pytorch_model.bin; then
echo "Model already found: aka_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aka_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aka_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=566 --save_steps=999999999 \
--max_steps=11320 \
--warmup_steps=1132 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aka_latn_10mb.txt \
--seed=43 \
--override_n_examples=9056 \
--output_dir=models/10mb/aka_latn_10mb
cp tokenizers/monolingual/aka_latn_10mb/* models/10mb/aka_latn_10mb

# als_latn
if test -f models/10mb/als_latn_10mb/pytorch_model.bin; then
echo "Model already found: als_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/als_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/als_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=328 --save_steps=999999999 \
--max_steps=6568 \
--warmup_steps=656 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/als_latn_10mb.txt \
--seed=43 \
--override_n_examples=5255 \
--output_dir=models/10mb/als_latn_10mb
cp tokenizers/monolingual/als_latn_10mb/* models/10mb/als_latn_10mb

# alt_cyrl
if test -f models/10mb/alt_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: alt_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/alt_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/alt_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5676 \
--warmup_steps=567 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/alt_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4541 \
--output_dir=models/10mb/alt_cyrl_10mb
cp tokenizers/monolingual/alt_cyrl_10mb/* models/10mb/alt_cyrl_10mb

# amh_ethi
if test -f models/10mb/amh_ethi_10mb/pytorch_model.bin; then
echo "Model already found: amh_ethi_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/amh_ethi_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/amh_ethi_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=270 --save_steps=999999999 \
--max_steps=5402 \
--warmup_steps=540 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/amh_ethi_10mb.txt \
--seed=43 \
--override_n_examples=4322 \
--output_dir=models/10mb/amh_ethi_10mb
cp tokenizers/monolingual/amh_ethi_10mb/* models/10mb/amh_ethi_10mb

# arb_arab
if test -f models/10mb/arb_arab_10mb/pytorch_model.bin; then
echo "Model already found: arb_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arb_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arb_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=249 --save_steps=999999999 \
--max_steps=4996 \
--warmup_steps=499 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arb_arab_10mb.txt \
--seed=43 \
--override_n_examples=3997 \
--output_dir=models/10mb/arb_arab_10mb
cp tokenizers/monolingual/arb_arab_10mb/* models/10mb/arb_arab_10mb

# arg_latn
if test -f models/10mb/arg_latn_10mb/pytorch_model.bin; then
echo "Model already found: arg_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arg_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arg_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=363 --save_steps=999999999 \
--max_steps=7271 \
--warmup_steps=727 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arg_latn_10mb.txt \
--seed=43 \
--override_n_examples=5817 \
--output_dir=models/10mb/arg_latn_10mb
cp tokenizers/monolingual/arg_latn_10mb/* models/10mb/arg_latn_10mb

# arz_arab
if test -f models/10mb/arz_arab_10mb/pytorch_model.bin; then
echo "Model already found: arz_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arz_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arz_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=323 --save_steps=999999999 \
--max_steps=6477 \
--warmup_steps=647 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arz_arab_10mb.txt \
--seed=43 \
--override_n_examples=5182 \
--output_dir=models/10mb/arz_arab_10mb
cp tokenizers/monolingual/arz_arab_10mb/* models/10mb/arz_arab_10mb

# asm_beng
if test -f models/10mb/asm_beng_10mb/pytorch_model.bin; then
echo "Model already found: asm_beng_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/asm_beng_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/asm_beng_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=281 --save_steps=999999999 \
--max_steps=5623 \
--warmup_steps=562 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/asm_beng_10mb.txt \
--seed=43 \
--override_n_examples=4499 \
--output_dir=models/10mb/asm_beng_10mb
cp tokenizers/monolingual/asm_beng_10mb/* models/10mb/asm_beng_10mb

# ast_latn
if test -f models/10mb/ast_latn_10mb/pytorch_model.bin; then
echo "Model already found: ast_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ast_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ast_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=516 --save_steps=999999999 \
--max_steps=10333 \
--warmup_steps=1033 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ast_latn_10mb.txt \
--seed=43 \
--override_n_examples=8267 \
--output_dir=models/10mb/ast_latn_10mb
cp tokenizers/monolingual/ast_latn_10mb/* models/10mb/ast_latn_10mb

# ava_cyrl
if test -f models/10mb/ava_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: ava_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ava_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ava_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=324 --save_steps=999999999 \
--max_steps=6483 \
--warmup_steps=648 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ava_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=5187 \
--output_dir=models/10mb/ava_cyrl_10mb
cp tokenizers/monolingual/ava_cyrl_10mb/* models/10mb/ava_cyrl_10mb

# aym_latn
if test -f models/10mb/aym_latn_10mb/pytorch_model.bin; then
echo "Model already found: aym_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aym_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aym_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=373 --save_steps=999999999 \
--max_steps=7467 \
--warmup_steps=746 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aym_latn_10mb.txt \
--seed=43 \
--override_n_examples=5974 \
--output_dir=models/10mb/aym_latn_10mb
cp tokenizers/monolingual/aym_latn_10mb/* models/10mb/aym_latn_10mb

# ayr_latn
if test -f models/10mb/ayr_latn_10mb/pytorch_model.bin; then
echo "Model already found: ayr_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ayr_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ayr_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=335 --save_steps=999999999 \
--max_steps=6712 \
--warmup_steps=671 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ayr_latn_10mb.txt \
--seed=43 \
--override_n_examples=5370 \
--output_dir=models/10mb/ayr_latn_10mb
cp tokenizers/monolingual/ayr_latn_10mb/* models/10mb/ayr_latn_10mb

# azb_arab
if test -f models/10mb/azb_arab_10mb/pytorch_model.bin; then
echo "Model already found: azb_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/azb_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/azb_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=314 --save_steps=999999999 \
--max_steps=6292 \
--warmup_steps=629 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/azb_arab_10mb.txt \
--seed=43 \
--override_n_examples=5034 \
--output_dir=models/10mb/azb_arab_10mb
cp tokenizers/monolingual/azb_arab_10mb/* models/10mb/azb_arab_10mb

# aze_arab
if test -f models/10mb/aze_arab_10mb/pytorch_model.bin; then
echo "Model already found: aze_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=268 --save_steps=999999999 \
--max_steps=5378 \
--warmup_steps=537 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_arab_10mb.txt \
--seed=43 \
--override_n_examples=4303 \
--output_dir=models/10mb/aze_arab_10mb
cp tokenizers/monolingual/aze_arab_10mb/* models/10mb/aze_arab_10mb

# aze_cyrl
if test -f models/10mb/aze_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: aze_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=262 --save_steps=999999999 \
--max_steps=5256 \
--warmup_steps=525 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4205 \
--output_dir=models/10mb/aze_cyrl_10mb
cp tokenizers/monolingual/aze_cyrl_10mb/* models/10mb/aze_cyrl_10mb

# aze_latn
if test -f models/10mb/aze_latn_10mb/pytorch_model.bin; then
echo "Model already found: aze_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=294 --save_steps=999999999 \
--max_steps=5883 \
--warmup_steps=588 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_latn_10mb.txt \
--seed=43 \
--override_n_examples=4707 \
--output_dir=models/10mb/aze_latn_10mb
cp tokenizers/monolingual/aze_latn_10mb/* models/10mb/aze_latn_10mb

# azj_latn
if test -f models/10mb/azj_latn_10mb/pytorch_model.bin; then
echo "Model already found: azj_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/azj_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/azj_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=229 --save_steps=999999999 \
--max_steps=4590 \
--warmup_steps=459 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/azj_latn_10mb.txt \
--seed=43 \
--override_n_examples=3672 \
--output_dir=models/10mb/azj_latn_10mb
cp tokenizers/monolingual/azj_latn_10mb/* models/10mb/azj_latn_10mb

# bak_cyrl
if test -f models/10mb/bak_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: bak_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bak_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bak_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=371 --save_steps=999999999 \
--max_steps=7431 \
--warmup_steps=743 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bak_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=5945 \
--output_dir=models/10mb/bak_cyrl_10mb
cp tokenizers/monolingual/bak_cyrl_10mb/* models/10mb/bak_cyrl_10mb

# bam_latn
if test -f models/10mb/bam_latn_10mb/pytorch_model.bin; then
echo "Model already found: bam_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bam_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bam_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8623 \
--warmup_steps=862 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bam_latn_10mb.txt \
--seed=43 \
--override_n_examples=6899 \
--output_dir=models/10mb/bam_latn_10mb
cp tokenizers/monolingual/bam_latn_10mb/* models/10mb/bam_latn_10mb

# ban_latn
if test -f models/10mb/ban_latn_10mb/pytorch_model.bin; then
echo "Model already found: ban_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ban_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ban_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=358 --save_steps=999999999 \
--max_steps=7175 \
--warmup_steps=717 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ban_latn_10mb.txt \
--seed=43 \
--override_n_examples=5740 \
--output_dir=models/10mb/ban_latn_10mb
cp tokenizers/monolingual/ban_latn_10mb/* models/10mb/ban_latn_10mb

# bar_latn
if test -f models/10mb/bar_latn_10mb/pytorch_model.bin; then
echo "Model already found: bar_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bar_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bar_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=357 --save_steps=999999999 \
--max_steps=7151 \
--warmup_steps=715 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bar_latn_10mb.txt \
--seed=43 \
--override_n_examples=5721 \
--output_dir=models/10mb/bar_latn_10mb
cp tokenizers/monolingual/bar_latn_10mb/* models/10mb/bar_latn_10mb

# bel_cyrl
if test -f models/10mb/bel_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: bel_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bel_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bel_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=321 --save_steps=999999999 \
--max_steps=6433 \
--warmup_steps=643 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bel_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=5147 \
--output_dir=models/10mb/bel_cyrl_10mb
cp tokenizers/monolingual/bel_cyrl_10mb/* models/10mb/bel_cyrl_10mb

# bem_latn
if test -f models/10mb/bem_latn_10mb/pytorch_model.bin; then
echo "Model already found: bem_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bem_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bem_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=359 --save_steps=999999999 \
--max_steps=7185 \
--warmup_steps=718 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bem_latn_10mb.txt \
--seed=43 \
--override_n_examples=5748 \
--output_dir=models/10mb/bem_latn_10mb
cp tokenizers/monolingual/bem_latn_10mb/* models/10mb/bem_latn_10mb

# ben_beng
if test -f models/10mb/ben_beng_10mb/pytorch_model.bin; then
echo "Model already found: ben_beng_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ben_beng_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ben_beng_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=247 --save_steps=999999999 \
--max_steps=4942 \
--warmup_steps=494 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ben_beng_10mb.txt \
--seed=43 \
--override_n_examples=3954 \
--output_dir=models/10mb/ben_beng_10mb
cp tokenizers/monolingual/ben_beng_10mb/* models/10mb/ben_beng_10mb

# bew_cyrl
if test -f models/10mb/bew_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: bew_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bew_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bew_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=281 --save_steps=999999999 \
--max_steps=5628 \
--warmup_steps=562 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bew_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4503 \
--output_dir=models/10mb/bew_cyrl_10mb
cp tokenizers/monolingual/bew_cyrl_10mb/* models/10mb/bew_cyrl_10mb

# bew_latn
if test -f models/10mb/bew_latn_10mb/pytorch_model.bin; then
echo "Model already found: bew_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bew_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bew_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6602 \
--warmup_steps=660 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bew_latn_10mb.txt \
--seed=43 \
--override_n_examples=5282 \
--output_dir=models/10mb/bew_latn_10mb
cp tokenizers/monolingual/bew_latn_10mb/* models/10mb/bew_latn_10mb

# bho_deva
if test -f models/10mb/bho_deva_10mb/pytorch_model.bin; then
echo "Model already found: bho_deva_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bho_deva_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bho_deva_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=322 --save_steps=999999999 \
--max_steps=6458 \
--warmup_steps=645 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bho_deva_10mb.txt \
--seed=43 \
--override_n_examples=5167 \
--output_dir=models/10mb/bho_deva_10mb
cp tokenizers/monolingual/bho_deva_10mb/* models/10mb/bho_deva_10mb

# bik_latn
if test -f models/10mb/bik_latn_10mb/pytorch_model.bin; then
echo "Model already found: bik_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bik_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bik_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=350 --save_steps=999999999 \
--max_steps=7013 \
--warmup_steps=701 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bik_latn_10mb.txt \
--seed=43 \
--override_n_examples=5611 \
--output_dir=models/10mb/bik_latn_10mb
cp tokenizers/monolingual/bik_latn_10mb/* models/10mb/bik_latn_10mb

# bjn_latn
if test -f models/10mb/bjn_latn_10mb/pytorch_model.bin; then
echo "Model already found: bjn_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bjn_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bjn_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=334 --save_steps=999999999 \
--max_steps=6696 \
--warmup_steps=669 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bjn_latn_10mb.txt \
--seed=43 \
--override_n_examples=5357 \
--output_dir=models/10mb/bjn_latn_10mb
cp tokenizers/monolingual/bjn_latn_10mb/* models/10mb/bjn_latn_10mb

# bod_tibt
if test -f models/10mb/bod_tibt_10mb/pytorch_model.bin; then
echo "Model already found: bod_tibt_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bod_tibt_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bod_tibt_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=228 --save_steps=999999999 \
--max_steps=4562 \
--warmup_steps=456 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bod_tibt_10mb.txt \
--seed=43 \
--override_n_examples=3650 \
--output_dir=models/10mb/bod_tibt_10mb
cp tokenizers/monolingual/bod_tibt_10mb/* models/10mb/bod_tibt_10mb

# bos_cyrl
if test -f models/10mb/bos_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: bos_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bos_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bos_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=296 --save_steps=999999999 \
--max_steps=5933 \
--warmup_steps=593 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bos_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4747 \
--output_dir=models/10mb/bos_cyrl_10mb
cp tokenizers/monolingual/bos_cyrl_10mb/* models/10mb/bos_cyrl_10mb

# bos_latn
if test -f models/10mb/bos_latn_10mb/pytorch_model.bin; then
echo "Model already found: bos_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bos_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bos_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=288 --save_steps=999999999 \
--max_steps=5771 \
--warmup_steps=577 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bos_latn_10mb.txt \
--seed=43 \
--override_n_examples=4617 \
--output_dir=models/10mb/bos_latn_10mb
cp tokenizers/monolingual/bos_latn_10mb/* models/10mb/bos_latn_10mb

# bre_latn
if test -f models/10mb/bre_latn_10mb/pytorch_model.bin; then
echo "Model already found: bre_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bre_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bre_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=333 --save_steps=999999999 \
--max_steps=6675 \
--warmup_steps=667 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bre_latn_10mb.txt \
--seed=43 \
--override_n_examples=5340 \
--output_dir=models/10mb/bre_latn_10mb
cp tokenizers/monolingual/bre_latn_10mb/* models/10mb/bre_latn_10mb

# bua_cyrl
if test -f models/10mb/bua_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: bua_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bua_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bua_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=286 --save_steps=999999999 \
--max_steps=5733 \
--warmup_steps=573 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bua_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4587 \
--output_dir=models/10mb/bua_cyrl_10mb
cp tokenizers/monolingual/bua_cyrl_10mb/* models/10mb/bua_cyrl_10mb

# bug_latn
if test -f models/10mb/bug_latn_10mb/pytorch_model.bin; then
echo "Model already found: bug_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bug_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bug_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7465 \
--warmup_steps=746 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bug_latn_10mb.txt \
--seed=43 \
--override_n_examples=5972 \
--output_dir=models/10mb/bug_latn_10mb
cp tokenizers/monolingual/bug_latn_10mb/* models/10mb/bug_latn_10mb

# bul_cyrl
if test -f models/10mb/bul_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: bul_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bul_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bul_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=283 --save_steps=999999999 \
--max_steps=5673 \
--warmup_steps=567 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bul_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4539 \
--output_dir=models/10mb/bul_cyrl_10mb
cp tokenizers/monolingual/bul_cyrl_10mb/* models/10mb/bul_cyrl_10mb

# bxr_cyrl
if test -f models/10mb/bxr_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: bxr_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bxr_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bxr_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=298 --save_steps=999999999 \
--max_steps=5960 \
--warmup_steps=596 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bxr_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4768 \
--output_dir=models/10mb/bxr_cyrl_10mb
cp tokenizers/monolingual/bxr_cyrl_10mb/* models/10mb/bxr_cyrl_10mb

# cat_latn
if test -f models/10mb/cat_latn_10mb/pytorch_model.bin; then
echo "Model already found: cat_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cat_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cat_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=298 --save_steps=999999999 \
--max_steps=5963 \
--warmup_steps=596 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cat_latn_10mb.txt \
--seed=43 \
--override_n_examples=4771 \
--output_dir=models/10mb/cat_latn_10mb
cp tokenizers/monolingual/cat_latn_10mb/* models/10mb/cat_latn_10mb

# ceb_latn
if test -f models/10mb/ceb_latn_10mb/pytorch_model.bin; then
echo "Model already found: ceb_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ceb_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ceb_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=332 --save_steps=999999999 \
--max_steps=6648 \
--warmup_steps=664 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ceb_latn_10mb.txt \
--seed=43 \
--override_n_examples=5319 \
--output_dir=models/10mb/ceb_latn_10mb
cp tokenizers/monolingual/ceb_latn_10mb/* models/10mb/ceb_latn_10mb

# ces_latn
if test -f models/10mb/ces_latn_10mb/pytorch_model.bin; then
echo "Model already found: ces_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ces_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ces_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=261 --save_steps=999999999 \
--max_steps=5237 \
--warmup_steps=523 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ces_latn_10mb.txt \
--seed=43 \
--override_n_examples=4190 \
--output_dir=models/10mb/ces_latn_10mb
cp tokenizers/monolingual/ces_latn_10mb/* models/10mb/ces_latn_10mb

# cfm_latn
if test -f models/10mb/cfm_latn_10mb/pytorch_model.bin; then
echo "Model already found: cfm_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cfm_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cfm_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=377 --save_steps=999999999 \
--max_steps=7543 \
--warmup_steps=754 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cfm_latn_10mb.txt \
--seed=43 \
--override_n_examples=6035 \
--output_dir=models/10mb/cfm_latn_10mb
cp tokenizers/monolingual/cfm_latn_10mb/* models/10mb/cfm_latn_10mb

# che_cyrl
if test -f models/10mb/che_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: che_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/che_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/che_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=344 --save_steps=999999999 \
--max_steps=6892 \
--warmup_steps=689 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/che_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=5514 \
--output_dir=models/10mb/che_cyrl_10mb
cp tokenizers/monolingual/che_cyrl_10mb/* models/10mb/che_cyrl_10mb

# chm_cyrl
if test -f models/10mb/chm_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: chm_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/chm_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/chm_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=286 --save_steps=999999999 \
--max_steps=5738 \
--warmup_steps=573 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/chm_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4591 \
--output_dir=models/10mb/chm_cyrl_10mb
cp tokenizers/monolingual/chm_cyrl_10mb/* models/10mb/chm_cyrl_10mb

# chv_cyrl
if test -f models/10mb/chv_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: chv_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/chv_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/chv_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=409 --save_steps=999999999 \
--max_steps=8183 \
--warmup_steps=818 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/chv_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=6547 \
--output_dir=models/10mb/chv_cyrl_10mb
cp tokenizers/monolingual/chv_cyrl_10mb/* models/10mb/chv_cyrl_10mb

# cjk_latn
if test -f models/10mb/cjk_latn_10mb/pytorch_model.bin; then
echo "Model already found: cjk_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cjk_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cjk_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7351 \
--warmup_steps=735 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cjk_latn_10mb.txt \
--seed=43 \
--override_n_examples=5881 \
--output_dir=models/10mb/cjk_latn_10mb
cp tokenizers/monolingual/cjk_latn_10mb/* models/10mb/cjk_latn_10mb

# ckb_arab
if test -f models/10mb/ckb_arab_10mb/pytorch_model.bin; then
echo "Model already found: ckb_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ckb_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ckb_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=285 --save_steps=999999999 \
--max_steps=5702 \
--warmup_steps=570 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ckb_arab_10mb.txt \
--seed=43 \
--override_n_examples=4562 \
--output_dir=models/10mb/ckb_arab_10mb
cp tokenizers/monolingual/ckb_arab_10mb/* models/10mb/ckb_arab_10mb

# cnh_latn
if test -f models/10mb/cnh_latn_10mb/pytorch_model.bin; then
echo "Model already found: cnh_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cnh_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cnh_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=389 --save_steps=999999999 \
--max_steps=7780 \
--warmup_steps=778 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cnh_latn_10mb.txt \
--seed=43 \
--override_n_examples=6224 \
--output_dir=models/10mb/cnh_latn_10mb
cp tokenizers/monolingual/cnh_latn_10mb/* models/10mb/cnh_latn_10mb

# cos_latn
if test -f models/10mb/cos_latn_10mb/pytorch_model.bin; then
echo "Model already found: cos_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cos_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cos_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=384 --save_steps=999999999 \
--max_steps=7688 \
--warmup_steps=768 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cos_latn_10mb.txt \
--seed=43 \
--override_n_examples=6151 \
--output_dir=models/10mb/cos_latn_10mb
cp tokenizers/monolingual/cos_latn_10mb/* models/10mb/cos_latn_10mb

# crh_latn
if test -f models/10mb/crh_latn_10mb/pytorch_model.bin; then
echo "Model already found: crh_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/crh_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/crh_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=344 --save_steps=999999999 \
--max_steps=6885 \
--warmup_steps=688 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/crh_latn_10mb.txt \
--seed=43 \
--override_n_examples=5508 \
--output_dir=models/10mb/crh_latn_10mb
cp tokenizers/monolingual/crh_latn_10mb/* models/10mb/crh_latn_10mb

# ctd_latn
if test -f models/10mb/ctd_latn_10mb/pytorch_model.bin; then
echo "Model already found: ctd_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ctd_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ctd_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=381 --save_steps=999999999 \
--max_steps=7632 \
--warmup_steps=763 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ctd_latn_10mb.txt \
--seed=43 \
--override_n_examples=6106 \
--output_dir=models/10mb/ctd_latn_10mb
cp tokenizers/monolingual/ctd_latn_10mb/* models/10mb/ctd_latn_10mb

# cym_latn
if test -f models/10mb/cym_latn_10mb/pytorch_model.bin; then
echo "Model already found: cym_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cym_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cym_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=295 --save_steps=999999999 \
--max_steps=5915 \
--warmup_steps=591 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cym_latn_10mb.txt \
--seed=43 \
--override_n_examples=4732 \
--output_dir=models/10mb/cym_latn_10mb
cp tokenizers/monolingual/cym_latn_10mb/* models/10mb/cym_latn_10mb

# dan_latn
if test -f models/10mb/dan_latn_10mb/pytorch_model.bin; then
echo "Model already found: dan_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dan_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dan_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=261 --save_steps=999999999 \
--max_steps=5232 \
--warmup_steps=523 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dan_latn_10mb.txt \
--seed=43 \
--override_n_examples=4186 \
--output_dir=models/10mb/dan_latn_10mb
cp tokenizers/monolingual/dan_latn_10mb/* models/10mb/dan_latn_10mb

# dar_cyrl
if test -f models/10mb/dar_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: dar_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dar_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dar_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=327 --save_steps=999999999 \
--max_steps=6553 \
--warmup_steps=655 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dar_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=5243 \
--output_dir=models/10mb/dar_cyrl_10mb
cp tokenizers/monolingual/dar_cyrl_10mb/* models/10mb/dar_cyrl_10mb

# deu_latn
if test -f models/10mb/deu_latn_10mb/pytorch_model.bin; then
echo "Model already found: deu_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/deu_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/deu_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=264 --save_steps=999999999 \
--max_steps=5295 \
--warmup_steps=529 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/deu_latn_10mb.txt \
--seed=43 \
--override_n_examples=4236 \
--output_dir=models/10mb/deu_latn_10mb
cp tokenizers/monolingual/deu_latn_10mb/* models/10mb/deu_latn_10mb

# dik_latn
if test -f models/10mb/dik_latn_10mb/pytorch_model.bin; then
echo "Model already found: dik_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dik_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dik_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7921 \
--warmup_steps=792 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dik_latn_10mb.txt \
--seed=43 \
--override_n_examples=6337 \
--output_dir=models/10mb/dik_latn_10mb
cp tokenizers/monolingual/dik_latn_10mb/* models/10mb/dik_latn_10mb

# din_latn
if test -f models/10mb/din_latn_10mb/pytorch_model.bin; then
echo "Model already found: din_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/din_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/din_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8660 \
--warmup_steps=866 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/din_latn_10mb.txt \
--seed=43 \
--override_n_examples=6928 \
--output_dir=models/10mb/din_latn_10mb
cp tokenizers/monolingual/din_latn_10mb/* models/10mb/din_latn_10mb

# diq_latn
if test -f models/10mb/diq_latn_10mb/pytorch_model.bin; then
echo "Model already found: diq_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/diq_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/diq_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=328 --save_steps=999999999 \
--max_steps=6562 \
--warmup_steps=656 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/diq_latn_10mb.txt \
--seed=43 \
--override_n_examples=5250 \
--output_dir=models/10mb/diq_latn_10mb
cp tokenizers/monolingual/diq_latn_10mb/* models/10mb/diq_latn_10mb

# div_thaa
if test -f models/10mb/div_thaa_10mb/pytorch_model.bin; then
echo "Model already found: div_thaa_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/div_thaa_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/div_thaa_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=230 --save_steps=999999999 \
--max_steps=4615 \
--warmup_steps=461 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/div_thaa_10mb.txt \
--seed=43 \
--override_n_examples=3692 \
--output_dir=models/10mb/div_thaa_10mb
cp tokenizers/monolingual/div_thaa_10mb/* models/10mb/div_thaa_10mb

# dyu_latn
if test -f models/10mb/dyu_latn_10mb/pytorch_model.bin; then
echo "Model already found: dyu_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dyu_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dyu_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7892 \
--warmup_steps=789 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dyu_latn_10mb.txt \
--seed=43 \
--override_n_examples=6314 \
--output_dir=models/10mb/dyu_latn_10mb
cp tokenizers/monolingual/dyu_latn_10mb/* models/10mb/dyu_latn_10mb

# ekk_latn
if test -f models/10mb/ekk_latn_10mb/pytorch_model.bin; then
echo "Model already found: ekk_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ekk_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ekk_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=254 --save_steps=999999999 \
--max_steps=5095 \
--warmup_steps=509 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ekk_latn_10mb.txt \
--seed=43 \
--override_n_examples=4076 \
--output_dir=models/10mb/ekk_latn_10mb
cp tokenizers/monolingual/ekk_latn_10mb/* models/10mb/ekk_latn_10mb

# ell_grek
if test -f models/10mb/ell_grek_10mb/pytorch_model.bin; then
echo "Model already found: ell_grek_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ell_grek_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ell_grek_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=303 --save_steps=999999999 \
--max_steps=6077 \
--warmup_steps=607 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ell_grek_10mb.txt \
--seed=43 \
--override_n_examples=4862 \
--output_dir=models/10mb/ell_grek_10mb
cp tokenizers/monolingual/ell_grek_10mb/* models/10mb/ell_grek_10mb

# ell_latn
if test -f models/10mb/ell_latn_10mb/pytorch_model.bin; then
echo "Model already found: ell_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ell_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ell_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=311 --save_steps=999999999 \
--max_steps=6232 \
--warmup_steps=623 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ell_latn_10mb.txt \
--seed=43 \
--override_n_examples=4986 \
--output_dir=models/10mb/ell_latn_10mb
cp tokenizers/monolingual/ell_latn_10mb/* models/10mb/ell_latn_10mb

# eng_latn
if test -f models/10mb/eng_latn_10mb/pytorch_model.bin; then
echo "Model already found: eng_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/eng_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/eng_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=267 --save_steps=999999999 \
--max_steps=5352 \
--warmup_steps=535 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/eng_latn_10mb.txt \
--seed=43 \
--override_n_examples=4282 \
--output_dir=models/10mb/eng_latn_10mb
cp tokenizers/monolingual/eng_latn_10mb/* models/10mb/eng_latn_10mb

# epo_latn
if test -f models/10mb/epo_latn_10mb/pytorch_model.bin; then
echo "Model already found: epo_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/epo_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/epo_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=290 --save_steps=999999999 \
--max_steps=5812 \
--warmup_steps=581 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/epo_latn_10mb.txt \
--seed=43 \
--override_n_examples=4650 \
--output_dir=models/10mb/epo_latn_10mb
cp tokenizers/monolingual/epo_latn_10mb/* models/10mb/epo_latn_10mb

# est_latn
if test -f models/10mb/est_latn_10mb/pytorch_model.bin; then
echo "Model already found: est_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/est_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/est_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=240 --save_steps=999999999 \
--max_steps=4802 \
--warmup_steps=480 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/est_latn_10mb.txt \
--seed=43 \
--override_n_examples=3842 \
--output_dir=models/10mb/est_latn_10mb
cp tokenizers/monolingual/est_latn_10mb/* models/10mb/est_latn_10mb

# eus_latn
if test -f models/10mb/eus_latn_10mb/pytorch_model.bin; then
echo "Model already found: eus_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/eus_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/eus_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=264 --save_steps=999999999 \
--max_steps=5283 \
--warmup_steps=528 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/eus_latn_10mb.txt \
--seed=43 \
--override_n_examples=4227 \
--output_dir=models/10mb/eus_latn_10mb
cp tokenizers/monolingual/eus_latn_10mb/* models/10mb/eus_latn_10mb

# ewe_latn
if test -f models/10mb/ewe_latn_10mb/pytorch_model.bin; then
echo "Model already found: ewe_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ewe_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ewe_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=364 --save_steps=999999999 \
--max_steps=7293 \
--warmup_steps=729 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ewe_latn_10mb.txt \
--seed=43 \
--override_n_examples=5835 \
--output_dir=models/10mb/ewe_latn_10mb
cp tokenizers/monolingual/ewe_latn_10mb/* models/10mb/ewe_latn_10mb

# fao_latn
if test -f models/10mb/fao_latn_10mb/pytorch_model.bin; then
echo "Model already found: fao_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fao_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fao_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=301 --save_steps=999999999 \
--max_steps=6033 \
--warmup_steps=603 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fao_latn_10mb.txt \
--seed=43 \
--override_n_examples=4827 \
--output_dir=models/10mb/fao_latn_10mb
cp tokenizers/monolingual/fao_latn_10mb/* models/10mb/fao_latn_10mb

# fas_arab
if test -f models/10mb/fas_arab_10mb/pytorch_model.bin; then
echo "Model already found: fas_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fas_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fas_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=304 --save_steps=999999999 \
--max_steps=6087 \
--warmup_steps=608 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fas_arab_10mb.txt \
--seed=43 \
--override_n_examples=4870 \
--output_dir=models/10mb/fas_arab_10mb
cp tokenizers/monolingual/fas_arab_10mb/* models/10mb/fas_arab_10mb

# fij_latn
if test -f models/10mb/fij_latn_10mb/pytorch_model.bin; then
echo "Model already found: fij_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fij_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fij_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=371 --save_steps=999999999 \
--max_steps=7431 \
--warmup_steps=743 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fij_latn_10mb.txt \
--seed=43 \
--override_n_examples=5945 \
--output_dir=models/10mb/fij_latn_10mb
cp tokenizers/monolingual/fij_latn_10mb/* models/10mb/fij_latn_10mb

# fil_latn
if test -f models/10mb/fil_latn_10mb/pytorch_model.bin; then
echo "Model already found: fil_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fil_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fil_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=345 --save_steps=999999999 \
--max_steps=6912 \
--warmup_steps=691 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fil_latn_10mb.txt \
--seed=43 \
--override_n_examples=5530 \
--output_dir=models/10mb/fil_latn_10mb
cp tokenizers/monolingual/fil_latn_10mb/* models/10mb/fil_latn_10mb

# fin_latn
if test -f models/10mb/fin_latn_10mb/pytorch_model.bin; then
echo "Model already found: fin_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fin_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fin_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=238 --save_steps=999999999 \
--max_steps=4772 \
--warmup_steps=477 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fin_latn_10mb.txt \
--seed=43 \
--override_n_examples=3818 \
--output_dir=models/10mb/fin_latn_10mb
cp tokenizers/monolingual/fin_latn_10mb/* models/10mb/fin_latn_10mb

# fon_latn
if test -f models/10mb/fon_latn_10mb/pytorch_model.bin; then
echo "Model already found: fon_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fon_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fon_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=428 --save_steps=999999999 \
--max_steps=8562 \
--warmup_steps=856 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fon_latn_10mb.txt \
--seed=43 \
--override_n_examples=6850 \
--output_dir=models/10mb/fon_latn_10mb
cp tokenizers/monolingual/fon_latn_10mb/* models/10mb/fon_latn_10mb

# fra_latn
if test -f models/10mb/fra_latn_10mb/pytorch_model.bin; then
echo "Model already found: fra_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fra_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fra_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=313 --save_steps=999999999 \
--max_steps=6268 \
--warmup_steps=626 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fra_latn_10mb.txt \
--seed=43 \
--override_n_examples=5015 \
--output_dir=models/10mb/fra_latn_10mb
cp tokenizers/monolingual/fra_latn_10mb/* models/10mb/fra_latn_10mb

# fry_latn
if test -f models/10mb/fry_latn_10mb/pytorch_model.bin; then
echo "Model already found: fry_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fry_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fry_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=351 --save_steps=999999999 \
--max_steps=7032 \
--warmup_steps=703 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fry_latn_10mb.txt \
--seed=43 \
--override_n_examples=5626 \
--output_dir=models/10mb/fry_latn_10mb
cp tokenizers/monolingual/fry_latn_10mb/* models/10mb/fry_latn_10mb

# ful_latn
if test -f models/10mb/ful_latn_10mb/pytorch_model.bin; then
echo "Model already found: ful_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ful_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ful_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=402 --save_steps=999999999 \
--max_steps=8052 \
--warmup_steps=805 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ful_latn_10mb.txt \
--seed=43 \
--override_n_examples=6442 \
--output_dir=models/10mb/ful_latn_10mb
cp tokenizers/monolingual/ful_latn_10mb/* models/10mb/ful_latn_10mb

# fur_latn
if test -f models/10mb/fur_latn_10mb/pytorch_model.bin; then
echo "Model already found: fur_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fur_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fur_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=326 --save_steps=999999999 \
--max_steps=6531 \
--warmup_steps=653 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fur_latn_10mb.txt \
--seed=43 \
--override_n_examples=5225 \
--output_dir=models/10mb/fur_latn_10mb
cp tokenizers/monolingual/fur_latn_10mb/* models/10mb/fur_latn_10mb

# fuv_latn
if test -f models/10mb/fuv_latn_10mb/pytorch_model.bin; then
echo "Model already found: fuv_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fuv_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fuv_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=359 --save_steps=999999999 \
--max_steps=7188 \
--warmup_steps=718 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fuv_latn_10mb.txt \
--seed=43 \
--override_n_examples=5751 \
--output_dir=models/10mb/fuv_latn_10mb
cp tokenizers/monolingual/fuv_latn_10mb/* models/10mb/fuv_latn_10mb

# gaz_latn
if test -f models/10mb/gaz_latn_10mb/pytorch_model.bin; then
echo "Model already found: gaz_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gaz_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gaz_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=403 --save_steps=999999999 \
--max_steps=8060 \
--warmup_steps=806 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gaz_latn_10mb.txt \
--seed=43 \
--override_n_examples=6448 \
--output_dir=models/10mb/gaz_latn_10mb
cp tokenizers/monolingual/gaz_latn_10mb/* models/10mb/gaz_latn_10mb

# gla_latn
if test -f models/10mb/gla_latn_10mb/pytorch_model.bin; then
echo "Model already found: gla_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gla_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gla_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=275 --save_steps=999999999 \
--max_steps=5511 \
--warmup_steps=551 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gla_latn_10mb.txt \
--seed=43 \
--override_n_examples=4409 \
--output_dir=models/10mb/gla_latn_10mb
cp tokenizers/monolingual/gla_latn_10mb/* models/10mb/gla_latn_10mb

# gle_latn
if test -f models/10mb/gle_latn_10mb/pytorch_model.bin; then
echo "Model already found: gle_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gle_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gle_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=517 --save_steps=999999999 \
--max_steps=10345 \
--warmup_steps=1034 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gle_latn_10mb.txt \
--seed=43 \
--override_n_examples=8276 \
--output_dir=models/10mb/gle_latn_10mb
cp tokenizers/monolingual/gle_latn_10mb/* models/10mb/gle_latn_10mb

# glg_latn
if test -f models/10mb/glg_latn_10mb/pytorch_model.bin; then
echo "Model already found: glg_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/glg_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/glg_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=278 --save_steps=999999999 \
--max_steps=5577 \
--warmup_steps=557 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/glg_latn_10mb.txt \
--seed=43 \
--override_n_examples=4462 \
--output_dir=models/10mb/glg_latn_10mb
cp tokenizers/monolingual/glg_latn_10mb/* models/10mb/glg_latn_10mb

# glk_arab
if test -f models/10mb/glk_arab_10mb/pytorch_model.bin; then
echo "Model already found: glk_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/glk_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/glk_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=321 --save_steps=999999999 \
--max_steps=6420 \
--warmup_steps=642 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/glk_arab_10mb.txt \
--seed=43 \
--override_n_examples=5136 \
--output_dir=models/10mb/glk_arab_10mb
cp tokenizers/monolingual/glk_arab_10mb/* models/10mb/glk_arab_10mb

# gom_deva
if test -f models/10mb/gom_deva_10mb/pytorch_model.bin; then
echo "Model already found: gom_deva_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gom_deva_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gom_deva_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4761 \
--warmup_steps=476 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gom_deva_10mb.txt \
--seed=43 \
--override_n_examples=3809 \
--output_dir=models/10mb/gom_deva_10mb
cp tokenizers/monolingual/gom_deva_10mb/* models/10mb/gom_deva_10mb

# gom_latn
if test -f models/10mb/gom_latn_10mb/pytorch_model.bin; then
echo "Model already found: gom_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gom_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gom_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6625 \
--warmup_steps=662 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gom_latn_10mb.txt \
--seed=43 \
--override_n_examples=5300 \
--output_dir=models/10mb/gom_latn_10mb
cp tokenizers/monolingual/gom_latn_10mb/* models/10mb/gom_latn_10mb

# grc_grek
if test -f models/10mb/grc_grek_10mb/pytorch_model.bin; then
echo "Model already found: grc_grek_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/grc_grek_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/grc_grek_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=295 --save_steps=999999999 \
--max_steps=5906 \
--warmup_steps=590 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/grc_grek_10mb.txt \
--seed=43 \
--override_n_examples=4725 \
--output_dir=models/10mb/grc_grek_10mb
cp tokenizers/monolingual/grc_grek_10mb/* models/10mb/grc_grek_10mb

# grn_latn
if test -f models/10mb/grn_latn_10mb/pytorch_model.bin; then
echo "Model already found: grn_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/grn_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/grn_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=319 --save_steps=999999999 \
--max_steps=6387 \
--warmup_steps=638 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/grn_latn_10mb.txt \
--seed=43 \
--override_n_examples=5110 \
--output_dir=models/10mb/grn_latn_10mb
cp tokenizers/monolingual/grn_latn_10mb/* models/10mb/grn_latn_10mb

# gsw_latn
if test -f models/10mb/gsw_latn_10mb/pytorch_model.bin; then
echo "Model already found: gsw_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gsw_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gsw_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=376 --save_steps=999999999 \
--max_steps=7537 \
--warmup_steps=753 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gsw_latn_10mb.txt \
--seed=43 \
--override_n_examples=6030 \
--output_dir=models/10mb/gsw_latn_10mb
cp tokenizers/monolingual/gsw_latn_10mb/* models/10mb/gsw_latn_10mb

# guj_gujr
if test -f models/10mb/guj_gujr_10mb/pytorch_model.bin; then
echo "Model already found: guj_gujr_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/guj_gujr_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/guj_gujr_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=247 --save_steps=999999999 \
--max_steps=4945 \
--warmup_steps=494 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/guj_gujr_10mb.txt \
--seed=43 \
--override_n_examples=3956 \
--output_dir=models/10mb/guj_gujr_10mb
cp tokenizers/monolingual/guj_gujr_10mb/* models/10mb/guj_gujr_10mb

# guj_latn
if test -f models/10mb/guj_latn_10mb/pytorch_model.bin; then
echo "Model already found: guj_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/guj_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/guj_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=306 --save_steps=999999999 \
--max_steps=6126 \
--warmup_steps=612 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/guj_latn_10mb.txt \
--seed=43 \
--override_n_examples=4901 \
--output_dir=models/10mb/guj_latn_10mb
cp tokenizers/monolingual/guj_latn_10mb/* models/10mb/guj_latn_10mb

# hat_latn
if test -f models/10mb/hat_latn_10mb/pytorch_model.bin; then
echo "Model already found: hat_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hat_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hat_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=299 --save_steps=999999999 \
--max_steps=5996 \
--warmup_steps=599 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hat_latn_10mb.txt \
--seed=43 \
--override_n_examples=4797 \
--output_dir=models/10mb/hat_latn_10mb
cp tokenizers/monolingual/hat_latn_10mb/* models/10mb/hat_latn_10mb

# hau_latn
if test -f models/10mb/hau_latn_10mb/pytorch_model.bin; then
echo "Model already found: hau_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hau_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hau_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=346 --save_steps=999999999 \
--max_steps=6921 \
--warmup_steps=692 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hau_latn_10mb.txt \
--seed=43 \
--override_n_examples=5537 \
--output_dir=models/10mb/hau_latn_10mb
cp tokenizers/monolingual/hau_latn_10mb/* models/10mb/hau_latn_10mb

# haw_latn
if test -f models/10mb/haw_latn_10mb/pytorch_model.bin; then
echo "Model already found: haw_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/haw_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/haw_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=411 --save_steps=999999999 \
--max_steps=8233 \
--warmup_steps=823 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/haw_latn_10mb.txt \
--seed=43 \
--override_n_examples=6587 \
--output_dir=models/10mb/haw_latn_10mb
cp tokenizers/monolingual/haw_latn_10mb/* models/10mb/haw_latn_10mb

# heb_hebr
if test -f models/10mb/heb_hebr_10mb/pytorch_model.bin; then
echo "Model already found: heb_hebr_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/heb_hebr_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/heb_hebr_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=245 --save_steps=999999999 \
--max_steps=4907 \
--warmup_steps=490 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/heb_hebr_10mb.txt \
--seed=43 \
--override_n_examples=3926 \
--output_dir=models/10mb/heb_hebr_10mb
cp tokenizers/monolingual/heb_hebr_10mb/* models/10mb/heb_hebr_10mb

# hif_latn
if test -f models/10mb/hif_latn_10mb/pytorch_model.bin; then
echo "Model already found: hif_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hif_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hif_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=395 --save_steps=999999999 \
--max_steps=7907 \
--warmup_steps=790 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hif_latn_10mb.txt \
--seed=43 \
--override_n_examples=6326 \
--output_dir=models/10mb/hif_latn_10mb
cp tokenizers/monolingual/hif_latn_10mb/* models/10mb/hif_latn_10mb

# hil_latn
if test -f models/10mb/hil_latn_10mb/pytorch_model.bin; then
echo "Model already found: hil_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hil_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hil_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=351 --save_steps=999999999 \
--max_steps=7026 \
--warmup_steps=702 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hil_latn_10mb.txt \
--seed=43 \
--override_n_examples=5621 \
--output_dir=models/10mb/hil_latn_10mb
cp tokenizers/monolingual/hil_latn_10mb/* models/10mb/hil_latn_10mb

# hin_deva
if test -f models/10mb/hin_deva_10mb/pytorch_model.bin; then
echo "Model already found: hin_deva_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hin_deva_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hin_deva_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=290 --save_steps=999999999 \
--max_steps=5813 \
--warmup_steps=581 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hin_deva_10mb.txt \
--seed=43 \
--override_n_examples=4651 \
--output_dir=models/10mb/hin_deva_10mb
cp tokenizers/monolingual/hin_deva_10mb/* models/10mb/hin_deva_10mb

# hin_latn
if test -f models/10mb/hin_latn_10mb/pytorch_model.bin; then
echo "Model already found: hin_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hin_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hin_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=362 --save_steps=999999999 \
--max_steps=7256 \
--warmup_steps=725 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hin_latn_10mb.txt \
--seed=43 \
--override_n_examples=5805 \
--output_dir=models/10mb/hin_latn_10mb
cp tokenizers/monolingual/hin_latn_10mb/* models/10mb/hin_latn_10mb

# hmn_latn
if test -f models/10mb/hmn_latn_10mb/pytorch_model.bin; then
echo "Model already found: hmn_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hmn_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hmn_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=359 --save_steps=999999999 \
--max_steps=7183 \
--warmup_steps=718 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hmn_latn_10mb.txt \
--seed=43 \
--override_n_examples=5747 \
--output_dir=models/10mb/hmn_latn_10mb
cp tokenizers/monolingual/hmn_latn_10mb/* models/10mb/hmn_latn_10mb

# hrv_latn
if test -f models/10mb/hrv_latn_10mb/pytorch_model.bin; then
echo "Model already found: hrv_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hrv_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hrv_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=278 --save_steps=999999999 \
--max_steps=5577 \
--warmup_steps=557 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hrv_latn_10mb.txt \
--seed=43 \
--override_n_examples=4462 \
--output_dir=models/10mb/hrv_latn_10mb
cp tokenizers/monolingual/hrv_latn_10mb/* models/10mb/hrv_latn_10mb

# hsb_latn
if test -f models/10mb/hsb_latn_10mb/pytorch_model.bin; then
echo "Model already found: hsb_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hsb_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hsb_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5956 \
--warmup_steps=595 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hsb_latn_10mb.txt \
--seed=43 \
--override_n_examples=4765 \
--output_dir=models/10mb/hsb_latn_10mb
cp tokenizers/monolingual/hsb_latn_10mb/* models/10mb/hsb_latn_10mb

# hun_latn
if test -f models/10mb/hun_latn_10mb/pytorch_model.bin; then
echo "Model already found: hun_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hun_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hun_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=242 --save_steps=999999999 \
--max_steps=4840 \
--warmup_steps=484 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hun_latn_10mb.txt \
--seed=43 \
--override_n_examples=3872 \
--output_dir=models/10mb/hun_latn_10mb
cp tokenizers/monolingual/hun_latn_10mb/* models/10mb/hun_latn_10mb

# hye_armn
if test -f models/10mb/hye_armn_10mb/pytorch_model.bin; then
echo "Model already found: hye_armn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hye_armn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hye_armn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=257 --save_steps=999999999 \
--max_steps=5158 \
--warmup_steps=515 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hye_armn_10mb.txt \
--seed=43 \
--override_n_examples=4127 \
--output_dir=models/10mb/hye_armn_10mb
cp tokenizers/monolingual/hye_armn_10mb/* models/10mb/hye_armn_10mb

# iba_latn
if test -f models/10mb/iba_latn_10mb/pytorch_model.bin; then
echo "Model already found: iba_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/iba_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/iba_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=327 --save_steps=999999999 \
--max_steps=6551 \
--warmup_steps=655 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/iba_latn_10mb.txt \
--seed=43 \
--override_n_examples=5241 \
--output_dir=models/10mb/iba_latn_10mb
cp tokenizers/monolingual/iba_latn_10mb/* models/10mb/iba_latn_10mb

# ibo_latn
if test -f models/10mb/ibo_latn_10mb/pytorch_model.bin; then
echo "Model already found: ibo_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ibo_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ibo_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=383 --save_steps=999999999 \
--max_steps=7666 \
--warmup_steps=766 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ibo_latn_10mb.txt \
--seed=43 \
--override_n_examples=6133 \
--output_dir=models/10mb/ibo_latn_10mb
cp tokenizers/monolingual/ibo_latn_10mb/* models/10mb/ibo_latn_10mb

# ido_latn
if test -f models/10mb/ido_latn_10mb/pytorch_model.bin; then
echo "Model already found: ido_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ido_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ido_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=351 --save_steps=999999999 \
--max_steps=7026 \
--warmup_steps=702 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ido_latn_10mb.txt \
--seed=43 \
--override_n_examples=5621 \
--output_dir=models/10mb/ido_latn_10mb
cp tokenizers/monolingual/ido_latn_10mb/* models/10mb/ido_latn_10mb

# iku_cans
if test -f models/10mb/iku_cans_10mb/pytorch_model.bin; then
echo "Model already found: iku_cans_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/iku_cans_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/iku_cans_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=233 --save_steps=999999999 \
--max_steps=4678 \
--warmup_steps=467 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/iku_cans_10mb.txt \
--seed=43 \
--override_n_examples=3743 \
--output_dir=models/10mb/iku_cans_10mb
cp tokenizers/monolingual/iku_cans_10mb/* models/10mb/iku_cans_10mb

# ilo_latn
if test -f models/10mb/ilo_latn_10mb/pytorch_model.bin; then
echo "Model already found: ilo_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ilo_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ilo_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=327 --save_steps=999999999 \
--max_steps=6542 \
--warmup_steps=654 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ilo_latn_10mb.txt \
--seed=43 \
--override_n_examples=5234 \
--output_dir=models/10mb/ilo_latn_10mb
cp tokenizers/monolingual/ilo_latn_10mb/* models/10mb/ilo_latn_10mb

# ina_latn
if test -f models/10mb/ina_latn_10mb/pytorch_model.bin; then
echo "Model already found: ina_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ina_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ina_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=362 --save_steps=999999999 \
--max_steps=7240 \
--warmup_steps=724 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ina_latn_10mb.txt \
--seed=43 \
--override_n_examples=5792 \
--output_dir=models/10mb/ina_latn_10mb
cp tokenizers/monolingual/ina_latn_10mb/* models/10mb/ina_latn_10mb

# ind_latn
if test -f models/10mb/ind_latn_10mb/pytorch_model.bin; then
echo "Model already found: ind_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ind_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ind_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=264 --save_steps=999999999 \
--max_steps=5288 \
--warmup_steps=528 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ind_latn_10mb.txt \
--seed=43 \
--override_n_examples=4231 \
--output_dir=models/10mb/ind_latn_10mb
cp tokenizers/monolingual/ind_latn_10mb/* models/10mb/ind_latn_10mb

# isl_latn
if test -f models/10mb/isl_latn_10mb/pytorch_model.bin; then
echo "Model already found: isl_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/isl_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/isl_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=299 --save_steps=999999999 \
--max_steps=5983 \
--warmup_steps=598 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/isl_latn_10mb.txt \
--seed=43 \
--override_n_examples=4787 \
--output_dir=models/10mb/isl_latn_10mb
cp tokenizers/monolingual/isl_latn_10mb/* models/10mb/isl_latn_10mb

# ita_latn
if test -f models/10mb/ita_latn_10mb/pytorch_model.bin; then
echo "Model already found: ita_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ita_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ita_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=272 --save_steps=999999999 \
--max_steps=5446 \
--warmup_steps=544 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ita_latn_10mb.txt \
--seed=43 \
--override_n_examples=4357 \
--output_dir=models/10mb/ita_latn_10mb
cp tokenizers/monolingual/ita_latn_10mb/* models/10mb/ita_latn_10mb

# jav_latn
if test -f models/10mb/jav_latn_10mb/pytorch_model.bin; then
echo "Model already found: jav_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/jav_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/jav_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=310 --save_steps=999999999 \
--max_steps=6211 \
--warmup_steps=621 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/jav_latn_10mb.txt \
--seed=43 \
--override_n_examples=4969 \
--output_dir=models/10mb/jav_latn_10mb
cp tokenizers/monolingual/jav_latn_10mb/* models/10mb/jav_latn_10mb

# jpn_jpan
if test -f models/10mb/jpn_jpan_10mb/pytorch_model.bin; then
echo "Model already found: jpn_jpan_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/jpn_jpan_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/jpn_jpan_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=274 --save_steps=999999999 \
--max_steps=5492 \
--warmup_steps=549 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/jpn_jpan_10mb.txt \
--seed=43 \
--override_n_examples=4394 \
--output_dir=models/10mb/jpn_jpan_10mb
cp tokenizers/monolingual/jpn_jpan_10mb/* models/10mb/jpn_jpan_10mb

# kaa_cyrl
if test -f models/10mb/kaa_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: kaa_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaa_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaa_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=268 --save_steps=999999999 \
--max_steps=5362 \
--warmup_steps=536 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaa_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4290 \
--output_dir=models/10mb/kaa_cyrl_10mb
cp tokenizers/monolingual/kaa_cyrl_10mb/* models/10mb/kaa_cyrl_10mb

# kaa_latn
if test -f models/10mb/kaa_latn_10mb/pytorch_model.bin; then
echo "Model already found: kaa_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaa_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaa_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=299 --save_steps=999999999 \
--max_steps=5986 \
--warmup_steps=598 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaa_latn_10mb.txt \
--seed=43 \
--override_n_examples=4789 \
--output_dir=models/10mb/kaa_latn_10mb
cp tokenizers/monolingual/kaa_latn_10mb/* models/10mb/kaa_latn_10mb

# kab_latn
if test -f models/10mb/kab_latn_10mb/pytorch_model.bin; then
echo "Model already found: kab_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kab_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kab_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=386 --save_steps=999999999 \
--max_steps=7731 \
--warmup_steps=773 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kab_latn_10mb.txt \
--seed=43 \
--override_n_examples=6185 \
--output_dir=models/10mb/kab_latn_10mb
cp tokenizers/monolingual/kab_latn_10mb/* models/10mb/kab_latn_10mb

# kac_latn
if test -f models/10mb/kac_latn_10mb/pytorch_model.bin; then
echo "Model already found: kac_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kac_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kac_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8597 \
--warmup_steps=859 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kac_latn_10mb.txt \
--seed=43 \
--override_n_examples=6878 \
--output_dir=models/10mb/kac_latn_10mb
cp tokenizers/monolingual/kac_latn_10mb/* models/10mb/kac_latn_10mb

# kal_latn
if test -f models/10mb/kal_latn_10mb/pytorch_model.bin; then
echo "Model already found: kal_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kal_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kal_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=271 --save_steps=999999999 \
--max_steps=5431 \
--warmup_steps=543 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kal_latn_10mb.txt \
--seed=43 \
--override_n_examples=4345 \
--output_dir=models/10mb/kal_latn_10mb
cp tokenizers/monolingual/kal_latn_10mb/* models/10mb/kal_latn_10mb

# kan_knda
if test -f models/10mb/kan_knda_10mb/pytorch_model.bin; then
echo "Model already found: kan_knda_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kan_knda_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kan_knda_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=269 --save_steps=999999999 \
--max_steps=5396 \
--warmup_steps=539 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kan_knda_10mb.txt \
--seed=43 \
--override_n_examples=4317 \
--output_dir=models/10mb/kan_knda_10mb
cp tokenizers/monolingual/kan_knda_10mb/* models/10mb/kan_knda_10mb

# kat_geor
if test -f models/10mb/kat_geor_10mb/pytorch_model.bin; then
echo "Model already found: kat_geor_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kat_geor_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kat_geor_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=443 --save_steps=999999999 \
--max_steps=8861 \
--warmup_steps=886 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kat_geor_10mb.txt \
--seed=43 \
--override_n_examples=7089 \
--output_dir=models/10mb/kat_geor_10mb
cp tokenizers/monolingual/kat_geor_10mb/* models/10mb/kat_geor_10mb

# kaz_cyrl
if test -f models/10mb/kaz_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: kaz_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaz_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaz_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=253 --save_steps=999999999 \
--max_steps=5078 \
--warmup_steps=507 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaz_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4063 \
--output_dir=models/10mb/kaz_cyrl_10mb
cp tokenizers/monolingual/kaz_cyrl_10mb/* models/10mb/kaz_cyrl_10mb

# kbd_cyrl
if test -f models/10mb/kbd_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: kbd_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kbd_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kbd_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=348 --save_steps=999999999 \
--max_steps=6972 \
--warmup_steps=697 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kbd_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=5578 \
--output_dir=models/10mb/kbd_cyrl_10mb
cp tokenizers/monolingual/kbd_cyrl_10mb/* models/10mb/kbd_cyrl_10mb

# kbp_latn
if test -f models/10mb/kbp_latn_10mb/pytorch_model.bin; then
echo "Model already found: kbp_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kbp_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kbp_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=415 --save_steps=999999999 \
--max_steps=8303 \
--warmup_steps=830 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kbp_latn_10mb.txt \
--seed=43 \
--override_n_examples=6643 \
--output_dir=models/10mb/kbp_latn_10mb
cp tokenizers/monolingual/kbp_latn_10mb/* models/10mb/kbp_latn_10mb

# kea_latn
if test -f models/10mb/kea_latn_10mb/pytorch_model.bin; then
echo "Model already found: kea_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kea_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kea_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=254 --save_steps=999999999 \
--max_steps=5090 \
--warmup_steps=509 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kea_latn_10mb.txt \
--seed=43 \
--override_n_examples=4072 \
--output_dir=models/10mb/kea_latn_10mb
cp tokenizers/monolingual/kea_latn_10mb/* models/10mb/kea_latn_10mb

# kha_latn
if test -f models/10mb/kha_latn_10mb/pytorch_model.bin; then
echo "Model already found: kha_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kha_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kha_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=373 --save_steps=999999999 \
--max_steps=7468 \
--warmup_steps=746 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kha_latn_10mb.txt \
--seed=43 \
--override_n_examples=5975 \
--output_dir=models/10mb/kha_latn_10mb
cp tokenizers/monolingual/kha_latn_10mb/* models/10mb/kha_latn_10mb

# khk_cyrl
if test -f models/10mb/khk_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: khk_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/khk_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/khk_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=276 --save_steps=999999999 \
--max_steps=5522 \
--warmup_steps=552 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/khk_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4418 \
--output_dir=models/10mb/khk_cyrl_10mb
cp tokenizers/monolingual/khk_cyrl_10mb/* models/10mb/khk_cyrl_10mb

# khm_khmr
if test -f models/10mb/khm_khmr_10mb/pytorch_model.bin; then
echo "Model already found: khm_khmr_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/khm_khmr_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/khm_khmr_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=392 --save_steps=999999999 \
--max_steps=7842 \
--warmup_steps=784 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/khm_khmr_10mb.txt \
--seed=43 \
--override_n_examples=6274 \
--output_dir=models/10mb/khm_khmr_10mb
cp tokenizers/monolingual/khm_khmr_10mb/* models/10mb/khm_khmr_10mb

# kin_latn
if test -f models/10mb/kin_latn_10mb/pytorch_model.bin; then
echo "Model already found: kin_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kin_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kin_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=303 --save_steps=999999999 \
--max_steps=6060 \
--warmup_steps=606 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kin_latn_10mb.txt \
--seed=43 \
--override_n_examples=4848 \
--output_dir=models/10mb/kin_latn_10mb
cp tokenizers/monolingual/kin_latn_10mb/* models/10mb/kin_latn_10mb

# kir_cyrl
if test -f models/10mb/kir_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: kir_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kir_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kir_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=281 --save_steps=999999999 \
--max_steps=5638 \
--warmup_steps=563 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kir_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4511 \
--output_dir=models/10mb/kir_cyrl_10mb
cp tokenizers/monolingual/kir_cyrl_10mb/* models/10mb/kir_cyrl_10mb

# kmb_latn
if test -f models/10mb/kmb_latn_10mb/pytorch_model.bin; then
echo "Model already found: kmb_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kmb_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kmb_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7417 \
--warmup_steps=741 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kmb_latn_10mb.txt \
--seed=43 \
--override_n_examples=5934 \
--output_dir=models/10mb/kmb_latn_10mb
cp tokenizers/monolingual/kmb_latn_10mb/* models/10mb/kmb_latn_10mb

# kmr_latn
if test -f models/10mb/kmr_latn_10mb/pytorch_model.bin; then
echo "Model already found: kmr_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kmr_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kmr_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=288 --save_steps=999999999 \
--max_steps=5777 \
--warmup_steps=577 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kmr_latn_10mb.txt \
--seed=43 \
--override_n_examples=4622 \
--output_dir=models/10mb/kmr_latn_10mb
cp tokenizers/monolingual/kmr_latn_10mb/* models/10mb/kmr_latn_10mb

# knc_arab
if test -f models/10mb/knc_arab_10mb/pytorch_model.bin; then
echo "Model already found: knc_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/knc_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/knc_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=1302 --save_steps=999999999 \
--max_steps=26042 \
--warmup_steps=2604 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/knc_arab_10mb.txt \
--seed=43 \
--override_n_examples=20834 \
--output_dir=models/10mb/knc_arab_10mb
cp tokenizers/monolingual/knc_arab_10mb/* models/10mb/knc_arab_10mb

# knc_latn
if test -f models/10mb/knc_latn_10mb/pytorch_model.bin; then
echo "Model already found: knc_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/knc_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/knc_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8333 \
--warmup_steps=833 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/knc_latn_10mb.txt \
--seed=43 \
--override_n_examples=6667 \
--output_dir=models/10mb/knc_latn_10mb
cp tokenizers/monolingual/knc_latn_10mb/* models/10mb/knc_latn_10mb

# kom_cyrl
if test -f models/10mb/kom_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: kom_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kom_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kom_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=320 --save_steps=999999999 \
--max_steps=6407 \
--warmup_steps=640 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kom_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=5126 \
--output_dir=models/10mb/kom_cyrl_10mb
cp tokenizers/monolingual/kom_cyrl_10mb/* models/10mb/kom_cyrl_10mb

# kon_latn
if test -f models/10mb/kon_latn_10mb/pytorch_model.bin; then
echo "Model already found: kon_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kon_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kon_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7683 \
--warmup_steps=768 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kon_latn_10mb.txt \
--seed=43 \
--override_n_examples=6147 \
--output_dir=models/10mb/kon_latn_10mb
cp tokenizers/monolingual/kon_latn_10mb/* models/10mb/kon_latn_10mb

# kor_hang
if test -f models/10mb/kor_hang_10mb/pytorch_model.bin; then
echo "Model already found: kor_hang_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kor_hang_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kor_hang_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=291 --save_steps=999999999 \
--max_steps=5831 \
--warmup_steps=583 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kor_hang_10mb.txt \
--seed=43 \
--override_n_examples=4665 \
--output_dir=models/10mb/kor_hang_10mb
cp tokenizers/monolingual/kor_hang_10mb/* models/10mb/kor_hang_10mb

# krc_cyrl
if test -f models/10mb/krc_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: krc_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/krc_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/krc_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=272 --save_steps=999999999 \
--max_steps=5458 \
--warmup_steps=545 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/krc_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4367 \
--output_dir=models/10mb/krc_cyrl_10mb
cp tokenizers/monolingual/krc_cyrl_10mb/* models/10mb/krc_cyrl_10mb

# kum_cyrl
if test -f models/10mb/kum_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: kum_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kum_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kum_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4910 \
--warmup_steps=491 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kum_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=3928 \
--output_dir=models/10mb/kum_cyrl_10mb
cp tokenizers/monolingual/kum_cyrl_10mb/* models/10mb/kum_cyrl_10mb

# kur_arab
if test -f models/10mb/kur_arab_10mb/pytorch_model.bin; then
echo "Model already found: kur_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kur_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kur_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=274 --save_steps=999999999 \
--max_steps=5497 \
--warmup_steps=549 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kur_arab_10mb.txt \
--seed=43 \
--override_n_examples=4398 \
--output_dir=models/10mb/kur_arab_10mb
cp tokenizers/monolingual/kur_arab_10mb/* models/10mb/kur_arab_10mb

# kur_latn
if test -f models/10mb/kur_latn_10mb/pytorch_model.bin; then
echo "Model already found: kur_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kur_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kur_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=351 --save_steps=999999999 \
--max_steps=7030 \
--warmup_steps=703 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kur_latn_10mb.txt \
--seed=43 \
--override_n_examples=5624 \
--output_dir=models/10mb/kur_latn_10mb
cp tokenizers/monolingual/kur_latn_10mb/* models/10mb/kur_latn_10mb

# lao_laoo
if test -f models/10mb/lao_laoo_10mb/pytorch_model.bin; then
echo "Model already found: lao_laoo_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lao_laoo_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lao_laoo_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=289 --save_steps=999999999 \
--max_steps=5788 \
--warmup_steps=578 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lao_laoo_10mb.txt \
--seed=43 \
--override_n_examples=4631 \
--output_dir=models/10mb/lao_laoo_10mb
cp tokenizers/monolingual/lao_laoo_10mb/* models/10mb/lao_laoo_10mb

# lat_latn
if test -f models/10mb/lat_latn_10mb/pytorch_model.bin; then
echo "Model already found: lat_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lat_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lat_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=239 --save_steps=999999999 \
--max_steps=4781 \
--warmup_steps=478 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lat_latn_10mb.txt \
--seed=43 \
--override_n_examples=3825 \
--output_dir=models/10mb/lat_latn_10mb
cp tokenizers/monolingual/lat_latn_10mb/* models/10mb/lat_latn_10mb

# lav_latn
if test -f models/10mb/lav_latn_10mb/pytorch_model.bin; then
echo "Model already found: lav_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lav_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lav_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=307 --save_steps=999999999 \
--max_steps=6157 \
--warmup_steps=615 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lav_latn_10mb.txt \
--seed=43 \
--override_n_examples=4926 \
--output_dir=models/10mb/lav_latn_10mb
cp tokenizers/monolingual/lav_latn_10mb/* models/10mb/lav_latn_10mb

# lbe_cyrl
if test -f models/10mb/lbe_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: lbe_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lbe_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lbe_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5907 \
--warmup_steps=590 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lbe_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4726 \
--output_dir=models/10mb/lbe_cyrl_10mb
cp tokenizers/monolingual/lbe_cyrl_10mb/* models/10mb/lbe_cyrl_10mb

# lij_latn
if test -f models/10mb/lij_latn_10mb/pytorch_model.bin; then
echo "Model already found: lij_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lij_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lij_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=380 --save_steps=999999999 \
--max_steps=7601 \
--warmup_steps=760 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lij_latn_10mb.txt \
--seed=43 \
--override_n_examples=6081 \
--output_dir=models/10mb/lij_latn_10mb
cp tokenizers/monolingual/lij_latn_10mb/* models/10mb/lij_latn_10mb

# lim_latn
if test -f models/10mb/lim_latn_10mb/pytorch_model.bin; then
echo "Model already found: lim_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lim_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lim_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=349 --save_steps=999999999 \
--max_steps=6980 \
--warmup_steps=698 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lim_latn_10mb.txt \
--seed=43 \
--override_n_examples=5584 \
--output_dir=models/10mb/lim_latn_10mb
cp tokenizers/monolingual/lim_latn_10mb/* models/10mb/lim_latn_10mb

# lin_latn
if test -f models/10mb/lin_latn_10mb/pytorch_model.bin; then
echo "Model already found: lin_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lin_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lin_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=349 --save_steps=999999999 \
--max_steps=6991 \
--warmup_steps=699 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lin_latn_10mb.txt \
--seed=43 \
--override_n_examples=5593 \
--output_dir=models/10mb/lin_latn_10mb
cp tokenizers/monolingual/lin_latn_10mb/* models/10mb/lin_latn_10mb

# lit_latn
if test -f models/10mb/lit_latn_10mb/pytorch_model.bin; then
echo "Model already found: lit_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lit_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lit_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=256 --save_steps=999999999 \
--max_steps=5132 \
--warmup_steps=513 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lit_latn_10mb.txt \
--seed=43 \
--override_n_examples=4106 \
--output_dir=models/10mb/lit_latn_10mb
cp tokenizers/monolingual/lit_latn_10mb/* models/10mb/lit_latn_10mb

# lmo_latn
if test -f models/10mb/lmo_latn_10mb/pytorch_model.bin; then
echo "Model already found: lmo_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lmo_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lmo_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=360 --save_steps=999999999 \
--max_steps=7217 \
--warmup_steps=721 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lmo_latn_10mb.txt \
--seed=43 \
--override_n_examples=5774 \
--output_dir=models/10mb/lmo_latn_10mb
cp tokenizers/monolingual/lmo_latn_10mb/* models/10mb/lmo_latn_10mb

# ltg_latn
if test -f models/10mb/ltg_latn_10mb/pytorch_model.bin; then
echo "Model already found: ltg_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ltg_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ltg_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=268 --save_steps=999999999 \
--max_steps=5373 \
--warmup_steps=537 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ltg_latn_10mb.txt \
--seed=43 \
--override_n_examples=4299 \
--output_dir=models/10mb/ltg_latn_10mb
cp tokenizers/monolingual/ltg_latn_10mb/* models/10mb/ltg_latn_10mb

# ltz_latn
if test -f models/10mb/ltz_latn_10mb/pytorch_model.bin; then
echo "Model already found: ltz_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ltz_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ltz_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=345 --save_steps=999999999 \
--max_steps=6905 \
--warmup_steps=690 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ltz_latn_10mb.txt \
--seed=43 \
--override_n_examples=5524 \
--output_dir=models/10mb/ltz_latn_10mb
cp tokenizers/monolingual/ltz_latn_10mb/* models/10mb/ltz_latn_10mb

# lua_latn
if test -f models/10mb/lua_latn_10mb/pytorch_model.bin; then
echo "Model already found: lua_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lua_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lua_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=376 --save_steps=999999999 \
--max_steps=7528 \
--warmup_steps=752 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lua_latn_10mb.txt \
--seed=43 \
--override_n_examples=6023 \
--output_dir=models/10mb/lua_latn_10mb
cp tokenizers/monolingual/lua_latn_10mb/* models/10mb/lua_latn_10mb

# lug_latn
if test -f models/10mb/lug_latn_10mb/pytorch_model.bin; then
echo "Model already found: lug_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lug_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lug_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=355 --save_steps=999999999 \
--max_steps=7117 \
--warmup_steps=711 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lug_latn_10mb.txt \
--seed=43 \
--override_n_examples=5694 \
--output_dir=models/10mb/lug_latn_10mb
cp tokenizers/monolingual/lug_latn_10mb/* models/10mb/lug_latn_10mb

# luo_latn
if test -f models/10mb/luo_latn_10mb/pytorch_model.bin; then
echo "Model already found: luo_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/luo_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/luo_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=355 --save_steps=999999999 \
--max_steps=7110 \
--warmup_steps=711 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/luo_latn_10mb.txt \
--seed=43 \
--override_n_examples=5688 \
--output_dir=models/10mb/luo_latn_10mb
cp tokenizers/monolingual/luo_latn_10mb/* models/10mb/luo_latn_10mb

# lus_latn
if test -f models/10mb/lus_latn_10mb/pytorch_model.bin; then
echo "Model already found: lus_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lus_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lus_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=367 --save_steps=999999999 \
--max_steps=7342 \
--warmup_steps=734 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lus_latn_10mb.txt \
--seed=43 \
--override_n_examples=5874 \
--output_dir=models/10mb/lus_latn_10mb
cp tokenizers/monolingual/lus_latn_10mb/* models/10mb/lus_latn_10mb

# lvs_latn
if test -f models/10mb/lvs_latn_10mb/pytorch_model.bin; then
echo "Model already found: lvs_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lvs_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lvs_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=297 --save_steps=999999999 \
--max_steps=5953 \
--warmup_steps=595 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lvs_latn_10mb.txt \
--seed=43 \
--override_n_examples=4763 \
--output_dir=models/10mb/lvs_latn_10mb
cp tokenizers/monolingual/lvs_latn_10mb/* models/10mb/lvs_latn_10mb

# lzh_hant
if test -f models/10mb/lzh_hant_10mb/pytorch_model.bin; then
echo "Model already found: lzh_hant_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lzh_hant_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lzh_hant_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4495 \
--warmup_steps=449 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lzh_hant_10mb.txt \
--seed=43 \
--override_n_examples=3596 \
--output_dir=models/10mb/lzh_hant_10mb
cp tokenizers/monolingual/lzh_hant_10mb/* models/10mb/lzh_hant_10mb

# mai_deva
if test -f models/10mb/mai_deva_10mb/pytorch_model.bin; then
echo "Model already found: mai_deva_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mai_deva_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mai_deva_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=333 --save_steps=999999999 \
--max_steps=6661 \
--warmup_steps=666 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mai_deva_10mb.txt \
--seed=43 \
--override_n_examples=5329 \
--output_dir=models/10mb/mai_deva_10mb
cp tokenizers/monolingual/mai_deva_10mb/* models/10mb/mai_deva_10mb

# mal_mlym
if test -f models/10mb/mal_mlym_10mb/pytorch_model.bin; then
echo "Model already found: mal_mlym_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mal_mlym_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mal_mlym_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=307 --save_steps=999999999 \
--max_steps=6153 \
--warmup_steps=615 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mal_mlym_10mb.txt \
--seed=43 \
--override_n_examples=4923 \
--output_dir=models/10mb/mal_mlym_10mb
cp tokenizers/monolingual/mal_mlym_10mb/* models/10mb/mal_mlym_10mb

# mar_deva
if test -f models/10mb/mar_deva_10mb/pytorch_model.bin; then
echo "Model already found: mar_deva_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mar_deva_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mar_deva_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=264 --save_steps=999999999 \
--max_steps=5285 \
--warmup_steps=528 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mar_deva_10mb.txt \
--seed=43 \
--override_n_examples=4228 \
--output_dir=models/10mb/mar_deva_10mb
cp tokenizers/monolingual/mar_deva_10mb/* models/10mb/mar_deva_10mb

# mhr_cyrl
if test -f models/10mb/mhr_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: mhr_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mhr_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mhr_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=293 --save_steps=999999999 \
--max_steps=5875 \
--warmup_steps=587 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mhr_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4700 \
--output_dir=models/10mb/mhr_cyrl_10mb
cp tokenizers/monolingual/mhr_cyrl_10mb/* models/10mb/mhr_cyrl_10mb

# min_latn
if test -f models/10mb/min_latn_10mb/pytorch_model.bin; then
echo "Model already found: min_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/min_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/min_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=299 --save_steps=999999999 \
--max_steps=5986 \
--warmup_steps=598 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/min_latn_10mb.txt \
--seed=43 \
--override_n_examples=4789 \
--output_dir=models/10mb/min_latn_10mb
cp tokenizers/monolingual/min_latn_10mb/* models/10mb/min_latn_10mb

# mkd_cyrl
if test -f models/10mb/mkd_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: mkd_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mkd_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mkd_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=279 --save_steps=999999999 \
--max_steps=5582 \
--warmup_steps=558 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mkd_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4466 \
--output_dir=models/10mb/mkd_cyrl_10mb
cp tokenizers/monolingual/mkd_cyrl_10mb/* models/10mb/mkd_cyrl_10mb

# mlg_latn
if test -f models/10mb/mlg_latn_10mb/pytorch_model.bin; then
echo "Model already found: mlg_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mlg_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mlg_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=366 --save_steps=999999999 \
--max_steps=7331 \
--warmup_steps=733 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mlg_latn_10mb.txt \
--seed=43 \
--override_n_examples=5865 \
--output_dir=models/10mb/mlg_latn_10mb
cp tokenizers/monolingual/mlg_latn_10mb/* models/10mb/mlg_latn_10mb

# mlt_latn
if test -f models/10mb/mlt_latn_10mb/pytorch_model.bin; then
echo "Model already found: mlt_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mlt_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mlt_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=357 --save_steps=999999999 \
--max_steps=7147 \
--warmup_steps=714 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mlt_latn_10mb.txt \
--seed=43 \
--override_n_examples=5718 \
--output_dir=models/10mb/mlt_latn_10mb
cp tokenizers/monolingual/mlt_latn_10mb/* models/10mb/mlt_latn_10mb

# mon_cyrl
if test -f models/10mb/mon_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: mon_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mon_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mon_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=263 --save_steps=999999999 \
--max_steps=5276 \
--warmup_steps=527 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mon_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4221 \
--output_dir=models/10mb/mon_cyrl_10mb
cp tokenizers/monolingual/mon_cyrl_10mb/* models/10mb/mon_cyrl_10mb

# mon_latn
if test -f models/10mb/mon_latn_10mb/pytorch_model.bin; then
echo "Model already found: mon_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mon_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mon_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=324 --save_steps=999999999 \
--max_steps=6497 \
--warmup_steps=649 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mon_latn_10mb.txt \
--seed=43 \
--override_n_examples=5198 \
--output_dir=models/10mb/mon_latn_10mb
cp tokenizers/monolingual/mon_latn_10mb/* models/10mb/mon_latn_10mb

# mos_latn
if test -f models/10mb/mos_latn_10mb/pytorch_model.bin; then
echo "Model already found: mos_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mos_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mos_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=8342 \
--warmup_steps=834 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mos_latn_10mb.txt \
--seed=43 \
--override_n_examples=6674 \
--output_dir=models/10mb/mos_latn_10mb
cp tokenizers/monolingual/mos_latn_10mb/* models/10mb/mos_latn_10mb

# mri_latn
if test -f models/10mb/mri_latn_10mb/pytorch_model.bin; then
echo "Model already found: mri_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mri_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mri_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=376 --save_steps=999999999 \
--max_steps=7527 \
--warmup_steps=752 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mri_latn_10mb.txt \
--seed=43 \
--override_n_examples=6022 \
--output_dir=models/10mb/mri_latn_10mb
cp tokenizers/monolingual/mri_latn_10mb/* models/10mb/mri_latn_10mb

# msa_latn
if test -f models/10mb/msa_latn_10mb/pytorch_model.bin; then
echo "Model already found: msa_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/msa_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/msa_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=297 --save_steps=999999999 \
--max_steps=5941 \
--warmup_steps=594 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/msa_latn_10mb.txt \
--seed=43 \
--override_n_examples=4753 \
--output_dir=models/10mb/msa_latn_10mb
cp tokenizers/monolingual/msa_latn_10mb/* models/10mb/msa_latn_10mb

# mya_mymr
if test -f models/10mb/mya_mymr_10mb/pytorch_model.bin; then
echo "Model already found: mya_mymr_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mya_mymr_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mya_mymr_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=511 --save_steps=999999999 \
--max_steps=10222 \
--warmup_steps=1022 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mya_mymr_10mb.txt \
--seed=43 \
--override_n_examples=8178 \
--output_dir=models/10mb/mya_mymr_10mb
cp tokenizers/monolingual/mya_mymr_10mb/* models/10mb/mya_mymr_10mb

# myv_cyrl
if test -f models/10mb/myv_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: myv_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/myv_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/myv_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=284 --save_steps=999999999 \
--max_steps=5692 \
--warmup_steps=569 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/myv_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4554 \
--output_dir=models/10mb/myv_cyrl_10mb
cp tokenizers/monolingual/myv_cyrl_10mb/* models/10mb/myv_cyrl_10mb

# nan_latn
if test -f models/10mb/nan_latn_10mb/pytorch_model.bin; then
echo "Model already found: nan_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nan_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nan_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=464 --save_steps=999999999 \
--max_steps=9287 \
--warmup_steps=928 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nan_latn_10mb.txt \
--seed=43 \
--override_n_examples=7430 \
--output_dir=models/10mb/nan_latn_10mb
cp tokenizers/monolingual/nan_latn_10mb/* models/10mb/nan_latn_10mb

# nde_latn
if test -f models/10mb/nde_latn_10mb/pytorch_model.bin; then
echo "Model already found: nde_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nde_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nde_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4252 \
--warmup_steps=425 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nde_latn_10mb.txt \
--seed=43 \
--override_n_examples=3402 \
--output_dir=models/10mb/nde_latn_10mb
cp tokenizers/monolingual/nde_latn_10mb/* models/10mb/nde_latn_10mb

# nds_latn
if test -f models/10mb/nds_latn_10mb/pytorch_model.bin; then
echo "Model already found: nds_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nds_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nds_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=339 --save_steps=999999999 \
--max_steps=6798 \
--warmup_steps=679 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nds_latn_10mb.txt \
--seed=43 \
--override_n_examples=5439 \
--output_dir=models/10mb/nds_latn_10mb
cp tokenizers/monolingual/nds_latn_10mb/* models/10mb/nds_latn_10mb

# nep_deva
if test -f models/10mb/nep_deva_10mb/pytorch_model.bin; then
echo "Model already found: nep_deva_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nep_deva_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nep_deva_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=276 --save_steps=999999999 \
--max_steps=5527 \
--warmup_steps=552 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nep_deva_10mb.txt \
--seed=43 \
--override_n_examples=4422 \
--output_dir=models/10mb/nep_deva_10mb
cp tokenizers/monolingual/nep_deva_10mb/* models/10mb/nep_deva_10mb

# new_deva
if test -f models/10mb/new_deva_10mb/pytorch_model.bin; then
echo "Model already found: new_deva_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/new_deva_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/new_deva_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5802 \
--warmup_steps=580 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/new_deva_10mb.txt \
--seed=43 \
--override_n_examples=4642 \
--output_dir=models/10mb/new_deva_10mb
cp tokenizers/monolingual/new_deva_10mb/* models/10mb/new_deva_10mb

# nld_latn
if test -f models/10mb/nld_latn_10mb/pytorch_model.bin; then
echo "Model already found: nld_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nld_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nld_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=272 --save_steps=999999999 \
--max_steps=5440 \
--warmup_steps=544 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nld_latn_10mb.txt \
--seed=43 \
--override_n_examples=4352 \
--output_dir=models/10mb/nld_latn_10mb
cp tokenizers/monolingual/nld_latn_10mb/* models/10mb/nld_latn_10mb

# nno_latn
if test -f models/10mb/nno_latn_10mb/pytorch_model.bin; then
echo "Model already found: nno_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nno_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nno_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=292 --save_steps=999999999 \
--max_steps=5845 \
--warmup_steps=584 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nno_latn_10mb.txt \
--seed=43 \
--override_n_examples=4676 \
--output_dir=models/10mb/nno_latn_10mb
cp tokenizers/monolingual/nno_latn_10mb/* models/10mb/nno_latn_10mb

# nob_latn
if test -f models/10mb/nob_latn_10mb/pytorch_model.bin; then
echo "Model already found: nob_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nob_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nob_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=259 --save_steps=999999999 \
--max_steps=5185 \
--warmup_steps=518 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nob_latn_10mb.txt \
--seed=43 \
--override_n_examples=4148 \
--output_dir=models/10mb/nob_latn_10mb
cp tokenizers/monolingual/nob_latn_10mb/* models/10mb/nob_latn_10mb

# nor_latn
if test -f models/10mb/nor_latn_10mb/pytorch_model.bin; then
echo "Model already found: nor_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nor_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nor_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=319 --save_steps=999999999 \
--max_steps=6386 \
--warmup_steps=638 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nor_latn_10mb.txt \
--seed=43 \
--override_n_examples=5109 \
--output_dir=models/10mb/nor_latn_10mb
cp tokenizers/monolingual/nor_latn_10mb/* models/10mb/nor_latn_10mb

# nso_latn
if test -f models/10mb/nso_latn_10mb/pytorch_model.bin; then
echo "Model already found: nso_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nso_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nso_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=367 --save_steps=999999999 \
--max_steps=7340 \
--warmup_steps=734 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nso_latn_10mb.txt \
--seed=43 \
--override_n_examples=5872 \
--output_dir=models/10mb/nso_latn_10mb
cp tokenizers/monolingual/nso_latn_10mb/* models/10mb/nso_latn_10mb

# nya_latn
if test -f models/10mb/nya_latn_10mb/pytorch_model.bin; then
echo "Model already found: nya_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nya_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nya_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=317 --save_steps=999999999 \
--max_steps=6358 \
--warmup_steps=635 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nya_latn_10mb.txt \
--seed=43 \
--override_n_examples=5087 \
--output_dir=models/10mb/nya_latn_10mb
cp tokenizers/monolingual/nya_latn_10mb/* models/10mb/nya_latn_10mb

# oci_latn
if test -f models/10mb/oci_latn_10mb/pytorch_model.bin; then
echo "Model already found: oci_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/oci_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/oci_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=335 --save_steps=999999999 \
--max_steps=6702 \
--warmup_steps=670 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/oci_latn_10mb.txt \
--seed=43 \
--override_n_examples=5362 \
--output_dir=models/10mb/oci_latn_10mb
cp tokenizers/monolingual/oci_latn_10mb/* models/10mb/oci_latn_10mb

# ori_orya
if test -f models/10mb/ori_orya_10mb/pytorch_model.bin; then
echo "Model already found: ori_orya_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ori_orya_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ori_orya_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=270 --save_steps=999999999 \
--max_steps=5408 \
--warmup_steps=540 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ori_orya_10mb.txt \
--seed=43 \
--override_n_examples=4327 \
--output_dir=models/10mb/ori_orya_10mb
cp tokenizers/monolingual/ori_orya_10mb/* models/10mb/ori_orya_10mb

# orm_latn
if test -f models/10mb/orm_latn_10mb/pytorch_model.bin; then
echo "Model already found: orm_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/orm_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/orm_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=361 --save_steps=999999999 \
--max_steps=7238 \
--warmup_steps=723 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/orm_latn_10mb.txt \
--seed=43 \
--override_n_examples=5791 \
--output_dir=models/10mb/orm_latn_10mb
cp tokenizers/monolingual/orm_latn_10mb/* models/10mb/orm_latn_10mb

# oss_cyrl
if test -f models/10mb/oss_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: oss_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/oss_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/oss_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=450 --save_steps=999999999 \
--max_steps=9003 \
--warmup_steps=900 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/oss_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=7203 \
--output_dir=models/10mb/oss_cyrl_10mb
cp tokenizers/monolingual/oss_cyrl_10mb/* models/10mb/oss_cyrl_10mb

# otq_latn
if test -f models/10mb/otq_latn_10mb/pytorch_model.bin; then
echo "Model already found: otq_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/otq_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/otq_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=402 --save_steps=999999999 \
--max_steps=8052 \
--warmup_steps=805 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/otq_latn_10mb.txt \
--seed=43 \
--override_n_examples=6442 \
--output_dir=models/10mb/otq_latn_10mb
cp tokenizers/monolingual/otq_latn_10mb/* models/10mb/otq_latn_10mb

# pag_latn
if test -f models/10mb/pag_latn_10mb/pytorch_model.bin; then
echo "Model already found: pag_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pag_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pag_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=357 --save_steps=999999999 \
--max_steps=7146 \
--warmup_steps=714 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pag_latn_10mb.txt \
--seed=43 \
--override_n_examples=5717 \
--output_dir=models/10mb/pag_latn_10mb
cp tokenizers/monolingual/pag_latn_10mb/* models/10mb/pag_latn_10mb

# pam_latn
if test -f models/10mb/pam_latn_10mb/pytorch_model.bin; then
echo "Model already found: pam_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pam_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pam_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=372 --save_steps=999999999 \
--max_steps=7445 \
--warmup_steps=744 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pam_latn_10mb.txt \
--seed=43 \
--override_n_examples=5956 \
--output_dir=models/10mb/pam_latn_10mb
cp tokenizers/monolingual/pam_latn_10mb/* models/10mb/pam_latn_10mb

# pan_guru
if test -f models/10mb/pan_guru_10mb/pytorch_model.bin; then
echo "Model already found: pan_guru_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pan_guru_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pan_guru_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=272 --save_steps=999999999 \
--max_steps=5456 \
--warmup_steps=545 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pan_guru_10mb.txt \
--seed=43 \
--override_n_examples=4365 \
--output_dir=models/10mb/pan_guru_10mb
cp tokenizers/monolingual/pan_guru_10mb/* models/10mb/pan_guru_10mb

# pap_latn
if test -f models/10mb/pap_latn_10mb/pytorch_model.bin; then
echo "Model already found: pap_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pap_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pap_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=293 --save_steps=999999999 \
--max_steps=5871 \
--warmup_steps=587 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pap_latn_10mb.txt \
--seed=43 \
--override_n_examples=4697 \
--output_dir=models/10mb/pap_latn_10mb
cp tokenizers/monolingual/pap_latn_10mb/* models/10mb/pap_latn_10mb

# pbt_arab
if test -f models/10mb/pbt_arab_10mb/pytorch_model.bin; then
echo "Model already found: pbt_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pbt_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pbt_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=340 --save_steps=999999999 \
--max_steps=6811 \
--warmup_steps=681 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pbt_arab_10mb.txt \
--seed=43 \
--override_n_examples=5449 \
--output_dir=models/10mb/pbt_arab_10mb
cp tokenizers/monolingual/pbt_arab_10mb/* models/10mb/pbt_arab_10mb

# pcm_latn
if test -f models/10mb/pcm_latn_10mb/pytorch_model.bin; then
echo "Model already found: pcm_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pcm_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pcm_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=267 --save_steps=999999999 \
--max_steps=5346 \
--warmup_steps=534 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pcm_latn_10mb.txt \
--seed=43 \
--override_n_examples=4277 \
--output_dir=models/10mb/pcm_latn_10mb
cp tokenizers/monolingual/pcm_latn_10mb/* models/10mb/pcm_latn_10mb

# pes_arab
if test -f models/10mb/pes_arab_10mb/pytorch_model.bin; then
echo "Model already found: pes_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pes_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pes_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=270 --save_steps=999999999 \
--max_steps=5418 \
--warmup_steps=541 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pes_arab_10mb.txt \
--seed=43 \
--override_n_examples=4335 \
--output_dir=models/10mb/pes_arab_10mb
cp tokenizers/monolingual/pes_arab_10mb/* models/10mb/pes_arab_10mb

# plt_latn
if test -f models/10mb/plt_latn_10mb/pytorch_model.bin; then
echo "Model already found: plt_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/plt_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/plt_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=329 --save_steps=999999999 \
--max_steps=6595 \
--warmup_steps=659 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/plt_latn_10mb.txt \
--seed=43 \
--override_n_examples=5276 \
--output_dir=models/10mb/plt_latn_10mb
cp tokenizers/monolingual/plt_latn_10mb/* models/10mb/plt_latn_10mb

# pms_latn
if test -f models/10mb/pms_latn_10mb/pytorch_model.bin; then
echo "Model already found: pms_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pms_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pms_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=391 --save_steps=999999999 \
--max_steps=7835 \
--warmup_steps=783 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pms_latn_10mb.txt \
--seed=43 \
--override_n_examples=6268 \
--output_dir=models/10mb/pms_latn_10mb
cp tokenizers/monolingual/pms_latn_10mb/* models/10mb/pms_latn_10mb

# pnb_arab
if test -f models/10mb/pnb_arab_10mb/pytorch_model.bin; then
echo "Model already found: pnb_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pnb_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pnb_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=313 --save_steps=999999999 \
--max_steps=6261 \
--warmup_steps=626 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pnb_arab_10mb.txt \
--seed=43 \
--override_n_examples=5009 \
--output_dir=models/10mb/pnb_arab_10mb
cp tokenizers/monolingual/pnb_arab_10mb/* models/10mb/pnb_arab_10mb

# pol_latn
if test -f models/10mb/pol_latn_10mb/pytorch_model.bin; then
echo "Model already found: pol_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pol_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pol_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=274 --save_steps=999999999 \
--max_steps=5487 \
--warmup_steps=548 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pol_latn_10mb.txt \
--seed=43 \
--override_n_examples=4390 \
--output_dir=models/10mb/pol_latn_10mb
cp tokenizers/monolingual/pol_latn_10mb/* models/10mb/pol_latn_10mb

# por_latn
if test -f models/10mb/por_latn_10mb/pytorch_model.bin; then
echo "Model already found: por_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/por_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/por_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=281 --save_steps=999999999 \
--max_steps=5625 \
--warmup_steps=562 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/por_latn_10mb.txt \
--seed=43 \
--override_n_examples=4500 \
--output_dir=models/10mb/por_latn_10mb
cp tokenizers/monolingual/por_latn_10mb/* models/10mb/por_latn_10mb

# prs_arab
if test -f models/10mb/prs_arab_10mb/pytorch_model.bin; then
echo "Model already found: prs_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/prs_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/prs_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=289 --save_steps=999999999 \
--max_steps=5781 \
--warmup_steps=578 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/prs_arab_10mb.txt \
--seed=43 \
--override_n_examples=4625 \
--output_dir=models/10mb/prs_arab_10mb
cp tokenizers/monolingual/prs_arab_10mb/* models/10mb/prs_arab_10mb

# pus_arab
if test -f models/10mb/pus_arab_10mb/pytorch_model.bin; then
echo "Model already found: pus_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pus_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pus_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=302 --save_steps=999999999 \
--max_steps=6042 \
--warmup_steps=604 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pus_arab_10mb.txt \
--seed=43 \
--override_n_examples=4834 \
--output_dir=models/10mb/pus_arab_10mb
cp tokenizers/monolingual/pus_arab_10mb/* models/10mb/pus_arab_10mb

# que_latn
if test -f models/10mb/que_latn_10mb/pytorch_model.bin; then
echo "Model already found: que_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/que_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/que_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=365 --save_steps=999999999 \
--max_steps=7313 \
--warmup_steps=731 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/que_latn_10mb.txt \
--seed=43 \
--override_n_examples=5851 \
--output_dir=models/10mb/que_latn_10mb
cp tokenizers/monolingual/que_latn_10mb/* models/10mb/que_latn_10mb

# quy_latn
if test -f models/10mb/quy_latn_10mb/pytorch_model.bin; then
echo "Model already found: quy_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/quy_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/quy_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=355 --save_steps=999999999 \
--max_steps=7107 \
--warmup_steps=710 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/quy_latn_10mb.txt \
--seed=43 \
--override_n_examples=5686 \
--output_dir=models/10mb/quy_latn_10mb
cp tokenizers/monolingual/quy_latn_10mb/* models/10mb/quy_latn_10mb

# roh_latn
if test -f models/10mb/roh_latn_10mb/pytorch_model.bin; then
echo "Model already found: roh_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/roh_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/roh_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=341 --save_steps=999999999 \
--max_steps=6823 \
--warmup_steps=682 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/roh_latn_10mb.txt \
--seed=43 \
--override_n_examples=5459 \
--output_dir=models/10mb/roh_latn_10mb
cp tokenizers/monolingual/roh_latn_10mb/* models/10mb/roh_latn_10mb

# ron_latn
if test -f models/10mb/ron_latn_10mb/pytorch_model.bin; then
echo "Model already found: ron_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ron_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ron_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=291 --save_steps=999999999 \
--max_steps=5831 \
--warmup_steps=583 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ron_latn_10mb.txt \
--seed=43 \
--override_n_examples=4665 \
--output_dir=models/10mb/ron_latn_10mb
cp tokenizers/monolingual/ron_latn_10mb/* models/10mb/ron_latn_10mb

# run_latn
if test -f models/10mb/run_latn_10mb/pytorch_model.bin; then
echo "Model already found: run_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/run_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/run_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=331 --save_steps=999999999 \
--max_steps=6627 \
--warmup_steps=662 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/run_latn_10mb.txt \
--seed=43 \
--override_n_examples=5302 \
--output_dir=models/10mb/run_latn_10mb
cp tokenizers/monolingual/run_latn_10mb/* models/10mb/run_latn_10mb

# rus_cyrl
if test -f models/10mb/rus_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: rus_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rus_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rus_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=281 --save_steps=999999999 \
--max_steps=5627 \
--warmup_steps=562 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rus_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4502 \
--output_dir=models/10mb/rus_cyrl_10mb
cp tokenizers/monolingual/rus_cyrl_10mb/* models/10mb/rus_cyrl_10mb

# rus_latn
if test -f models/10mb/rus_latn_10mb/pytorch_model.bin; then
echo "Model already found: rus_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rus_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rus_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=345 --save_steps=999999999 \
--max_steps=6912 \
--warmup_steps=691 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rus_latn_10mb.txt \
--seed=43 \
--override_n_examples=5530 \
--output_dir=models/10mb/rus_latn_10mb
cp tokenizers/monolingual/rus_latn_10mb/* models/10mb/rus_latn_10mb

# sag_latn
if test -f models/10mb/sag_latn_10mb/pytorch_model.bin; then
echo "Model already found: sag_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sag_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sag_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=383 --save_steps=999999999 \
--max_steps=7661 \
--warmup_steps=766 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sag_latn_10mb.txt \
--seed=43 \
--override_n_examples=6129 \
--output_dir=models/10mb/sag_latn_10mb
cp tokenizers/monolingual/sag_latn_10mb/* models/10mb/sag_latn_10mb

# sah_cyrl
if test -f models/10mb/sah_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: sah_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sah_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sah_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=290 --save_steps=999999999 \
--max_steps=5817 \
--warmup_steps=581 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sah_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4654 \
--output_dir=models/10mb/sah_cyrl_10mb
cp tokenizers/monolingual/sah_cyrl_10mb/* models/10mb/sah_cyrl_10mb

# san_deva
if test -f models/10mb/san_deva_10mb/pytorch_model.bin; then
echo "Model already found: san_deva_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/san_deva_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/san_deva_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=335 --save_steps=999999999 \
--max_steps=6718 \
--warmup_steps=671 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/san_deva_10mb.txt \
--seed=43 \
--override_n_examples=5375 \
--output_dir=models/10mb/san_deva_10mb
cp tokenizers/monolingual/san_deva_10mb/* models/10mb/san_deva_10mb

# scn_latn
if test -f models/10mb/scn_latn_10mb/pytorch_model.bin; then
echo "Model already found: scn_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/scn_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/scn_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=352 --save_steps=999999999 \
--max_steps=7050 \
--warmup_steps=705 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/scn_latn_10mb.txt \
--seed=43 \
--override_n_examples=5640 \
--output_dir=models/10mb/scn_latn_10mb
cp tokenizers/monolingual/scn_latn_10mb/* models/10mb/scn_latn_10mb

# sco_latn
if test -f models/10mb/sco_latn_10mb/pytorch_model.bin; then
echo "Model already found: sco_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sco_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sco_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=366 --save_steps=999999999 \
--max_steps=7336 \
--warmup_steps=733 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sco_latn_10mb.txt \
--seed=43 \
--override_n_examples=5869 \
--output_dir=models/10mb/sco_latn_10mb
cp tokenizers/monolingual/sco_latn_10mb/* models/10mb/sco_latn_10mb

# shn_mymr
if test -f models/10mb/shn_mymr_10mb/pytorch_model.bin; then
echo "Model already found: shn_mymr_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/shn_mymr_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/shn_mymr_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=4722 \
--warmup_steps=472 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/shn_mymr_10mb.txt \
--seed=43 \
--override_n_examples=3778 \
--output_dir=models/10mb/shn_mymr_10mb
cp tokenizers/monolingual/shn_mymr_10mb/* models/10mb/shn_mymr_10mb

# sin_sinh
if test -f models/10mb/sin_sinh_10mb/pytorch_model.bin; then
echo "Model already found: sin_sinh_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sin_sinh_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sin_sinh_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=293 --save_steps=999999999 \
--max_steps=5867 \
--warmup_steps=586 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sin_sinh_10mb.txt \
--seed=43 \
--override_n_examples=4694 \
--output_dir=models/10mb/sin_sinh_10mb
cp tokenizers/monolingual/sin_sinh_10mb/* models/10mb/sin_sinh_10mb

# slk_latn
if test -f models/10mb/slk_latn_10mb/pytorch_model.bin; then
echo "Model already found: slk_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/slk_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/slk_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=268 --save_steps=999999999 \
--max_steps=5372 \
--warmup_steps=537 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/slk_latn_10mb.txt \
--seed=43 \
--override_n_examples=4298 \
--output_dir=models/10mb/slk_latn_10mb
cp tokenizers/monolingual/slk_latn_10mb/* models/10mb/slk_latn_10mb

# slv_latn
if test -f models/10mb/slv_latn_10mb/pytorch_model.bin; then
echo "Model already found: slv_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/slv_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/slv_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=251 --save_steps=999999999 \
--max_steps=5031 \
--warmup_steps=503 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/slv_latn_10mb.txt \
--seed=43 \
--override_n_examples=4025 \
--output_dir=models/10mb/slv_latn_10mb
cp tokenizers/monolingual/slv_latn_10mb/* models/10mb/slv_latn_10mb

# sme_latn
if test -f models/10mb/sme_latn_10mb/pytorch_model.bin; then
echo "Model already found: sme_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sme_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sme_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=299 --save_steps=999999999 \
--max_steps=5990 \
--warmup_steps=599 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sme_latn_10mb.txt \
--seed=43 \
--override_n_examples=4792 \
--output_dir=models/10mb/sme_latn_10mb
cp tokenizers/monolingual/sme_latn_10mb/* models/10mb/sme_latn_10mb

# smo_latn
if test -f models/10mb/smo_latn_10mb/pytorch_model.bin; then
echo "Model already found: smo_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/smo_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/smo_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=401 --save_steps=999999999 \
--max_steps=8037 \
--warmup_steps=803 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/smo_latn_10mb.txt \
--seed=43 \
--override_n_examples=6430 \
--output_dir=models/10mb/smo_latn_10mb
cp tokenizers/monolingual/smo_latn_10mb/* models/10mb/smo_latn_10mb

# sna_latn
if test -f models/10mb/sna_latn_10mb/pytorch_model.bin; then
echo "Model already found: sna_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sna_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sna_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=315 --save_steps=999999999 \
--max_steps=6313 \
--warmup_steps=631 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sna_latn_10mb.txt \
--seed=43 \
--override_n_examples=5051 \
--output_dir=models/10mb/sna_latn_10mb
cp tokenizers/monolingual/sna_latn_10mb/* models/10mb/sna_latn_10mb

# snd_arab
if test -f models/10mb/snd_arab_10mb/pytorch_model.bin; then
echo "Model already found: snd_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/snd_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/snd_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=314 --save_steps=999999999 \
--max_steps=6291 \
--warmup_steps=629 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/snd_arab_10mb.txt \
--seed=43 \
--override_n_examples=5033 \
--output_dir=models/10mb/snd_arab_10mb
cp tokenizers/monolingual/snd_arab_10mb/* models/10mb/snd_arab_10mb

# som_latn
if test -f models/10mb/som_latn_10mb/pytorch_model.bin; then
echo "Model already found: som_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/som_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/som_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=377 --save_steps=999999999 \
--max_steps=7557 \
--warmup_steps=755 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/som_latn_10mb.txt \
--seed=43 \
--override_n_examples=6046 \
--output_dir=models/10mb/som_latn_10mb
cp tokenizers/monolingual/som_latn_10mb/* models/10mb/som_latn_10mb

# sot_latn
if test -f models/10mb/sot_latn_10mb/pytorch_model.bin; then
echo "Model already found: sot_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sot_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sot_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=353 --save_steps=999999999 \
--max_steps=7060 \
--warmup_steps=706 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sot_latn_10mb.txt \
--seed=43 \
--override_n_examples=5648 \
--output_dir=models/10mb/sot_latn_10mb
cp tokenizers/monolingual/sot_latn_10mb/* models/10mb/sot_latn_10mb

# spa_latn
if test -f models/10mb/spa_latn_10mb/pytorch_model.bin; then
echo "Model already found: spa_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/spa_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/spa_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=277 --save_steps=999999999 \
--max_steps=5552 \
--warmup_steps=555 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/spa_latn_10mb.txt \
--seed=43 \
--override_n_examples=4442 \
--output_dir=models/10mb/spa_latn_10mb
cp tokenizers/monolingual/spa_latn_10mb/* models/10mb/spa_latn_10mb

# sqi_latn
if test -f models/10mb/sqi_latn_10mb/pytorch_model.bin; then
echo "Model already found: sqi_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sqi_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sqi_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=345 --save_steps=999999999 \
--max_steps=6906 \
--warmup_steps=690 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sqi_latn_10mb.txt \
--seed=43 \
--override_n_examples=5525 \
--output_dir=models/10mb/sqi_latn_10mb
cp tokenizers/monolingual/sqi_latn_10mb/* models/10mb/sqi_latn_10mb

# srd_latn
if test -f models/10mb/srd_latn_10mb/pytorch_model.bin; then
echo "Model already found: srd_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srd_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srd_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=343 --save_steps=999999999 \
--max_steps=6865 \
--warmup_steps=686 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srd_latn_10mb.txt \
--seed=43 \
--override_n_examples=5492 \
--output_dir=models/10mb/srd_latn_10mb
cp tokenizers/monolingual/srd_latn_10mb/* models/10mb/srd_latn_10mb

# srn_latn
if test -f models/10mb/srn_latn_10mb/pytorch_model.bin; then
echo "Model already found: srn_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srn_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srn_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6631 \
--warmup_steps=663 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srn_latn_10mb.txt \
--seed=43 \
--override_n_examples=5305 \
--output_dir=models/10mb/srn_latn_10mb
cp tokenizers/monolingual/srn_latn_10mb/* models/10mb/srn_latn_10mb

# srp_cyrl
if test -f models/10mb/srp_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: srp_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srp_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srp_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=235 --save_steps=999999999 \
--max_steps=4708 \
--warmup_steps=470 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srp_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=3767 \
--output_dir=models/10mb/srp_cyrl_10mb
cp tokenizers/monolingual/srp_cyrl_10mb/* models/10mb/srp_cyrl_10mb

# srp_latn
if test -f models/10mb/srp_latn_10mb/pytorch_model.bin; then
echo "Model already found: srp_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srp_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srp_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=259 --save_steps=999999999 \
--max_steps=5195 \
--warmup_steps=519 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srp_latn_10mb.txt \
--seed=43 \
--override_n_examples=4156 \
--output_dir=models/10mb/srp_latn_10mb
cp tokenizers/monolingual/srp_latn_10mb/* models/10mb/srp_latn_10mb

# ssw_latn
if test -f models/10mb/ssw_latn_10mb/pytorch_model.bin; then
echo "Model already found: ssw_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ssw_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ssw_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=329 --save_steps=999999999 \
--max_steps=6588 \
--warmup_steps=658 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ssw_latn_10mb.txt \
--seed=43 \
--override_n_examples=5271 \
--output_dir=models/10mb/ssw_latn_10mb
cp tokenizers/monolingual/ssw_latn_10mb/* models/10mb/ssw_latn_10mb

# sun_latn
if test -f models/10mb/sun_latn_10mb/pytorch_model.bin; then
echo "Model already found: sun_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sun_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sun_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=309 --save_steps=999999999 \
--max_steps=6183 \
--warmup_steps=618 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sun_latn_10mb.txt \
--seed=43 \
--override_n_examples=4947 \
--output_dir=models/10mb/sun_latn_10mb
cp tokenizers/monolingual/sun_latn_10mb/* models/10mb/sun_latn_10mb

# swa_latn
if test -f models/10mb/swa_latn_10mb/pytorch_model.bin; then
echo "Model already found: swa_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/swa_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/swa_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=329 --save_steps=999999999 \
--max_steps=6595 \
--warmup_steps=659 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/swa_latn_10mb.txt \
--seed=43 \
--override_n_examples=5276 \
--output_dir=models/10mb/swa_latn_10mb
cp tokenizers/monolingual/swa_latn_10mb/* models/10mb/swa_latn_10mb

# swe_latn
if test -f models/10mb/swe_latn_10mb/pytorch_model.bin; then
echo "Model already found: swe_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/swe_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/swe_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=260 --save_steps=999999999 \
--max_steps=5205 \
--warmup_steps=520 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/swe_latn_10mb.txt \
--seed=43 \
--override_n_examples=4164 \
--output_dir=models/10mb/swe_latn_10mb
cp tokenizers/monolingual/swe_latn_10mb/* models/10mb/swe_latn_10mb

# syr_syrc
if test -f models/10mb/syr_syrc_10mb/pytorch_model.bin; then
echo "Model already found: syr_syrc_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/syr_syrc_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/syr_syrc_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=5396 \
--warmup_steps=539 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/syr_syrc_10mb.txt \
--seed=43 \
--override_n_examples=4317 \
--output_dir=models/10mb/syr_syrc_10mb
cp tokenizers/monolingual/syr_syrc_10mb/* models/10mb/syr_syrc_10mb

# szl_latn
if test -f models/10mb/szl_latn_10mb/pytorch_model.bin; then
echo "Model already found: szl_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/szl_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/szl_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=352 --save_steps=999999999 \
--max_steps=7047 \
--warmup_steps=704 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/szl_latn_10mb.txt \
--seed=43 \
--override_n_examples=5638 \
--output_dir=models/10mb/szl_latn_10mb
cp tokenizers/monolingual/szl_latn_10mb/* models/10mb/szl_latn_10mb

# tam_taml
if test -f models/10mb/tam_taml_10mb/pytorch_model.bin; then
echo "Model already found: tam_taml_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tam_taml_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tam_taml_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=254 --save_steps=999999999 \
--max_steps=5080 \
--warmup_steps=508 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tam_taml_10mb.txt \
--seed=43 \
--override_n_examples=4064 \
--output_dir=models/10mb/tam_taml_10mb
cp tokenizers/monolingual/tam_taml_10mb/* models/10mb/tam_taml_10mb

# tat_cyrl
if test -f models/10mb/tat_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: tat_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tat_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tat_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=293 --save_steps=999999999 \
--max_steps=5860 \
--warmup_steps=586 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tat_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4688 \
--output_dir=models/10mb/tat_cyrl_10mb
cp tokenizers/monolingual/tat_cyrl_10mb/* models/10mb/tat_cyrl_10mb

# tel_latn
if test -f models/10mb/tel_latn_10mb/pytorch_model.bin; then
echo "Model already found: tel_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tel_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tel_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=326 --save_steps=999999999 \
--max_steps=6538 \
--warmup_steps=653 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tel_latn_10mb.txt \
--seed=43 \
--override_n_examples=5231 \
--output_dir=models/10mb/tel_latn_10mb
cp tokenizers/monolingual/tel_latn_10mb/* models/10mb/tel_latn_10mb

# tel_telu
if test -f models/10mb/tel_telu_10mb/pytorch_model.bin; then
echo "Model already found: tel_telu_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tel_telu_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tel_telu_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=264 --save_steps=999999999 \
--max_steps=5287 \
--warmup_steps=528 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tel_telu_10mb.txt \
--seed=43 \
--override_n_examples=4230 \
--output_dir=models/10mb/tel_telu_10mb
cp tokenizers/monolingual/tel_telu_10mb/* models/10mb/tel_telu_10mb

# tet_latn
if test -f models/10mb/tet_latn_10mb/pytorch_model.bin; then
echo "Model already found: tet_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tet_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tet_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=365 --save_steps=999999999 \
--max_steps=7301 \
--warmup_steps=730 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tet_latn_10mb.txt \
--seed=43 \
--override_n_examples=5841 \
--output_dir=models/10mb/tet_latn_10mb
cp tokenizers/monolingual/tet_latn_10mb/* models/10mb/tet_latn_10mb

# tgk_cyrl
if test -f models/10mb/tgk_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: tgk_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tgk_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tgk_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=279 --save_steps=999999999 \
--max_steps=5586 \
--warmup_steps=558 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tgk_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4469 \
--output_dir=models/10mb/tgk_cyrl_10mb
cp tokenizers/monolingual/tgk_cyrl_10mb/* models/10mb/tgk_cyrl_10mb

# tgl_latn
if test -f models/10mb/tgl_latn_10mb/pytorch_model.bin; then
echo "Model already found: tgl_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tgl_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tgl_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=308 --save_steps=999999999 \
--max_steps=6175 \
--warmup_steps=617 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tgl_latn_10mb.txt \
--seed=43 \
--override_n_examples=4940 \
--output_dir=models/10mb/tgl_latn_10mb
cp tokenizers/monolingual/tgl_latn_10mb/* models/10mb/tgl_latn_10mb

# tha_thai
if test -f models/10mb/tha_thai_10mb/pytorch_model.bin; then
echo "Model already found: tha_thai_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tha_thai_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tha_thai_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=257 --save_steps=999999999 \
--max_steps=5157 \
--warmup_steps=515 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tha_thai_10mb.txt \
--seed=43 \
--override_n_examples=4126 \
--output_dir=models/10mb/tha_thai_10mb
cp tokenizers/monolingual/tha_thai_10mb/* models/10mb/tha_thai_10mb

# tir_ethi
if test -f models/10mb/tir_ethi_10mb/pytorch_model.bin; then
echo "Model already found: tir_ethi_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tir_ethi_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tir_ethi_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=283 --save_steps=999999999 \
--max_steps=5666 \
--warmup_steps=566 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tir_ethi_10mb.txt \
--seed=43 \
--override_n_examples=4533 \
--output_dir=models/10mb/tir_ethi_10mb
cp tokenizers/monolingual/tir_ethi_10mb/* models/10mb/tir_ethi_10mb

# ton_latn
if test -f models/10mb/ton_latn_10mb/pytorch_model.bin; then
echo "Model already found: ton_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ton_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ton_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=439 --save_steps=999999999 \
--max_steps=8791 \
--warmup_steps=879 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ton_latn_10mb.txt \
--seed=43 \
--override_n_examples=7033 \
--output_dir=models/10mb/ton_latn_10mb
cp tokenizers/monolingual/ton_latn_10mb/* models/10mb/ton_latn_10mb

# tpi_latn
if test -f models/10mb/tpi_latn_10mb/pytorch_model.bin; then
echo "Model already found: tpi_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tpi_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tpi_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=377 --save_steps=999999999 \
--max_steps=7550 \
--warmup_steps=755 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tpi_latn_10mb.txt \
--seed=43 \
--override_n_examples=6040 \
--output_dir=models/10mb/tpi_latn_10mb
cp tokenizers/monolingual/tpi_latn_10mb/* models/10mb/tpi_latn_10mb

# tsn_latn
if test -f models/10mb/tsn_latn_10mb/pytorch_model.bin; then
echo "Model already found: tsn_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tsn_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tsn_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=386 --save_steps=999999999 \
--max_steps=7727 \
--warmup_steps=772 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tsn_latn_10mb.txt \
--seed=43 \
--override_n_examples=6182 \
--output_dir=models/10mb/tsn_latn_10mb
cp tokenizers/monolingual/tsn_latn_10mb/* models/10mb/tsn_latn_10mb

# tso_latn
if test -f models/10mb/tso_latn_10mb/pytorch_model.bin; then
echo "Model already found: tso_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tso_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tso_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=377 --save_steps=999999999 \
--max_steps=7545 \
--warmup_steps=754 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tso_latn_10mb.txt \
--seed=43 \
--override_n_examples=6036 \
--output_dir=models/10mb/tso_latn_10mb
cp tokenizers/monolingual/tso_latn_10mb/* models/10mb/tso_latn_10mb

# tuk_latn
if test -f models/10mb/tuk_latn_10mb/pytorch_model.bin; then
echo "Model already found: tuk_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tuk_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tuk_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=385 --save_steps=999999999 \
--max_steps=7703 \
--warmup_steps=770 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tuk_latn_10mb.txt \
--seed=43 \
--override_n_examples=6163 \
--output_dir=models/10mb/tuk_latn_10mb
cp tokenizers/monolingual/tuk_latn_10mb/* models/10mb/tuk_latn_10mb

# tum_latn
if test -f models/10mb/tum_latn_10mb/pytorch_model.bin; then
echo "Model already found: tum_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tum_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tum_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=335 --save_steps=999999999 \
--max_steps=6711 \
--warmup_steps=671 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tum_latn_10mb.txt \
--seed=43 \
--override_n_examples=5369 \
--output_dir=models/10mb/tum_latn_10mb
cp tokenizers/monolingual/tum_latn_10mb/* models/10mb/tum_latn_10mb

# tur_latn
if test -f models/10mb/tur_latn_10mb/pytorch_model.bin; then
echo "Model already found: tur_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tur_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tur_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=237 --save_steps=999999999 \
--max_steps=4740 \
--warmup_steps=474 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tur_latn_10mb.txt \
--seed=43 \
--override_n_examples=3792 \
--output_dir=models/10mb/tur_latn_10mb
cp tokenizers/monolingual/tur_latn_10mb/* models/10mb/tur_latn_10mb

# twi_latn
if test -f models/10mb/twi_latn_10mb/pytorch_model.bin; then
echo "Model already found: twi_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/twi_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/twi_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=376 --save_steps=999999999 \
--max_steps=7527 \
--warmup_steps=752 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/twi_latn_10mb.txt \
--seed=43 \
--override_n_examples=6022 \
--output_dir=models/10mb/twi_latn_10mb
cp tokenizers/monolingual/twi_latn_10mb/* models/10mb/twi_latn_10mb

# tyv_cyrl
if test -f models/10mb/tyv_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: tyv_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tyv_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tyv_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=286 --save_steps=999999999 \
--max_steps=5732 \
--warmup_steps=573 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tyv_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4586 \
--output_dir=models/10mb/tyv_cyrl_10mb
cp tokenizers/monolingual/tyv_cyrl_10mb/* models/10mb/tyv_cyrl_10mb

# udm_cyrl
if test -f models/10mb/udm_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: udm_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/udm_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/udm_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=264 --save_steps=999999999 \
--max_steps=5296 \
--warmup_steps=529 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/udm_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4237 \
--output_dir=models/10mb/udm_cyrl_10mb
cp tokenizers/monolingual/udm_cyrl_10mb/* models/10mb/udm_cyrl_10mb

# uig_arab
if test -f models/10mb/uig_arab_10mb/pytorch_model.bin; then
echo "Model already found: uig_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uig_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uig_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=331 --save_steps=999999999 \
--max_steps=6627 \
--warmup_steps=662 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uig_arab_10mb.txt \
--seed=43 \
--override_n_examples=5302 \
--output_dir=models/10mb/uig_arab_10mb
cp tokenizers/monolingual/uig_arab_10mb/* models/10mb/uig_arab_10mb

# ukr_cyrl
if test -f models/10mb/ukr_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: ukr_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ukr_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ukr_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=274 --save_steps=999999999 \
--max_steps=5486 \
--warmup_steps=548 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ukr_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4389 \
--output_dir=models/10mb/ukr_cyrl_10mb
cp tokenizers/monolingual/ukr_cyrl_10mb/* models/10mb/ukr_cyrl_10mb

# umb_latn
if test -f models/10mb/umb_latn_10mb/pytorch_model.bin; then
echo "Model already found: umb_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/umb_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/umb_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=363 --save_steps=999999999 \
--max_steps=7262 \
--warmup_steps=726 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/umb_latn_10mb.txt \
--seed=43 \
--override_n_examples=5810 \
--output_dir=models/10mb/umb_latn_10mb
cp tokenizers/monolingual/umb_latn_10mb/* models/10mb/umb_latn_10mb

# urd_arab
if test -f models/10mb/urd_arab_10mb/pytorch_model.bin; then
echo "Model already found: urd_arab_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/urd_arab_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/urd_arab_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=313 --save_steps=999999999 \
--max_steps=6260 \
--warmup_steps=626 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/urd_arab_10mb.txt \
--seed=43 \
--override_n_examples=5008 \
--output_dir=models/10mb/urd_arab_10mb
cp tokenizers/monolingual/urd_arab_10mb/* models/10mb/urd_arab_10mb

# uzb_cyrl
if test -f models/10mb/uzb_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: uzb_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzb_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzb_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=267 --save_steps=999999999 \
--max_steps=5340 \
--warmup_steps=534 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzb_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4272 \
--output_dir=models/10mb/uzb_cyrl_10mb
cp tokenizers/monolingual/uzb_cyrl_10mb/* models/10mb/uzb_cyrl_10mb

# uzb_latn
if test -f models/10mb/uzb_latn_10mb/pytorch_model.bin; then
echo "Model already found: uzb_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzb_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzb_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=328 --save_steps=999999999 \
--max_steps=6563 \
--warmup_steps=656 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzb_latn_10mb.txt \
--seed=43 \
--override_n_examples=5251 \
--output_dir=models/10mb/uzb_latn_10mb
cp tokenizers/monolingual/uzb_latn_10mb/* models/10mb/uzb_latn_10mb

# uzn_cyrl
if test -f models/10mb/uzn_cyrl_10mb/pytorch_model.bin; then
echo "Model already found: uzn_cyrl_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzn_cyrl_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzn_cyrl_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=271 --save_steps=999999999 \
--max_steps=5430 \
--warmup_steps=543 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzn_cyrl_10mb.txt \
--seed=43 \
--override_n_examples=4344 \
--output_dir=models/10mb/uzn_cyrl_10mb
cp tokenizers/monolingual/uzn_cyrl_10mb/* models/10mb/uzn_cyrl_10mb

# uzn_latn
if test -f models/10mb/uzn_latn_10mb/pytorch_model.bin; then
echo "Model already found: uzn_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzn_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzn_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=446 --save_steps=999999999 \
--max_steps=8926 \
--warmup_steps=892 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzn_latn_10mb.txt \
--seed=43 \
--override_n_examples=7141 \
--output_dir=models/10mb/uzn_latn_10mb
cp tokenizers/monolingual/uzn_latn_10mb/* models/10mb/uzn_latn_10mb

# vec_latn
if test -f models/10mb/vec_latn_10mb/pytorch_model.bin; then
echo "Model already found: vec_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vec_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vec_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=337 --save_steps=999999999 \
--max_steps=6741 \
--warmup_steps=674 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vec_latn_10mb.txt \
--seed=43 \
--override_n_examples=5393 \
--output_dir=models/10mb/vec_latn_10mb
cp tokenizers/monolingual/vec_latn_10mb/* models/10mb/vec_latn_10mb

# ven_latn
if test -f models/10mb/ven_latn_10mb/pytorch_model.bin; then
echo "Model already found: ven_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ven_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ven_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=6788 \
--warmup_steps=678 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ven_latn_10mb.txt \
--seed=43 \
--override_n_examples=5431 \
--output_dir=models/10mb/ven_latn_10mb
cp tokenizers/monolingual/ven_latn_10mb/* models/10mb/ven_latn_10mb

# vie_latn
if test -f models/10mb/vie_latn_10mb/pytorch_model.bin; then
echo "Model already found: vie_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vie_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vie_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=325 --save_steps=999999999 \
--max_steps=6512 \
--warmup_steps=651 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vie_latn_10mb.txt \
--seed=43 \
--override_n_examples=5210 \
--output_dir=models/10mb/vie_latn_10mb
cp tokenizers/monolingual/vie_latn_10mb/* models/10mb/vie_latn_10mb

# vol_latn
if test -f models/10mb/vol_latn_10mb/pytorch_model.bin; then
echo "Model already found: vol_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vol_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vol_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=351 --save_steps=999999999 \
--max_steps=7023 \
--warmup_steps=702 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vol_latn_10mb.txt \
--seed=43 \
--override_n_examples=5619 \
--output_dir=models/10mb/vol_latn_10mb
cp tokenizers/monolingual/vol_latn_10mb/* models/10mb/vol_latn_10mb

# war_latn
if test -f models/10mb/war_latn_10mb/pytorch_model.bin; then
echo "Model already found: war_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/war_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/war_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=356 --save_steps=999999999 \
--max_steps=7133 \
--warmup_steps=713 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/war_latn_10mb.txt \
--seed=43 \
--override_n_examples=5707 \
--output_dir=models/10mb/war_latn_10mb
cp tokenizers/monolingual/war_latn_10mb/* models/10mb/war_latn_10mb

# wln_latn
if test -f models/10mb/wln_latn_10mb/pytorch_model.bin; then
echo "Model already found: wln_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wln_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wln_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=388 --save_steps=999999999 \
--max_steps=7777 \
--warmup_steps=777 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wln_latn_10mb.txt \
--seed=43 \
--override_n_examples=6222 \
--output_dir=models/10mb/wln_latn_10mb
cp tokenizers/monolingual/wln_latn_10mb/* models/10mb/wln_latn_10mb

# wol_latn
if test -f models/10mb/wol_latn_10mb/pytorch_model.bin; then
echo "Model already found: wol_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wol_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wol_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=399 --save_steps=999999999 \
--max_steps=7997 \
--warmup_steps=799 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wol_latn_10mb.txt \
--seed=43 \
--override_n_examples=6398 \
--output_dir=models/10mb/wol_latn_10mb
cp tokenizers/monolingual/wol_latn_10mb/* models/10mb/wol_latn_10mb

# wuu_hani
if test -f models/10mb/wuu_hani_10mb/pytorch_model.bin; then
echo "Model already found: wuu_hani_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wuu_hani_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wuu_hani_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=209 --save_steps=999999999 \
--max_steps=4187 \
--warmup_steps=418 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wuu_hani_10mb.txt \
--seed=43 \
--override_n_examples=3350 \
--output_dir=models/10mb/wuu_hani_10mb
cp tokenizers/monolingual/wuu_hani_10mb/* models/10mb/wuu_hani_10mb

# xho_latn
if test -f models/10mb/xho_latn_10mb/pytorch_model.bin; then
echo "Model already found: xho_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/xho_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/xho_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=337 --save_steps=999999999 \
--max_steps=6752 \
--warmup_steps=675 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/xho_latn_10mb.txt \
--seed=43 \
--override_n_examples=5402 \
--output_dir=models/10mb/xho_latn_10mb
cp tokenizers/monolingual/xho_latn_10mb/* models/10mb/xho_latn_10mb

# ydd_hebr
if test -f models/10mb/ydd_hebr_10mb/pytorch_model.bin; then
echo "Model already found: ydd_hebr_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ydd_hebr_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ydd_hebr_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=293 --save_steps=999999999 \
--max_steps=5867 \
--warmup_steps=586 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ydd_hebr_10mb.txt \
--seed=43 \
--override_n_examples=4694 \
--output_dir=models/10mb/ydd_hebr_10mb
cp tokenizers/monolingual/ydd_hebr_10mb/* models/10mb/ydd_hebr_10mb

# yid_hebr
if test -f models/10mb/yid_hebr_10mb/pytorch_model.bin; then
echo "Model already found: yid_hebr_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yid_hebr_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yid_hebr_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=241 --save_steps=999999999 \
--max_steps=4822 \
--warmup_steps=482 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yid_hebr_10mb.txt \
--seed=43 \
--override_n_examples=3858 \
--output_dir=models/10mb/yid_hebr_10mb
cp tokenizers/monolingual/yid_hebr_10mb/* models/10mb/yid_hebr_10mb

# yor_latn
if test -f models/10mb/yor_latn_10mb/pytorch_model.bin; then
echo "Model already found: yor_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yor_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yor_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=388 --save_steps=999999999 \
--max_steps=7773 \
--warmup_steps=777 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yor_latn_10mb.txt \
--seed=43 \
--override_n_examples=6219 \
--output_dir=models/10mb/yor_latn_10mb
cp tokenizers/monolingual/yor_latn_10mb/* models/10mb/yor_latn_10mb

# yua_latn
if test -f models/10mb/yua_latn_10mb/pytorch_model.bin; then
echo "Model already found: yua_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yua_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yua_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=7828 \
--warmup_steps=782 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yua_latn_10mb.txt \
--seed=43 \
--override_n_examples=6263 \
--output_dir=models/10mb/yua_latn_10mb
cp tokenizers/monolingual/yua_latn_10mb/* models/10mb/yua_latn_10mb

# yue_hant
if test -f models/10mb/yue_hant_10mb/pytorch_model.bin; then
echo "Model already found: yue_hant_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yue_hant_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yue_hant_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=255 --save_steps=999999999 \
--max_steps=5102 \
--warmup_steps=510 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yue_hant_10mb.txt \
--seed=43 \
--override_n_examples=4082 \
--output_dir=models/10mb/yue_hant_10mb
cp tokenizers/monolingual/yue_hant_10mb/* models/10mb/yue_hant_10mb

# zho_hans
if test -f models/10mb/zho_hans_10mb/pytorch_model.bin; then
echo "Model already found: zho_hans_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zho_hans_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zho_hans_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=260 --save_steps=999999999 \
--max_steps=5217 \
--warmup_steps=521 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zho_hans_10mb.txt \
--seed=43 \
--override_n_examples=4174 \
--output_dir=models/10mb/zho_hans_10mb
cp tokenizers/monolingual/zho_hans_10mb/* models/10mb/zho_hans_10mb

# zho_hant
if test -f models/10mb/zho_hant_10mb/pytorch_model.bin; then
echo "Model already found: zho_hant_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zho_hant_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zho_hant_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=298 --save_steps=999999999 \
--max_steps=5971 \
--warmup_steps=597 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zho_hant_10mb.txt \
--seed=43 \
--override_n_examples=4777 \
--output_dir=models/10mb/zho_hant_10mb
cp tokenizers/monolingual/zho_hant_10mb/* models/10mb/zho_hant_10mb

# zsm_latn
if test -f models/10mb/zsm_latn_10mb/pytorch_model.bin; then
echo "Model already found: zsm_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zsm_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zsm_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=271 --save_steps=999999999 \
--max_steps=5433 \
--warmup_steps=543 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zsm_latn_10mb.txt \
--seed=43 \
--override_n_examples=4347 \
--output_dir=models/10mb/zsm_latn_10mb
cp tokenizers/monolingual/zsm_latn_10mb/* models/10mb/zsm_latn_10mb

# zul_latn
if test -f models/10mb/zul_latn_10mb/pytorch_model.bin; then
echo "Model already found: zul_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zul_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zul_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=327 --save_steps=999999999 \
--max_steps=6552 \
--warmup_steps=655 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zul_latn_10mb.txt \
--seed=43 \
--override_n_examples=5242 \
--output_dir=models/10mb/zul_latn_10mb
cp tokenizers/monolingual/zul_latn_10mb/* models/10mb/zul_latn_10mb

# zza_latn
if test -f models/10mb/zza_latn_10mb/pytorch_model.bin; then
echo "Model already found: zza_latn_10mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zza_latn_10mb \
--config_name="gpt_small_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zza_latn_10mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=1 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=398 --save_steps=999999999 \
--max_steps=7960 \
--warmup_steps=796 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zza_latn_10mb.txt \
--seed=43 \
--override_n_examples=6368 \
--output_dir=models/10mb/zza_latn_10mb
cp tokenizers/monolingual/zza_latn_10mb/* models/10mb/zza_latn_10mb
