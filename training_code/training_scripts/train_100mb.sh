export CUDA_VISIBLE_DEVICES=0

# afr_latn
if test -f models/100mb/afr_latn_100mb/pytorch_model.bin; then
echo "Model already found: afr_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/afr_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/afr_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=731 --save_steps=999999999 \
--max_steps=14629 \
--warmup_steps=1462 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/afr_latn_100mb.txt \
--seed=43 \
--override_n_examples=46814 \
--output_dir=models/100mb/afr_latn_100mb
cp tokenizers/monolingual/afr_latn_100mb/* models/100mb/afr_latn_100mb

# als_latn
if test -f models/100mb/als_latn_100mb/pytorch_model.bin; then
echo "Model already found: als_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/als_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/als_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=794 --save_steps=999999999 \
--max_steps=15895 \
--warmup_steps=1589 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/als_latn_100mb.txt \
--seed=43 \
--override_n_examples=50865 \
--output_dir=models/100mb/als_latn_100mb
cp tokenizers/monolingual/als_latn_100mb/* models/100mb/als_latn_100mb

# amh_ethi
if test -f models/100mb/amh_ethi_100mb/pytorch_model.bin; then
echo "Model already found: amh_ethi_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/amh_ethi_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/amh_ethi_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=646 --save_steps=999999999 \
--max_steps=12927 \
--warmup_steps=1292 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/amh_ethi_100mb.txt \
--seed=43 \
--override_n_examples=41367 \
--output_dir=models/100mb/amh_ethi_100mb
cp tokenizers/monolingual/amh_ethi_100mb/* models/100mb/amh_ethi_100mb

# arb_arab
if test -f models/100mb/arb_arab_100mb/pytorch_model.bin; then
echo "Model already found: arb_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arb_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arb_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=598 --save_steps=999999999 \
--max_steps=11967 \
--warmup_steps=1196 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arb_arab_100mb.txt \
--seed=43 \
--override_n_examples=38297 \
--output_dir=models/100mb/arb_arab_100mb
cp tokenizers/monolingual/arb_arab_100mb/* models/100mb/arb_arab_100mb

# arz_arab
if test -f models/100mb/arz_arab_100mb/pytorch_model.bin; then
echo "Model already found: arz_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arz_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arz_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=756 --save_steps=999999999 \
--max_steps=15120 \
--warmup_steps=1512 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arz_arab_100mb.txt \
--seed=43 \
--override_n_examples=48386 \
--output_dir=models/100mb/arz_arab_100mb
cp tokenizers/monolingual/arz_arab_100mb/* models/100mb/arz_arab_100mb

# asm_beng
if test -f models/100mb/asm_beng_100mb/pytorch_model.bin; then
echo "Model already found: asm_beng_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/asm_beng_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/asm_beng_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=674 --save_steps=999999999 \
--max_steps=13499 \
--warmup_steps=1349 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/asm_beng_100mb.txt \
--seed=43 \
--override_n_examples=43198 \
--output_dir=models/100mb/asm_beng_100mb
cp tokenizers/monolingual/asm_beng_100mb/* models/100mb/asm_beng_100mb

# ast_latn
if test -f models/100mb/ast_latn_100mb/pytorch_model.bin; then
echo "Model already found: ast_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ast_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ast_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=1261 --save_steps=999999999 \
--max_steps=25239 \
--warmup_steps=2523 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ast_latn_100mb.txt \
--seed=43 \
--override_n_examples=80767 \
--output_dir=models/100mb/ast_latn_100mb
cp tokenizers/monolingual/ast_latn_100mb/* models/100mb/ast_latn_100mb

# azb_arab
if test -f models/100mb/azb_arab_100mb/pytorch_model.bin; then
echo "Model already found: azb_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/azb_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/azb_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=763 --save_steps=999999999 \
--max_steps=15276 \
--warmup_steps=1527 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/azb_arab_100mb.txt \
--seed=43 \
--override_n_examples=48885 \
--output_dir=models/100mb/azb_arab_100mb
cp tokenizers/monolingual/azb_arab_100mb/* models/100mb/azb_arab_100mb

# aze_arab
if test -f models/100mb/aze_arab_100mb/pytorch_model.bin; then
echo "Model already found: aze_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=645 --save_steps=999999999 \
--max_steps=12906 \
--warmup_steps=1290 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_arab_100mb.txt \
--seed=43 \
--override_n_examples=41300 \
--output_dir=models/100mb/aze_arab_100mb
cp tokenizers/monolingual/aze_arab_100mb/* models/100mb/aze_arab_100mb

# aze_latn
if test -f models/100mb/aze_latn_100mb/pytorch_model.bin; then
echo "Model already found: aze_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=711 --save_steps=999999999 \
--max_steps=14226 \
--warmup_steps=1422 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_latn_100mb.txt \
--seed=43 \
--override_n_examples=45526 \
--output_dir=models/100mb/aze_latn_100mb
cp tokenizers/monolingual/aze_latn_100mb/* models/100mb/aze_latn_100mb

# azj_latn
if test -f models/100mb/azj_latn_100mb/pytorch_model.bin; then
echo "Model already found: azj_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/azj_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/azj_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=550 --save_steps=999999999 \
--max_steps=11016 \
--warmup_steps=1101 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/azj_latn_100mb.txt \
--seed=43 \
--override_n_examples=35252 \
--output_dir=models/100mb/azj_latn_100mb
cp tokenizers/monolingual/azj_latn_100mb/* models/100mb/azj_latn_100mb

# bak_cyrl
if test -f models/100mb/bak_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: bak_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bak_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bak_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=907 --save_steps=999999999 \
--max_steps=18140 \
--warmup_steps=1814 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bak_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=58049 \
--output_dir=models/100mb/bak_cyrl_100mb
cp tokenizers/monolingual/bak_cyrl_100mb/* models/100mb/bak_cyrl_100mb

# bel_cyrl
if test -f models/100mb/bel_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: bel_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bel_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bel_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=775 --save_steps=999999999 \
--max_steps=15509 \
--warmup_steps=1550 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bel_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=49629 \
--output_dir=models/100mb/bel_cyrl_100mb
cp tokenizers/monolingual/bel_cyrl_100mb/* models/100mb/bel_cyrl_100mb

# ben_beng
if test -f models/100mb/ben_beng_100mb/pytorch_model.bin; then
echo "Model already found: ben_beng_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ben_beng_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ben_beng_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=594 --save_steps=999999999 \
--max_steps=11884 \
--warmup_steps=1188 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ben_beng_100mb.txt \
--seed=43 \
--override_n_examples=38031 \
--output_dir=models/100mb/ben_beng_100mb
cp tokenizers/monolingual/ben_beng_100mb/* models/100mb/ben_beng_100mb

# bod_tibt
if test -f models/100mb/bod_tibt_100mb/pytorch_model.bin; then
echo "Model already found: bod_tibt_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bod_tibt_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bod_tibt_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=543 --save_steps=999999999 \
--max_steps=10860 \
--warmup_steps=1086 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bod_tibt_100mb.txt \
--seed=43 \
--override_n_examples=34753 \
--output_dir=models/100mb/bod_tibt_100mb
cp tokenizers/monolingual/bod_tibt_100mb/* models/100mb/bod_tibt_100mb

# bos_cyrl
if test -f models/100mb/bos_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: bos_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bos_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bos_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=709 --save_steps=999999999 \
--max_steps=14194 \
--warmup_steps=1419 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bos_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=45423 \
--output_dir=models/100mb/bos_cyrl_100mb
cp tokenizers/monolingual/bos_cyrl_100mb/* models/100mb/bos_cyrl_100mb

# bos_latn
if test -f models/100mb/bos_latn_100mb/pytorch_model.bin; then
echo "Model already found: bos_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bos_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bos_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=696 --save_steps=999999999 \
--max_steps=13932 \
--warmup_steps=1393 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bos_latn_100mb.txt \
--seed=43 \
--override_n_examples=44585 \
--output_dir=models/100mb/bos_latn_100mb
cp tokenizers/monolingual/bos_latn_100mb/* models/100mb/bos_latn_100mb

# bre_latn
if test -f models/100mb/bre_latn_100mb/pytorch_model.bin; then
echo "Model already found: bre_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bre_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bre_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=812 --save_steps=999999999 \
--max_steps=16251 \
--warmup_steps=1625 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bre_latn_100mb.txt \
--seed=43 \
--override_n_examples=52006 \
--output_dir=models/100mb/bre_latn_100mb
cp tokenizers/monolingual/bre_latn_100mb/* models/100mb/bre_latn_100mb

# bul_cyrl
if test -f models/100mb/bul_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: bul_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bul_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bul_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=684 --save_steps=999999999 \
--max_steps=13688 \
--warmup_steps=1368 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bul_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=43803 \
--output_dir=models/100mb/bul_cyrl_100mb
cp tokenizers/monolingual/bul_cyrl_100mb/* models/100mb/bul_cyrl_100mb

# cat_latn
if test -f models/100mb/cat_latn_100mb/pytorch_model.bin; then
echo "Model already found: cat_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cat_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cat_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=729 --save_steps=999999999 \
--max_steps=14586 \
--warmup_steps=1458 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cat_latn_100mb.txt \
--seed=43 \
--override_n_examples=46678 \
--output_dir=models/100mb/cat_latn_100mb
cp tokenizers/monolingual/cat_latn_100mb/* models/100mb/cat_latn_100mb

# ceb_latn
if test -f models/100mb/ceb_latn_100mb/pytorch_model.bin; then
echo "Model already found: ceb_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ceb_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ceb_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=792 --save_steps=999999999 \
--max_steps=15853 \
--warmup_steps=1585 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ceb_latn_100mb.txt \
--seed=43 \
--override_n_examples=50731 \
--output_dir=models/100mb/ceb_latn_100mb
cp tokenizers/monolingual/ceb_latn_100mb/* models/100mb/ceb_latn_100mb

# ces_latn
if test -f models/100mb/ces_latn_100mb/pytorch_model.bin; then
echo "Model already found: ces_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ces_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ces_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=629 --save_steps=999999999 \
--max_steps=12586 \
--warmup_steps=1258 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ces_latn_100mb.txt \
--seed=43 \
--override_n_examples=40276 \
--output_dir=models/100mb/ces_latn_100mb
cp tokenizers/monolingual/ces_latn_100mb/* models/100mb/ces_latn_100mb

# chv_cyrl
if test -f models/100mb/chv_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: chv_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/chv_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/chv_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=1003 --save_steps=999999999 \
--max_steps=20066 \
--warmup_steps=2006 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/chv_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=64213 \
--output_dir=models/100mb/chv_cyrl_100mb
cp tokenizers/monolingual/chv_cyrl_100mb/* models/100mb/chv_cyrl_100mb

# ckb_arab
if test -f models/100mb/ckb_arab_100mb/pytorch_model.bin; then
echo "Model already found: ckb_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ckb_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ckb_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=692 --save_steps=999999999 \
--max_steps=13848 \
--warmup_steps=1384 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ckb_arab_100mb.txt \
--seed=43 \
--override_n_examples=44314 \
--output_dir=models/100mb/ckb_arab_100mb
cp tokenizers/monolingual/ckb_arab_100mb/* models/100mb/ckb_arab_100mb

# cos_latn
if test -f models/100mb/cos_latn_100mb/pytorch_model.bin; then
echo "Model already found: cos_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cos_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cos_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=930 --save_steps=999999999 \
--max_steps=18603 \
--warmup_steps=1860 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cos_latn_100mb.txt \
--seed=43 \
--override_n_examples=59530 \
--output_dir=models/100mb/cos_latn_100mb
cp tokenizers/monolingual/cos_latn_100mb/* models/100mb/cos_latn_100mb

# cym_latn
if test -f models/100mb/cym_latn_100mb/pytorch_model.bin; then
echo "Model already found: cym_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cym_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cym_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=720 --save_steps=999999999 \
--max_steps=14417 \
--warmup_steps=1441 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cym_latn_100mb.txt \
--seed=43 \
--override_n_examples=46135 \
--output_dir=models/100mb/cym_latn_100mb
cp tokenizers/monolingual/cym_latn_100mb/* models/100mb/cym_latn_100mb

# dan_latn
if test -f models/100mb/dan_latn_100mb/pytorch_model.bin; then
echo "Model already found: dan_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dan_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dan_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=634 --save_steps=999999999 \
--max_steps=12696 \
--warmup_steps=1269 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dan_latn_100mb.txt \
--seed=43 \
--override_n_examples=40629 \
--output_dir=models/100mb/dan_latn_100mb
cp tokenizers/monolingual/dan_latn_100mb/* models/100mb/dan_latn_100mb

# deu_latn
if test -f models/100mb/deu_latn_100mb/pytorch_model.bin; then
echo "Model already found: deu_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/deu_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/deu_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=643 --save_steps=999999999 \
--max_steps=12861 \
--warmup_steps=1286 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/deu_latn_100mb.txt \
--seed=43 \
--override_n_examples=41158 \
--output_dir=models/100mb/deu_latn_100mb
cp tokenizers/monolingual/deu_latn_100mb/* models/100mb/deu_latn_100mb

# div_thaa
if test -f models/100mb/div_thaa_100mb/pytorch_model.bin; then
echo "Model already found: div_thaa_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/div_thaa_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/div_thaa_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=550 --save_steps=999999999 \
--max_steps=11008 \
--warmup_steps=1100 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/div_thaa_100mb.txt \
--seed=43 \
--override_n_examples=35227 \
--output_dir=models/100mb/div_thaa_100mb
cp tokenizers/monolingual/div_thaa_100mb/* models/100mb/div_thaa_100mb

# ell_grek
if test -f models/100mb/ell_grek_100mb/pytorch_model.bin; then
echo "Model already found: ell_grek_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ell_grek_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ell_grek_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=728 --save_steps=999999999 \
--max_steps=14572 \
--warmup_steps=1457 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ell_grek_100mb.txt \
--seed=43 \
--override_n_examples=46632 \
--output_dir=models/100mb/ell_grek_100mb
cp tokenizers/monolingual/ell_grek_100mb/* models/100mb/ell_grek_100mb

# ell_latn
if test -f models/100mb/ell_latn_100mb/pytorch_model.bin; then
echo "Model already found: ell_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ell_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ell_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=748 --save_steps=999999999 \
--max_steps=14960 \
--warmup_steps=1496 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ell_latn_100mb.txt \
--seed=43 \
--override_n_examples=47873 \
--output_dir=models/100mb/ell_latn_100mb
cp tokenizers/monolingual/ell_latn_100mb/* models/100mb/ell_latn_100mb

# eng_latn
if test -f models/100mb/eng_latn_100mb/pytorch_model.bin; then
echo "Model already found: eng_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/eng_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/eng_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=652 --save_steps=999999999 \
--max_steps=13055 \
--warmup_steps=1305 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/eng_latn_100mb.txt \
--seed=43 \
--override_n_examples=41779 \
--output_dir=models/100mb/eng_latn_100mb
cp tokenizers/monolingual/eng_latn_100mb/* models/100mb/eng_latn_100mb

# epo_latn
if test -f models/100mb/epo_latn_100mb/pytorch_model.bin; then
echo "Model already found: epo_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/epo_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/epo_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=706 --save_steps=999999999 \
--max_steps=14123 \
--warmup_steps=1412 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/epo_latn_100mb.txt \
--seed=43 \
--override_n_examples=45194 \
--output_dir=models/100mb/epo_latn_100mb
cp tokenizers/monolingual/epo_latn_100mb/* models/100mb/epo_latn_100mb

# est_latn
if test -f models/100mb/est_latn_100mb/pytorch_model.bin; then
echo "Model already found: est_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/est_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/est_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=578 --save_steps=999999999 \
--max_steps=11569 \
--warmup_steps=1156 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/est_latn_100mb.txt \
--seed=43 \
--override_n_examples=37022 \
--output_dir=models/100mb/est_latn_100mb
cp tokenizers/monolingual/est_latn_100mb/* models/100mb/est_latn_100mb

# eus_latn
if test -f models/100mb/eus_latn_100mb/pytorch_model.bin; then
echo "Model already found: eus_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/eus_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/eus_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=640 --save_steps=999999999 \
--max_steps=12806 \
--warmup_steps=1280 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/eus_latn_100mb.txt \
--seed=43 \
--override_n_examples=40982 \
--output_dir=models/100mb/eus_latn_100mb
cp tokenizers/monolingual/eus_latn_100mb/* models/100mb/eus_latn_100mb

# fao_latn
if test -f models/100mb/fao_latn_100mb/pytorch_model.bin; then
echo "Model already found: fao_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fao_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fao_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=736 --save_steps=999999999 \
--max_steps=14727 \
--warmup_steps=1472 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fao_latn_100mb.txt \
--seed=43 \
--override_n_examples=47127 \
--output_dir=models/100mb/fao_latn_100mb
cp tokenizers/monolingual/fao_latn_100mb/* models/100mb/fao_latn_100mb

# fas_arab
if test -f models/100mb/fas_arab_100mb/pytorch_model.bin; then
echo "Model already found: fas_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fas_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fas_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=745 --save_steps=999999999 \
--max_steps=14915 \
--warmup_steps=1491 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fas_arab_100mb.txt \
--seed=43 \
--override_n_examples=47731 \
--output_dir=models/100mb/fas_arab_100mb
cp tokenizers/monolingual/fas_arab_100mb/* models/100mb/fas_arab_100mb

# fil_latn
if test -f models/100mb/fil_latn_100mb/pytorch_model.bin; then
echo "Model already found: fil_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fil_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fil_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=839 --save_steps=999999999 \
--max_steps=16783 \
--warmup_steps=1678 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fil_latn_100mb.txt \
--seed=43 \
--override_n_examples=53706 \
--output_dir=models/100mb/fil_latn_100mb
cp tokenizers/monolingual/fil_latn_100mb/* models/100mb/fil_latn_100mb

# fin_latn
if test -f models/100mb/fin_latn_100mb/pytorch_model.bin; then
echo "Model already found: fin_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fin_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fin_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=567 --save_steps=999999999 \
--max_steps=11348 \
--warmup_steps=1134 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fin_latn_100mb.txt \
--seed=43 \
--override_n_examples=36314 \
--output_dir=models/100mb/fin_latn_100mb
cp tokenizers/monolingual/fin_latn_100mb/* models/100mb/fin_latn_100mb

# fra_latn
if test -f models/100mb/fra_latn_100mb/pytorch_model.bin; then
echo "Model already found: fra_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fra_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fra_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=767 --save_steps=999999999 \
--max_steps=15342 \
--warmup_steps=1534 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fra_latn_100mb.txt \
--seed=43 \
--override_n_examples=49095 \
--output_dir=models/100mb/fra_latn_100mb
cp tokenizers/monolingual/fra_latn_100mb/* models/100mb/fra_latn_100mb

# fry_latn
if test -f models/100mb/fry_latn_100mb/pytorch_model.bin; then
echo "Model already found: fry_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fry_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fry_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=859 --save_steps=999999999 \
--max_steps=17186 \
--warmup_steps=1718 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fry_latn_100mb.txt \
--seed=43 \
--override_n_examples=54998 \
--output_dir=models/100mb/fry_latn_100mb
cp tokenizers/monolingual/fry_latn_100mb/* models/100mb/fry_latn_100mb

# gla_latn
if test -f models/100mb/gla_latn_100mb/pytorch_model.bin; then
echo "Model already found: gla_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gla_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gla_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=675 --save_steps=999999999 \
--max_steps=13510 \
--warmup_steps=1351 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gla_latn_100mb.txt \
--seed=43 \
--override_n_examples=43235 \
--output_dir=models/100mb/gla_latn_100mb
cp tokenizers/monolingual/gla_latn_100mb/* models/100mb/gla_latn_100mb

# gle_latn
if test -f models/100mb/gle_latn_100mb/pytorch_model.bin; then
echo "Model already found: gle_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gle_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gle_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=1265 --save_steps=999999999 \
--max_steps=25300 \
--warmup_steps=2530 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gle_latn_100mb.txt \
--seed=43 \
--override_n_examples=80960 \
--output_dir=models/100mb/gle_latn_100mb
cp tokenizers/monolingual/gle_latn_100mb/* models/100mb/gle_latn_100mb

# glg_latn
if test -f models/100mb/glg_latn_100mb/pytorch_model.bin; then
echo "Model already found: glg_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/glg_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/glg_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=678 --save_steps=999999999 \
--max_steps=13560 \
--warmup_steps=1356 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/glg_latn_100mb.txt \
--seed=43 \
--override_n_examples=43392 \
--output_dir=models/100mb/glg_latn_100mb
cp tokenizers/monolingual/glg_latn_100mb/* models/100mb/glg_latn_100mb

# grc_grek
if test -f models/100mb/grc_grek_100mb/pytorch_model.bin; then
echo "Model already found: grc_grek_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/grc_grek_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/grc_grek_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=707 --save_steps=999999999 \
--max_steps=14154 \
--warmup_steps=1415 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/grc_grek_100mb.txt \
--seed=43 \
--override_n_examples=45294 \
--output_dir=models/100mb/grc_grek_100mb
cp tokenizers/monolingual/grc_grek_100mb/* models/100mb/grc_grek_100mb

# gsw_latn
if test -f models/100mb/gsw_latn_100mb/pytorch_model.bin; then
echo "Model already found: gsw_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/gsw_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/gsw_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=913 --save_steps=999999999 \
--max_steps=18279 \
--warmup_steps=1827 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/gsw_latn_100mb.txt \
--seed=43 \
--override_n_examples=58494 \
--output_dir=models/100mb/gsw_latn_100mb
cp tokenizers/monolingual/gsw_latn_100mb/* models/100mb/gsw_latn_100mb

# guj_gujr
if test -f models/100mb/guj_gujr_100mb/pytorch_model.bin; then
echo "Model already found: guj_gujr_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/guj_gujr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/guj_gujr_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=591 --save_steps=999999999 \
--max_steps=11830 \
--warmup_steps=1183 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/guj_gujr_100mb.txt \
--seed=43 \
--override_n_examples=37857 \
--output_dir=models/100mb/guj_gujr_100mb
cp tokenizers/monolingual/guj_gujr_100mb/* models/100mb/guj_gujr_100mb

# guj_latn
if test -f models/100mb/guj_latn_100mb/pytorch_model.bin; then
echo "Model already found: guj_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/guj_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/guj_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=14800 \
--warmup_steps=1480 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/guj_latn_100mb.txt \
--seed=43 \
--override_n_examples=47360 \
--output_dir=models/100mb/guj_latn_100mb
cp tokenizers/monolingual/guj_latn_100mb/* models/100mb/guj_latn_100mb

# hat_latn
if test -f models/100mb/hat_latn_100mb/pytorch_model.bin; then
echo "Model already found: hat_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hat_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hat_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=728 --save_steps=999999999 \
--max_steps=14577 \
--warmup_steps=1457 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hat_latn_100mb.txt \
--seed=43 \
--override_n_examples=46649 \
--output_dir=models/100mb/hat_latn_100mb
cp tokenizers/monolingual/hat_latn_100mb/* models/100mb/hat_latn_100mb

# hau_latn
if test -f models/100mb/hau_latn_100mb/pytorch_model.bin; then
echo "Model already found: hau_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hau_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hau_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=846 --save_steps=999999999 \
--max_steps=16934 \
--warmup_steps=1693 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hau_latn_100mb.txt \
--seed=43 \
--override_n_examples=54189 \
--output_dir=models/100mb/hau_latn_100mb
cp tokenizers/monolingual/hau_latn_100mb/* models/100mb/hau_latn_100mb

# haw_latn
if test -f models/100mb/haw_latn_100mb/pytorch_model.bin; then
echo "Model already found: haw_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/haw_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/haw_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=1014 --save_steps=999999999 \
--max_steps=20291 \
--warmup_steps=2029 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/haw_latn_100mb.txt \
--seed=43 \
--override_n_examples=64932 \
--output_dir=models/100mb/haw_latn_100mb
cp tokenizers/monolingual/haw_latn_100mb/* models/100mb/haw_latn_100mb

# heb_hebr
if test -f models/100mb/heb_hebr_100mb/pytorch_model.bin; then
echo "Model already found: heb_hebr_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/heb_hebr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/heb_hebr_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=588 --save_steps=999999999 \
--max_steps=11776 \
--warmup_steps=1177 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/heb_hebr_100mb.txt \
--seed=43 \
--override_n_examples=37685 \
--output_dir=models/100mb/heb_hebr_100mb
cp tokenizers/monolingual/heb_hebr_100mb/* models/100mb/heb_hebr_100mb

# hin_deva
if test -f models/100mb/hin_deva_100mb/pytorch_model.bin; then
echo "Model already found: hin_deva_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hin_deva_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hin_deva_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=696 --save_steps=999999999 \
--max_steps=13925 \
--warmup_steps=1392 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hin_deva_100mb.txt \
--seed=43 \
--override_n_examples=44561 \
--output_dir=models/100mb/hin_deva_100mb
cp tokenizers/monolingual/hin_deva_100mb/* models/100mb/hin_deva_100mb

# hin_latn
if test -f models/100mb/hin_latn_100mb/pytorch_model.bin; then
echo "Model already found: hin_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hin_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hin_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=872 --save_steps=999999999 \
--max_steps=17444 \
--warmup_steps=1744 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hin_latn_100mb.txt \
--seed=43 \
--override_n_examples=55821 \
--output_dir=models/100mb/hin_latn_100mb
cp tokenizers/monolingual/hin_latn_100mb/* models/100mb/hin_latn_100mb

# hmn_latn
if test -f models/100mb/hmn_latn_100mb/pytorch_model.bin; then
echo "Model already found: hmn_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hmn_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hmn_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=882 --save_steps=999999999 \
--max_steps=17641 \
--warmup_steps=1764 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hmn_latn_100mb.txt \
--seed=43 \
--override_n_examples=56452 \
--output_dir=models/100mb/hmn_latn_100mb
cp tokenizers/monolingual/hmn_latn_100mb/* models/100mb/hmn_latn_100mb

# hrv_latn
if test -f models/100mb/hrv_latn_100mb/pytorch_model.bin; then
echo "Model already found: hrv_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hrv_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hrv_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=669 --save_steps=999999999 \
--max_steps=13390 \
--warmup_steps=1339 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hrv_latn_100mb.txt \
--seed=43 \
--override_n_examples=42848 \
--output_dir=models/100mb/hrv_latn_100mb
cp tokenizers/monolingual/hrv_latn_100mb/* models/100mb/hrv_latn_100mb

# hun_latn
if test -f models/100mb/hun_latn_100mb/pytorch_model.bin; then
echo "Model already found: hun_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hun_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hun_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=583 --save_steps=999999999 \
--max_steps=11660 \
--warmup_steps=1166 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hun_latn_100mb.txt \
--seed=43 \
--override_n_examples=37315 \
--output_dir=models/100mb/hun_latn_100mb
cp tokenizers/monolingual/hun_latn_100mb/* models/100mb/hun_latn_100mb

# hye_armn
if test -f models/100mb/hye_armn_100mb/pytorch_model.bin; then
echo "Model already found: hye_armn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hye_armn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hye_armn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=622 --save_steps=999999999 \
--max_steps=12444 \
--warmup_steps=1244 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hye_armn_100mb.txt \
--seed=43 \
--override_n_examples=39821 \
--output_dir=models/100mb/hye_armn_100mb
cp tokenizers/monolingual/hye_armn_100mb/* models/100mb/hye_armn_100mb

# ibo_latn
if test -f models/100mb/ibo_latn_100mb/pytorch_model.bin; then
echo "Model already found: ibo_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ibo_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ibo_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=940 --save_steps=999999999 \
--max_steps=18816 \
--warmup_steps=1881 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ibo_latn_100mb.txt \
--seed=43 \
--override_n_examples=60214 \
--output_dir=models/100mb/ibo_latn_100mb
cp tokenizers/monolingual/ibo_latn_100mb/* models/100mb/ibo_latn_100mb

# ind_latn
if test -f models/100mb/ind_latn_100mb/pytorch_model.bin; then
echo "Model already found: ind_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ind_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ind_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=642 --save_steps=999999999 \
--max_steps=12841 \
--warmup_steps=1284 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ind_latn_100mb.txt \
--seed=43 \
--override_n_examples=41092 \
--output_dir=models/100mb/ind_latn_100mb
cp tokenizers/monolingual/ind_latn_100mb/* models/100mb/ind_latn_100mb

# isl_latn
if test -f models/100mb/isl_latn_100mb/pytorch_model.bin; then
echo "Model already found: isl_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/isl_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/isl_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=723 --save_steps=999999999 \
--max_steps=14460 \
--warmup_steps=1446 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/isl_latn_100mb.txt \
--seed=43 \
--override_n_examples=46272 \
--output_dir=models/100mb/isl_latn_100mb
cp tokenizers/monolingual/isl_latn_100mb/* models/100mb/isl_latn_100mb

# ita_latn
if test -f models/100mb/ita_latn_100mb/pytorch_model.bin; then
echo "Model already found: ita_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ita_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ita_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=659 --save_steps=999999999 \
--max_steps=13191 \
--warmup_steps=1319 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ita_latn_100mb.txt \
--seed=43 \
--override_n_examples=42213 \
--output_dir=models/100mb/ita_latn_100mb
cp tokenizers/monolingual/ita_latn_100mb/* models/100mb/ita_latn_100mb

# jav_latn
if test -f models/100mb/jav_latn_100mb/pytorch_model.bin; then
echo "Model already found: jav_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/jav_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/jav_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=756 --save_steps=999999999 \
--max_steps=15127 \
--warmup_steps=1512 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/jav_latn_100mb.txt \
--seed=43 \
--override_n_examples=48408 \
--output_dir=models/100mb/jav_latn_100mb
cp tokenizers/monolingual/jav_latn_100mb/* models/100mb/jav_latn_100mb

# jpn_jpan
if test -f models/100mb/jpn_jpan_100mb/pytorch_model.bin; then
echo "Model already found: jpn_jpan_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/jpn_jpan_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/jpn_jpan_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=669 --save_steps=999999999 \
--max_steps=13382 \
--warmup_steps=1338 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/jpn_jpan_100mb.txt \
--seed=43 \
--override_n_examples=42823 \
--output_dir=models/100mb/jpn_jpan_100mb
cp tokenizers/monolingual/jpn_jpan_100mb/* models/100mb/jpn_jpan_100mb

# kaa_cyrl
if test -f models/100mb/kaa_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: kaa_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaa_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaa_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=647 --save_steps=999999999 \
--max_steps=12950 \
--warmup_steps=1295 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaa_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=41442 \
--output_dir=models/100mb/kaa_cyrl_100mb
cp tokenizers/monolingual/kaa_cyrl_100mb/* models/100mb/kaa_cyrl_100mb

# kaa_latn
if test -f models/100mb/kaa_latn_100mb/pytorch_model.bin; then
echo "Model already found: kaa_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaa_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaa_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=716 --save_steps=999999999 \
--max_steps=14320 \
--warmup_steps=1432 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaa_latn_100mb.txt \
--seed=43 \
--override_n_examples=45826 \
--output_dir=models/100mb/kaa_latn_100mb
cp tokenizers/monolingual/kaa_latn_100mb/* models/100mb/kaa_latn_100mb

# kal_latn
if test -f models/100mb/kal_latn_100mb/pytorch_model.bin; then
echo "Model already found: kal_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kal_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kal_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=653 --save_steps=999999999 \
--max_steps=13066 \
--warmup_steps=1306 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kal_latn_100mb.txt \
--seed=43 \
--override_n_examples=41812 \
--output_dir=models/100mb/kal_latn_100mb
cp tokenizers/monolingual/kal_latn_100mb/* models/100mb/kal_latn_100mb

# kan_knda
if test -f models/100mb/kan_knda_100mb/pytorch_model.bin; then
echo "Model already found: kan_knda_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kan_knda_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kan_knda_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=648 --save_steps=999999999 \
--max_steps=12979 \
--warmup_steps=1297 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kan_knda_100mb.txt \
--seed=43 \
--override_n_examples=41533 \
--output_dir=models/100mb/kan_knda_100mb
cp tokenizers/monolingual/kan_knda_100mb/* models/100mb/kan_knda_100mb

# kat_geor
if test -f models/100mb/kat_geor_100mb/pytorch_model.bin; then
echo "Model already found: kat_geor_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kat_geor_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kat_geor_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=1083 --save_steps=999999999 \
--max_steps=21675 \
--warmup_steps=2167 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kat_geor_100mb.txt \
--seed=43 \
--override_n_examples=69360 \
--output_dir=models/100mb/kat_geor_100mb
cp tokenizers/monolingual/kat_geor_100mb/* models/100mb/kat_geor_100mb

# kaz_cyrl
if test -f models/100mb/kaz_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: kaz_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaz_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaz_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=610 --save_steps=999999999 \
--max_steps=12201 \
--warmup_steps=1220 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaz_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=39046 \
--output_dir=models/100mb/kaz_cyrl_100mb
cp tokenizers/monolingual/kaz_cyrl_100mb/* models/100mb/kaz_cyrl_100mb

# khk_cyrl
if test -f models/100mb/khk_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: khk_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/khk_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/khk_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=665 --save_steps=999999999 \
--max_steps=13309 \
--warmup_steps=1330 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/khk_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=42591 \
--output_dir=models/100mb/khk_cyrl_100mb
cp tokenizers/monolingual/khk_cyrl_100mb/* models/100mb/khk_cyrl_100mb

# khm_khmr
if test -f models/100mb/khm_khmr_100mb/pytorch_model.bin; then
echo "Model already found: khm_khmr_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/khm_khmr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/khm_khmr_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=968 --save_steps=999999999 \
--max_steps=19376 \
--warmup_steps=1937 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/khm_khmr_100mb.txt \
--seed=43 \
--override_n_examples=62004 \
--output_dir=models/100mb/khm_khmr_100mb
cp tokenizers/monolingual/khm_khmr_100mb/* models/100mb/khm_khmr_100mb

# kin_latn
if test -f models/100mb/kin_latn_100mb/pytorch_model.bin; then
echo "Model already found: kin_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kin_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kin_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=728 --save_steps=999999999 \
--max_steps=14571 \
--warmup_steps=1457 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kin_latn_100mb.txt \
--seed=43 \
--override_n_examples=46630 \
--output_dir=models/100mb/kin_latn_100mb
cp tokenizers/monolingual/kin_latn_100mb/* models/100mb/kin_latn_100mb

# kir_cyrl
if test -f models/100mb/kir_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: kir_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kir_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kir_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=681 --save_steps=999999999 \
--max_steps=13623 \
--warmup_steps=1362 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kir_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=43594 \
--output_dir=models/100mb/kir_cyrl_100mb
cp tokenizers/monolingual/kir_cyrl_100mb/* models/100mb/kir_cyrl_100mb

# knc_arab
if test -f models/100mb/knc_arab_100mb/pytorch_model.bin; then
echo "Model already found: knc_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/knc_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/knc_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=3269 --save_steps=999999999 \
--max_steps=65380 \
--warmup_steps=6538 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/knc_arab_100mb.txt \
--seed=43 \
--override_n_examples=209217 \
--output_dir=models/100mb/knc_arab_100mb
cp tokenizers/monolingual/knc_arab_100mb/* models/100mb/knc_arab_100mb

# kor_hang
if test -f models/100mb/kor_hang_100mb/pytorch_model.bin; then
echo "Model already found: kor_hang_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kor_hang_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kor_hang_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=692 --save_steps=999999999 \
--max_steps=13859 \
--warmup_steps=1385 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kor_hang_100mb.txt \
--seed=43 \
--override_n_examples=44351 \
--output_dir=models/100mb/kor_hang_100mb
cp tokenizers/monolingual/kor_hang_100mb/* models/100mb/kor_hang_100mb

# kur_arab
if test -f models/100mb/kur_arab_100mb/pytorch_model.bin; then
echo "Model already found: kur_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kur_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kur_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=665 --save_steps=999999999 \
--max_steps=13307 \
--warmup_steps=1330 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kur_arab_100mb.txt \
--seed=43 \
--override_n_examples=42584 \
--output_dir=models/100mb/kur_arab_100mb
cp tokenizers/monolingual/kur_arab_100mb/* models/100mb/kur_arab_100mb

# kur_latn
if test -f models/100mb/kur_latn_100mb/pytorch_model.bin; then
echo "Model already found: kur_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kur_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kur_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=844 --save_steps=999999999 \
--max_steps=16899 \
--warmup_steps=1689 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kur_latn_100mb.txt \
--seed=43 \
--override_n_examples=54077 \
--output_dir=models/100mb/kur_latn_100mb
cp tokenizers/monolingual/kur_latn_100mb/* models/100mb/kur_latn_100mb

# lao_laoo
if test -f models/100mb/lao_laoo_100mb/pytorch_model.bin; then
echo "Model already found: lao_laoo_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lao_laoo_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lao_laoo_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=710 --save_steps=999999999 \
--max_steps=14210 \
--warmup_steps=1421 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lao_laoo_100mb.txt \
--seed=43 \
--override_n_examples=45474 \
--output_dir=models/100mb/lao_laoo_100mb
cp tokenizers/monolingual/lao_laoo_100mb/* models/100mb/lao_laoo_100mb

# lat_latn
if test -f models/100mb/lat_latn_100mb/pytorch_model.bin; then
echo "Model already found: lat_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lat_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lat_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=576 --save_steps=999999999 \
--max_steps=11533 \
--warmup_steps=1153 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lat_latn_100mb.txt \
--seed=43 \
--override_n_examples=36907 \
--output_dir=models/100mb/lat_latn_100mb
cp tokenizers/monolingual/lat_latn_100mb/* models/100mb/lat_latn_100mb

# lav_latn
if test -f models/100mb/lav_latn_100mb/pytorch_model.bin; then
echo "Model already found: lav_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lav_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lav_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=743 --save_steps=999999999 \
--max_steps=14864 \
--warmup_steps=1486 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lav_latn_100mb.txt \
--seed=43 \
--override_n_examples=47566 \
--output_dir=models/100mb/lav_latn_100mb
cp tokenizers/monolingual/lav_latn_100mb/* models/100mb/lav_latn_100mb

# lim_latn
if test -f models/100mb/lim_latn_100mb/pytorch_model.bin; then
echo "Model already found: lim_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lim_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lim_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=851 --save_steps=999999999 \
--max_steps=17026 \
--warmup_steps=1702 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lim_latn_100mb.txt \
--seed=43 \
--override_n_examples=54485 \
--output_dir=models/100mb/lim_latn_100mb
cp tokenizers/monolingual/lim_latn_100mb/* models/100mb/lim_latn_100mb

# lit_latn
if test -f models/100mb/lit_latn_100mb/pytorch_model.bin; then
echo "Model already found: lit_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lit_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lit_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=613 --save_steps=999999999 \
--max_steps=12278 \
--warmup_steps=1227 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lit_latn_100mb.txt \
--seed=43 \
--override_n_examples=39290 \
--output_dir=models/100mb/lit_latn_100mb
cp tokenizers/monolingual/lit_latn_100mb/* models/100mb/lit_latn_100mb

# lmo_latn
if test -f models/100mb/lmo_latn_100mb/pytorch_model.bin; then
echo "Model already found: lmo_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lmo_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lmo_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=881 --save_steps=999999999 \
--max_steps=17636 \
--warmup_steps=1763 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lmo_latn_100mb.txt \
--seed=43 \
--override_n_examples=56436 \
--output_dir=models/100mb/lmo_latn_100mb
cp tokenizers/monolingual/lmo_latn_100mb/* models/100mb/lmo_latn_100mb

# ltz_latn
if test -f models/100mb/ltz_latn_100mb/pytorch_model.bin; then
echo "Model already found: ltz_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ltz_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ltz_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=844 --save_steps=999999999 \
--max_steps=16889 \
--warmup_steps=1688 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ltz_latn_100mb.txt \
--seed=43 \
--override_n_examples=54047 \
--output_dir=models/100mb/ltz_latn_100mb
cp tokenizers/monolingual/ltz_latn_100mb/* models/100mb/ltz_latn_100mb

# lug_latn
if test -f models/100mb/lug_latn_100mb/pytorch_model.bin; then
echo "Model already found: lug_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lug_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lug_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=863 --save_steps=999999999 \
--max_steps=17271 \
--warmup_steps=1727 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lug_latn_100mb.txt \
--seed=43 \
--override_n_examples=55270 \
--output_dir=models/100mb/lug_latn_100mb
cp tokenizers/monolingual/lug_latn_100mb/* models/100mb/lug_latn_100mb

# lus_latn
if test -f models/100mb/lus_latn_100mb/pytorch_model.bin; then
echo "Model already found: lus_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lus_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lus_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=898 --save_steps=999999999 \
--max_steps=17973 \
--warmup_steps=1797 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lus_latn_100mb.txt \
--seed=43 \
--override_n_examples=57516 \
--output_dir=models/100mb/lus_latn_100mb
cp tokenizers/monolingual/lus_latn_100mb/* models/100mb/lus_latn_100mb

# mal_mlym
if test -f models/100mb/mal_mlym_100mb/pytorch_model.bin; then
echo "Model already found: mal_mlym_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mal_mlym_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mal_mlym_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=747 --save_steps=999999999 \
--max_steps=14941 \
--warmup_steps=1494 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mal_mlym_100mb.txt \
--seed=43 \
--override_n_examples=47814 \
--output_dir=models/100mb/mal_mlym_100mb
cp tokenizers/monolingual/mal_mlym_100mb/* models/100mb/mal_mlym_100mb

# mar_deva
if test -f models/100mb/mar_deva_100mb/pytorch_model.bin; then
echo "Model already found: mar_deva_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mar_deva_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mar_deva_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=630 --save_steps=999999999 \
--max_steps=12618 \
--warmup_steps=1261 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mar_deva_100mb.txt \
--seed=43 \
--override_n_examples=40378 \
--output_dir=models/100mb/mar_deva_100mb
cp tokenizers/monolingual/mar_deva_100mb/* models/100mb/mar_deva_100mb

# mkd_cyrl
if test -f models/100mb/mkd_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: mkd_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mkd_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mkd_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=675 --save_steps=999999999 \
--max_steps=13507 \
--warmup_steps=1350 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mkd_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=43224 \
--output_dir=models/100mb/mkd_cyrl_100mb
cp tokenizers/monolingual/mkd_cyrl_100mb/* models/100mb/mkd_cyrl_100mb

# mlg_latn
if test -f models/100mb/mlg_latn_100mb/pytorch_model.bin; then
echo "Model already found: mlg_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mlg_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mlg_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=891 --save_steps=999999999 \
--max_steps=17827 \
--warmup_steps=1782 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mlg_latn_100mb.txt \
--seed=43 \
--override_n_examples=57048 \
--output_dir=models/100mb/mlg_latn_100mb
cp tokenizers/monolingual/mlg_latn_100mb/* models/100mb/mlg_latn_100mb

# mlt_latn
if test -f models/100mb/mlt_latn_100mb/pytorch_model.bin; then
echo "Model already found: mlt_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mlt_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mlt_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=864 --save_steps=999999999 \
--max_steps=17290 \
--warmup_steps=1729 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mlt_latn_100mb.txt \
--seed=43 \
--override_n_examples=55331 \
--output_dir=models/100mb/mlt_latn_100mb
cp tokenizers/monolingual/mlt_latn_100mb/* models/100mb/mlt_latn_100mb

# mon_cyrl
if test -f models/100mb/mon_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: mon_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mon_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mon_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=627 --save_steps=999999999 \
--max_steps=12541 \
--warmup_steps=1254 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mon_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=40132 \
--output_dir=models/100mb/mon_cyrl_100mb
cp tokenizers/monolingual/mon_cyrl_100mb/* models/100mb/mon_cyrl_100mb

# mri_latn
if test -f models/100mb/mri_latn_100mb/pytorch_model.bin; then
echo "Model already found: mri_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mri_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mri_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=922 --save_steps=999999999 \
--max_steps=18446 \
--warmup_steps=1844 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mri_latn_100mb.txt \
--seed=43 \
--override_n_examples=59029 \
--output_dir=models/100mb/mri_latn_100mb
cp tokenizers/monolingual/mri_latn_100mb/* models/100mb/mri_latn_100mb

# msa_latn
if test -f models/100mb/msa_latn_100mb/pytorch_model.bin; then
echo "Model already found: msa_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/msa_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/msa_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=721 --save_steps=999999999 \
--max_steps=14422 \
--warmup_steps=1442 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/msa_latn_100mb.txt \
--seed=43 \
--override_n_examples=46152 \
--output_dir=models/100mb/msa_latn_100mb
cp tokenizers/monolingual/msa_latn_100mb/* models/100mb/msa_latn_100mb

# mya_mymr
if test -f models/100mb/mya_mymr_100mb/pytorch_model.bin; then
echo "Model already found: mya_mymr_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mya_mymr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mya_mymr_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=1263 --save_steps=999999999 \
--max_steps=25270 \
--warmup_steps=2527 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mya_mymr_100mb.txt \
--seed=43 \
--override_n_examples=80866 \
--output_dir=models/100mb/mya_mymr_100mb
cp tokenizers/monolingual/mya_mymr_100mb/* models/100mb/mya_mymr_100mb

# nep_deva
if test -f models/100mb/nep_deva_100mb/pytorch_model.bin; then
echo "Model already found: nep_deva_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nep_deva_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nep_deva_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=657 --save_steps=999999999 \
--max_steps=13143 \
--warmup_steps=1314 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nep_deva_100mb.txt \
--seed=43 \
--override_n_examples=42060 \
--output_dir=models/100mb/nep_deva_100mb
cp tokenizers/monolingual/nep_deva_100mb/* models/100mb/nep_deva_100mb

# nld_latn
if test -f models/100mb/nld_latn_100mb/pytorch_model.bin; then
echo "Model already found: nld_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nld_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nld_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=662 --save_steps=999999999 \
--max_steps=13243 \
--warmup_steps=1324 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nld_latn_100mb.txt \
--seed=43 \
--override_n_examples=42380 \
--output_dir=models/100mb/nld_latn_100mb
cp tokenizers/monolingual/nld_latn_100mb/* models/100mb/nld_latn_100mb

# nno_latn
if test -f models/100mb/nno_latn_100mb/pytorch_model.bin; then
echo "Model already found: nno_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nno_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nno_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=709 --save_steps=999999999 \
--max_steps=14195 \
--warmup_steps=1419 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nno_latn_100mb.txt \
--seed=43 \
--override_n_examples=45424 \
--output_dir=models/100mb/nno_latn_100mb
cp tokenizers/monolingual/nno_latn_100mb/* models/100mb/nno_latn_100mb

# nob_latn
if test -f models/100mb/nob_latn_100mb/pytorch_model.bin; then
echo "Model already found: nob_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nob_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nob_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=628 --save_steps=999999999 \
--max_steps=12566 \
--warmup_steps=1256 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nob_latn_100mb.txt \
--seed=43 \
--override_n_examples=40214 \
--output_dir=models/100mb/nob_latn_100mb
cp tokenizers/monolingual/nob_latn_100mb/* models/100mb/nob_latn_100mb

# nor_latn
if test -f models/100mb/nor_latn_100mb/pytorch_model.bin; then
echo "Model already found: nor_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nor_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nor_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=779 --save_steps=999999999 \
--max_steps=15592 \
--warmup_steps=1559 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nor_latn_100mb.txt \
--seed=43 \
--override_n_examples=49897 \
--output_dir=models/100mb/nor_latn_100mb
cp tokenizers/monolingual/nor_latn_100mb/* models/100mb/nor_latn_100mb

# nya_latn
if test -f models/100mb/nya_latn_100mb/pytorch_model.bin; then
echo "Model already found: nya_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nya_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nya_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=772 --save_steps=999999999 \
--max_steps=15455 \
--warmup_steps=1545 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nya_latn_100mb.txt \
--seed=43 \
--override_n_examples=49457 \
--output_dir=models/100mb/nya_latn_100mb
cp tokenizers/monolingual/nya_latn_100mb/* models/100mb/nya_latn_100mb

# oci_latn
if test -f models/100mb/oci_latn_100mb/pytorch_model.bin; then
echo "Model already found: oci_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/oci_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/oci_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=811 --save_steps=999999999 \
--max_steps=16225 \
--warmup_steps=1622 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/oci_latn_100mb.txt \
--seed=43 \
--override_n_examples=51923 \
--output_dir=models/100mb/oci_latn_100mb
cp tokenizers/monolingual/oci_latn_100mb/* models/100mb/oci_latn_100mb

# ori_orya
if test -f models/100mb/ori_orya_100mb/pytorch_model.bin; then
echo "Model already found: ori_orya_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ori_orya_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ori_orya_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=651 --save_steps=999999999 \
--max_steps=13038 \
--warmup_steps=1303 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ori_orya_100mb.txt \
--seed=43 \
--override_n_examples=41722 \
--output_dir=models/100mb/ori_orya_100mb
cp tokenizers/monolingual/ori_orya_100mb/* models/100mb/ori_orya_100mb

# orm_latn
if test -f models/100mb/orm_latn_100mb/pytorch_model.bin; then
echo "Model already found: orm_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/orm_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/orm_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=879 --save_steps=999999999 \
--max_steps=17597 \
--warmup_steps=1759 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/orm_latn_100mb.txt \
--seed=43 \
--override_n_examples=56313 \
--output_dir=models/100mb/orm_latn_100mb
cp tokenizers/monolingual/orm_latn_100mb/* models/100mb/orm_latn_100mb

# pan_guru
if test -f models/100mb/pan_guru_100mb/pytorch_model.bin; then
echo "Model already found: pan_guru_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pan_guru_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pan_guru_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=658 --save_steps=999999999 \
--max_steps=13168 \
--warmup_steps=1316 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pan_guru_100mb.txt \
--seed=43 \
--override_n_examples=42140 \
--output_dir=models/100mb/pan_guru_100mb
cp tokenizers/monolingual/pan_guru_100mb/* models/100mb/pan_guru_100mb

# pap_latn
if test -f models/100mb/pap_latn_100mb/pytorch_model.bin; then
echo "Model already found: pap_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pap_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pap_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=717 --save_steps=999999999 \
--max_steps=14342 \
--warmup_steps=1434 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pap_latn_100mb.txt \
--seed=43 \
--override_n_examples=45895 \
--output_dir=models/100mb/pap_latn_100mb
cp tokenizers/monolingual/pap_latn_100mb/* models/100mb/pap_latn_100mb

# pbt_arab
if test -f models/100mb/pbt_arab_100mb/pytorch_model.bin; then
echo "Model already found: pbt_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pbt_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pbt_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=821 --save_steps=999999999 \
--max_steps=16434 \
--warmup_steps=1643 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pbt_arab_100mb.txt \
--seed=43 \
--override_n_examples=52589 \
--output_dir=models/100mb/pbt_arab_100mb
cp tokenizers/monolingual/pbt_arab_100mb/* models/100mb/pbt_arab_100mb

# pes_arab
if test -f models/100mb/pes_arab_100mb/pytorch_model.bin; then
echo "Model already found: pes_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pes_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pes_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=659 --save_steps=999999999 \
--max_steps=13180 \
--warmup_steps=1318 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pes_arab_100mb.txt \
--seed=43 \
--override_n_examples=42179 \
--output_dir=models/100mb/pes_arab_100mb
cp tokenizers/monolingual/pes_arab_100mb/* models/100mb/pes_arab_100mb

# plt_latn
if test -f models/100mb/plt_latn_100mb/pytorch_model.bin; then
echo "Model already found: plt_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/plt_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/plt_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=803 --save_steps=999999999 \
--max_steps=16066 \
--warmup_steps=1606 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/plt_latn_100mb.txt \
--seed=43 \
--override_n_examples=51412 \
--output_dir=models/100mb/plt_latn_100mb
cp tokenizers/monolingual/plt_latn_100mb/* models/100mb/plt_latn_100mb

# pnb_arab
if test -f models/100mb/pnb_arab_100mb/pytorch_model.bin; then
echo "Model already found: pnb_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pnb_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pnb_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=755 --save_steps=999999999 \
--max_steps=15113 \
--warmup_steps=1511 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pnb_arab_100mb.txt \
--seed=43 \
--override_n_examples=48362 \
--output_dir=models/100mb/pnb_arab_100mb
cp tokenizers/monolingual/pnb_arab_100mb/* models/100mb/pnb_arab_100mb

# pol_latn
if test -f models/100mb/pol_latn_100mb/pytorch_model.bin; then
echo "Model already found: pol_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pol_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pol_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=659 --save_steps=999999999 \
--max_steps=13197 \
--warmup_steps=1319 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pol_latn_100mb.txt \
--seed=43 \
--override_n_examples=42231 \
--output_dir=models/100mb/pol_latn_100mb
cp tokenizers/monolingual/pol_latn_100mb/* models/100mb/pol_latn_100mb

# por_latn
if test -f models/100mb/por_latn_100mb/pytorch_model.bin; then
echo "Model already found: por_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/por_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/por_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=687 --save_steps=999999999 \
--max_steps=13750 \
--warmup_steps=1375 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/por_latn_100mb.txt \
--seed=43 \
--override_n_examples=44002 \
--output_dir=models/100mb/por_latn_100mb
cp tokenizers/monolingual/por_latn_100mb/* models/100mb/por_latn_100mb

# prs_arab
if test -f models/100mb/prs_arab_100mb/pytorch_model.bin; then
echo "Model already found: prs_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/prs_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/prs_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=704 --save_steps=999999999 \
--max_steps=14085 \
--warmup_steps=1408 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/prs_arab_100mb.txt \
--seed=43 \
--override_n_examples=45075 \
--output_dir=models/100mb/prs_arab_100mb
cp tokenizers/monolingual/prs_arab_100mb/* models/100mb/prs_arab_100mb

# pus_arab
if test -f models/100mb/pus_arab_100mb/pytorch_model.bin; then
echo "Model already found: pus_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pus_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pus_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=725 --save_steps=999999999 \
--max_steps=14512 \
--warmup_steps=1451 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pus_arab_100mb.txt \
--seed=43 \
--override_n_examples=46441 \
--output_dir=models/100mb/pus_arab_100mb
cp tokenizers/monolingual/pus_arab_100mb/* models/100mb/pus_arab_100mb

# que_latn
if test -f models/100mb/que_latn_100mb/pytorch_model.bin; then
echo "Model already found: que_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/que_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/que_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=888 --save_steps=999999999 \
--max_steps=17776 \
--warmup_steps=1777 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/que_latn_100mb.txt \
--seed=43 \
--override_n_examples=56886 \
--output_dir=models/100mb/que_latn_100mb
cp tokenizers/monolingual/que_latn_100mb/* models/100mb/que_latn_100mb

# quy_latn
if test -f models/100mb/quy_latn_100mb/pytorch_model.bin; then
echo "Model already found: quy_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/quy_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/quy_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=860 --save_steps=999999999 \
--max_steps=17215 \
--warmup_steps=1721 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/quy_latn_100mb.txt \
--seed=43 \
--override_n_examples=55088 \
--output_dir=models/100mb/quy_latn_100mb
cp tokenizers/monolingual/quy_latn_100mb/* models/100mb/quy_latn_100mb

# ron_latn
if test -f models/100mb/ron_latn_100mb/pytorch_model.bin; then
echo "Model already found: ron_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ron_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ron_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=703 --save_steps=999999999 \
--max_steps=14072 \
--warmup_steps=1407 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ron_latn_100mb.txt \
--seed=43 \
--override_n_examples=45033 \
--output_dir=models/100mb/ron_latn_100mb
cp tokenizers/monolingual/ron_latn_100mb/* models/100mb/ron_latn_100mb

# rus_cyrl
if test -f models/100mb/rus_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: rus_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rus_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rus_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=672 --save_steps=999999999 \
--max_steps=13454 \
--warmup_steps=1345 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rus_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=43055 \
--output_dir=models/100mb/rus_cyrl_100mb
cp tokenizers/monolingual/rus_cyrl_100mb/* models/100mb/rus_cyrl_100mb

# sah_cyrl
if test -f models/100mb/sah_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: sah_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sah_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sah_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=700 --save_steps=999999999 \
--max_steps=14000 \
--warmup_steps=1400 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sah_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=44802 \
--output_dir=models/100mb/sah_cyrl_100mb
cp tokenizers/monolingual/sah_cyrl_100mb/* models/100mb/sah_cyrl_100mb

# san_deva
if test -f models/100mb/san_deva_100mb/pytorch_model.bin; then
echo "Model already found: san_deva_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/san_deva_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/san_deva_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=814 --save_steps=999999999 \
--max_steps=16291 \
--warmup_steps=1629 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/san_deva_100mb.txt \
--seed=43 \
--override_n_examples=52134 \
--output_dir=models/100mb/san_deva_100mb
cp tokenizers/monolingual/san_deva_100mb/* models/100mb/san_deva_100mb

# scn_latn
if test -f models/100mb/scn_latn_100mb/pytorch_model.bin; then
echo "Model already found: scn_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/scn_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/scn_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=858 --save_steps=999999999 \
--max_steps=17168 \
--warmup_steps=1716 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/scn_latn_100mb.txt \
--seed=43 \
--override_n_examples=54939 \
--output_dir=models/100mb/scn_latn_100mb
cp tokenizers/monolingual/scn_latn_100mb/* models/100mb/scn_latn_100mb

# sin_sinh
if test -f models/100mb/sin_sinh_100mb/pytorch_model.bin; then
echo "Model already found: sin_sinh_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sin_sinh_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sin_sinh_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=710 --save_steps=999999999 \
--max_steps=14205 \
--warmup_steps=1420 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sin_sinh_100mb.txt \
--seed=43 \
--override_n_examples=45459 \
--output_dir=models/100mb/sin_sinh_100mb
cp tokenizers/monolingual/sin_sinh_100mb/* models/100mb/sin_sinh_100mb

# slk_latn
if test -f models/100mb/slk_latn_100mb/pytorch_model.bin; then
echo "Model already found: slk_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/slk_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/slk_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=644 --save_steps=999999999 \
--max_steps=12892 \
--warmup_steps=1289 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/slk_latn_100mb.txt \
--seed=43 \
--override_n_examples=41257 \
--output_dir=models/100mb/slk_latn_100mb
cp tokenizers/monolingual/slk_latn_100mb/* models/100mb/slk_latn_100mb

# slv_latn
if test -f models/100mb/slv_latn_100mb/pytorch_model.bin; then
echo "Model already found: slv_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/slv_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/slv_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=604 --save_steps=999999999 \
--max_steps=12084 \
--warmup_steps=1208 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/slv_latn_100mb.txt \
--seed=43 \
--override_n_examples=38669 \
--output_dir=models/100mb/slv_latn_100mb
cp tokenizers/monolingual/slv_latn_100mb/* models/100mb/slv_latn_100mb

# smo_latn
if test -f models/100mb/smo_latn_100mb/pytorch_model.bin; then
echo "Model already found: smo_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/smo_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/smo_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=987 --save_steps=999999999 \
--max_steps=19752 \
--warmup_steps=1975 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/smo_latn_100mb.txt \
--seed=43 \
--override_n_examples=63207 \
--output_dir=models/100mb/smo_latn_100mb
cp tokenizers/monolingual/smo_latn_100mb/* models/100mb/smo_latn_100mb

# sna_latn
if test -f models/100mb/sna_latn_100mb/pytorch_model.bin; then
echo "Model already found: sna_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sna_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sna_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=760 --save_steps=999999999 \
--max_steps=15219 \
--warmup_steps=1521 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sna_latn_100mb.txt \
--seed=43 \
--override_n_examples=48703 \
--output_dir=models/100mb/sna_latn_100mb
cp tokenizers/monolingual/sna_latn_100mb/* models/100mb/sna_latn_100mb

# snd_arab
if test -f models/100mb/snd_arab_100mb/pytorch_model.bin; then
echo "Model already found: snd_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/snd_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/snd_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=761 --save_steps=999999999 \
--max_steps=15235 \
--warmup_steps=1523 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/snd_arab_100mb.txt \
--seed=43 \
--override_n_examples=48753 \
--output_dir=models/100mb/snd_arab_100mb
cp tokenizers/monolingual/snd_arab_100mb/* models/100mb/snd_arab_100mb

# som_latn
if test -f models/100mb/som_latn_100mb/pytorch_model.bin; then
echo "Model already found: som_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/som_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/som_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=923 --save_steps=999999999 \
--max_steps=18468 \
--warmup_steps=1846 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/som_latn_100mb.txt \
--seed=43 \
--override_n_examples=59099 \
--output_dir=models/100mb/som_latn_100mb
cp tokenizers/monolingual/som_latn_100mb/* models/100mb/som_latn_100mb

# sot_latn
if test -f models/100mb/sot_latn_100mb/pytorch_model.bin; then
echo "Model already found: sot_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sot_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sot_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=862 --save_steps=999999999 \
--max_steps=17256 \
--warmup_steps=1725 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sot_latn_100mb.txt \
--seed=43 \
--override_n_examples=55221 \
--output_dir=models/100mb/sot_latn_100mb
cp tokenizers/monolingual/sot_latn_100mb/* models/100mb/sot_latn_100mb

# spa_latn
if test -f models/100mb/spa_latn_100mb/pytorch_model.bin; then
echo "Model already found: spa_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/spa_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/spa_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=676 --save_steps=999999999 \
--max_steps=13531 \
--warmup_steps=1353 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/spa_latn_100mb.txt \
--seed=43 \
--override_n_examples=43301 \
--output_dir=models/100mb/spa_latn_100mb
cp tokenizers/monolingual/spa_latn_100mb/* models/100mb/spa_latn_100mb

# sqi_latn
if test -f models/100mb/sqi_latn_100mb/pytorch_model.bin; then
echo "Model already found: sqi_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sqi_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sqi_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=838 --save_steps=999999999 \
--max_steps=16763 \
--warmup_steps=1676 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sqi_latn_100mb.txt \
--seed=43 \
--override_n_examples=53644 \
--output_dir=models/100mb/sqi_latn_100mb
cp tokenizers/monolingual/sqi_latn_100mb/* models/100mb/sqi_latn_100mb

# srp_cyrl
if test -f models/100mb/srp_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: srp_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srp_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srp_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=562 --save_steps=999999999 \
--max_steps=11249 \
--warmup_steps=1124 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srp_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=35998 \
--output_dir=models/100mb/srp_cyrl_100mb
cp tokenizers/monolingual/srp_cyrl_100mb/* models/100mb/srp_cyrl_100mb

# srp_latn
if test -f models/100mb/srp_latn_100mb/pytorch_model.bin; then
echo "Model already found: srp_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srp_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srp_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=633 --save_steps=999999999 \
--max_steps=12662 \
--warmup_steps=1266 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srp_latn_100mb.txt \
--seed=43 \
--override_n_examples=40521 \
--output_dir=models/100mb/srp_latn_100mb
cp tokenizers/monolingual/srp_latn_100mb/* models/100mb/srp_latn_100mb

# sun_latn
if test -f models/100mb/sun_latn_100mb/pytorch_model.bin; then
echo "Model already found: sun_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sun_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sun_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=751 --save_steps=999999999 \
--max_steps=15034 \
--warmup_steps=1503 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sun_latn_100mb.txt \
--seed=43 \
--override_n_examples=48110 \
--output_dir=models/100mb/sun_latn_100mb
cp tokenizers/monolingual/sun_latn_100mb/* models/100mb/sun_latn_100mb

# swa_latn
if test -f models/100mb/swa_latn_100mb/pytorch_model.bin; then
echo "Model already found: swa_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/swa_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/swa_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=793 --save_steps=999999999 \
--max_steps=15874 \
--warmup_steps=1587 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/swa_latn_100mb.txt \
--seed=43 \
--override_n_examples=50799 \
--output_dir=models/100mb/swa_latn_100mb
cp tokenizers/monolingual/swa_latn_100mb/* models/100mb/swa_latn_100mb

# swe_latn
if test -f models/100mb/swe_latn_100mb/pytorch_model.bin; then
echo "Model already found: swe_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/swe_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/swe_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=630 --save_steps=999999999 \
--max_steps=12601 \
--warmup_steps=1260 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/swe_latn_100mb.txt \
--seed=43 \
--override_n_examples=40325 \
--output_dir=models/100mb/swe_latn_100mb
cp tokenizers/monolingual/swe_latn_100mb/* models/100mb/swe_latn_100mb

# tam_taml
if test -f models/100mb/tam_taml_100mb/pytorch_model.bin; then
echo "Model already found: tam_taml_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tam_taml_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tam_taml_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=612 --save_steps=999999999 \
--max_steps=12246 \
--warmup_steps=1224 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tam_taml_100mb.txt \
--seed=43 \
--override_n_examples=39188 \
--output_dir=models/100mb/tam_taml_100mb
cp tokenizers/monolingual/tam_taml_100mb/* models/100mb/tam_taml_100mb

# tat_cyrl
if test -f models/100mb/tat_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: tat_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tat_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tat_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=711 --save_steps=999999999 \
--max_steps=14229 \
--warmup_steps=1422 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tat_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=45533 \
--output_dir=models/100mb/tat_cyrl_100mb
cp tokenizers/monolingual/tat_cyrl_100mb/* models/100mb/tat_cyrl_100mb

# tel_telu
if test -f models/100mb/tel_telu_100mb/pytorch_model.bin; then
echo "Model already found: tel_telu_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tel_telu_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tel_telu_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=639 --save_steps=999999999 \
--max_steps=12780 \
--warmup_steps=1278 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tel_telu_100mb.txt \
--seed=43 \
--override_n_examples=40898 \
--output_dir=models/100mb/tel_telu_100mb
cp tokenizers/monolingual/tel_telu_100mb/* models/100mb/tel_telu_100mb

# tgk_cyrl
if test -f models/100mb/tgk_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: tgk_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tgk_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tgk_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=662 --save_steps=999999999 \
--max_steps=13243 \
--warmup_steps=1324 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tgk_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=42379 \
--output_dir=models/100mb/tgk_cyrl_100mb
cp tokenizers/monolingual/tgk_cyrl_100mb/* models/100mb/tgk_cyrl_100mb

# tgl_latn
if test -f models/100mb/tgl_latn_100mb/pytorch_model.bin; then
echo "Model already found: tgl_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tgl_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tgl_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=749 --save_steps=999999999 \
--max_steps=14980 \
--warmup_steps=1498 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tgl_latn_100mb.txt \
--seed=43 \
--override_n_examples=47936 \
--output_dir=models/100mb/tgl_latn_100mb
cp tokenizers/monolingual/tgl_latn_100mb/* models/100mb/tgl_latn_100mb

# tha_thai
if test -f models/100mb/tha_thai_100mb/pytorch_model.bin; then
echo "Model already found: tha_thai_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tha_thai_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tha_thai_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=628 --save_steps=999999999 \
--max_steps=12579 \
--warmup_steps=1257 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tha_thai_100mb.txt \
--seed=43 \
--override_n_examples=40253 \
--output_dir=models/100mb/tha_thai_100mb
cp tokenizers/monolingual/tha_thai_100mb/* models/100mb/tha_thai_100mb

# tir_ethi
if test -f models/100mb/tir_ethi_100mb/pytorch_model.bin; then
echo "Model already found: tir_ethi_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tir_ethi_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tir_ethi_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=681 --save_steps=999999999 \
--max_steps=13634 \
--warmup_steps=1363 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tir_ethi_100mb.txt \
--seed=43 \
--override_n_examples=43629 \
--output_dir=models/100mb/tir_ethi_100mb
cp tokenizers/monolingual/tir_ethi_100mb/* models/100mb/tir_ethi_100mb

# tsn_latn
if test -f models/100mb/tsn_latn_100mb/pytorch_model.bin; then
echo "Model already found: tsn_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tsn_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tsn_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=18871 \
--warmup_steps=1887 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tsn_latn_100mb.txt \
--seed=43 \
--override_n_examples=60390 \
--output_dir=models/100mb/tsn_latn_100mb
cp tokenizers/monolingual/tsn_latn_100mb/* models/100mb/tsn_latn_100mb

# tuk_latn
if test -f models/100mb/tuk_latn_100mb/pytorch_model.bin; then
echo "Model already found: tuk_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tuk_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tuk_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=935 --save_steps=999999999 \
--max_steps=18713 \
--warmup_steps=1871 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tuk_latn_100mb.txt \
--seed=43 \
--override_n_examples=59884 \
--output_dir=models/100mb/tuk_latn_100mb
cp tokenizers/monolingual/tuk_latn_100mb/* models/100mb/tuk_latn_100mb

# tur_latn
if test -f models/100mb/tur_latn_100mb/pytorch_model.bin; then
echo "Model already found: tur_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tur_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tur_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=570 --save_steps=999999999 \
--max_steps=11419 \
--warmup_steps=1141 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tur_latn_100mb.txt \
--seed=43 \
--override_n_examples=36541 \
--output_dir=models/100mb/tur_latn_100mb
cp tokenizers/monolingual/tur_latn_100mb/* models/100mb/tur_latn_100mb

# uig_arab
if test -f models/100mb/uig_arab_100mb/pytorch_model.bin; then
echo "Model already found: uig_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uig_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uig_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=799 --save_steps=999999999 \
--max_steps=15981 \
--warmup_steps=1598 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uig_arab_100mb.txt \
--seed=43 \
--override_n_examples=51140 \
--output_dir=models/100mb/uig_arab_100mb
cp tokenizers/monolingual/uig_arab_100mb/* models/100mb/uig_arab_100mb

# ukr_cyrl
if test -f models/100mb/ukr_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: ukr_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ukr_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ukr_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=658 --save_steps=999999999 \
--max_steps=13165 \
--warmup_steps=1316 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ukr_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=42131 \
--output_dir=models/100mb/ukr_cyrl_100mb
cp tokenizers/monolingual/ukr_cyrl_100mb/* models/100mb/ukr_cyrl_100mb

# urd_arab
if test -f models/100mb/urd_arab_100mb/pytorch_model.bin; then
echo "Model already found: urd_arab_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/urd_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/urd_arab_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=756 --save_steps=999999999 \
--max_steps=15137 \
--warmup_steps=1513 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/urd_arab_100mb.txt \
--seed=43 \
--override_n_examples=48441 \
--output_dir=models/100mb/urd_arab_100mb
cp tokenizers/monolingual/urd_arab_100mb/* models/100mb/urd_arab_100mb

# uzb_cyrl
if test -f models/100mb/uzb_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: uzb_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzb_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzb_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=643 --save_steps=999999999 \
--max_steps=12879 \
--warmup_steps=1287 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzb_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=41214 \
--output_dir=models/100mb/uzb_cyrl_100mb
cp tokenizers/monolingual/uzb_cyrl_100mb/* models/100mb/uzb_cyrl_100mb

# uzb_latn
if test -f models/100mb/uzb_latn_100mb/pytorch_model.bin; then
echo "Model already found: uzb_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzb_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzb_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=796 --save_steps=999999999 \
--max_steps=15935 \
--warmup_steps=1593 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzb_latn_100mb.txt \
--seed=43 \
--override_n_examples=50995 \
--output_dir=models/100mb/uzb_latn_100mb
cp tokenizers/monolingual/uzb_latn_100mb/* models/100mb/uzb_latn_100mb

# uzn_cyrl
if test -f models/100mb/uzn_cyrl_100mb/pytorch_model.bin; then
echo "Model already found: uzn_cyrl_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzn_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzn_cyrl_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=656 --save_steps=999999999 \
--max_steps=13137 \
--warmup_steps=1313 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzn_cyrl_100mb.txt \
--seed=43 \
--override_n_examples=42040 \
--output_dir=models/100mb/uzn_cyrl_100mb
cp tokenizers/monolingual/uzn_cyrl_100mb/* models/100mb/uzn_cyrl_100mb

# uzn_latn
if test -f models/100mb/uzn_latn_100mb/pytorch_model.bin; then
echo "Model already found: uzn_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzn_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzn_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=1090 --save_steps=999999999 \
--max_steps=21813 \
--warmup_steps=2181 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzn_latn_100mb.txt \
--seed=43 \
--override_n_examples=69802 \
--output_dir=models/100mb/uzn_latn_100mb
cp tokenizers/monolingual/uzn_latn_100mb/* models/100mb/uzn_latn_100mb

# vec_latn
if test -f models/100mb/vec_latn_100mb/pytorch_model.bin; then
echo "Model already found: vec_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vec_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vec_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=821 --save_steps=999999999 \
--max_steps=16420 \
--warmup_steps=1642 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vec_latn_100mb.txt \
--seed=43 \
--override_n_examples=52547 \
--output_dir=models/100mb/vec_latn_100mb
cp tokenizers/monolingual/vec_latn_100mb/* models/100mb/vec_latn_100mb

# vie_latn
if test -f models/100mb/vie_latn_100mb/pytorch_model.bin; then
echo "Model already found: vie_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vie_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vie_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=800 --save_steps=999999999 \
--max_steps=16005 \
--warmup_steps=1600 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vie_latn_100mb.txt \
--seed=43 \
--override_n_examples=51217 \
--output_dir=models/100mb/vie_latn_100mb
cp tokenizers/monolingual/vie_latn_100mb/* models/100mb/vie_latn_100mb

# war_latn
if test -f models/100mb/war_latn_100mb/pytorch_model.bin; then
echo "Model already found: war_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/war_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/war_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=853 --save_steps=999999999 \
--max_steps=17063 \
--warmup_steps=1706 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/war_latn_100mb.txt \
--seed=43 \
--override_n_examples=54604 \
--output_dir=models/100mb/war_latn_100mb
cp tokenizers/monolingual/war_latn_100mb/* models/100mb/war_latn_100mb

# wln_latn
if test -f models/100mb/wln_latn_100mb/pytorch_model.bin; then
echo "Model already found: wln_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/wln_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/wln_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=no --save_strategy=no \
--eval_steps=999999999 --save_steps=999999999 \
--max_steps=17353 \
--warmup_steps=1735 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/wln_latn_100mb.txt \
--seed=43 \
--override_n_examples=55532 \
--output_dir=models/100mb/wln_latn_100mb
cp tokenizers/monolingual/wln_latn_100mb/* models/100mb/wln_latn_100mb

# xho_latn
if test -f models/100mb/xho_latn_100mb/pytorch_model.bin; then
echo "Model already found: xho_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/xho_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/xho_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=817 --save_steps=999999999 \
--max_steps=16347 \
--warmup_steps=1634 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/xho_latn_100mb.txt \
--seed=43 \
--override_n_examples=52312 \
--output_dir=models/100mb/xho_latn_100mb
cp tokenizers/monolingual/xho_latn_100mb/* models/100mb/xho_latn_100mb

# ydd_hebr
if test -f models/100mb/ydd_hebr_100mb/pytorch_model.bin; then
echo "Model already found: ydd_hebr_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ydd_hebr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ydd_hebr_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=718 --save_steps=999999999 \
--max_steps=14374 \
--warmup_steps=1437 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ydd_hebr_100mb.txt \
--seed=43 \
--override_n_examples=45998 \
--output_dir=models/100mb/ydd_hebr_100mb
cp tokenizers/monolingual/ydd_hebr_100mb/* models/100mb/ydd_hebr_100mb

# yid_hebr
if test -f models/100mb/yid_hebr_100mb/pytorch_model.bin; then
echo "Model already found: yid_hebr_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yid_hebr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yid_hebr_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=586 --save_steps=999999999 \
--max_steps=11728 \
--warmup_steps=1172 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yid_hebr_100mb.txt \
--seed=43 \
--override_n_examples=37532 \
--output_dir=models/100mb/yid_hebr_100mb
cp tokenizers/monolingual/yid_hebr_100mb/* models/100mb/yid_hebr_100mb

# yor_latn
if test -f models/100mb/yor_latn_100mb/pytorch_model.bin; then
echo "Model already found: yor_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/yor_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/yor_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=946 --save_steps=999999999 \
--max_steps=18927 \
--warmup_steps=1892 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/yor_latn_100mb.txt \
--seed=43 \
--override_n_examples=60568 \
--output_dir=models/100mb/yor_latn_100mb
cp tokenizers/monolingual/yor_latn_100mb/* models/100mb/yor_latn_100mb

# zho_hans
if test -f models/100mb/zho_hans_100mb/pytorch_model.bin; then
echo "Model already found: zho_hans_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zho_hans_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zho_hans_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=629 --save_steps=999999999 \
--max_steps=12595 \
--warmup_steps=1259 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zho_hans_100mb.txt \
--seed=43 \
--override_n_examples=40305 \
--output_dir=models/100mb/zho_hans_100mb
cp tokenizers/monolingual/zho_hans_100mb/* models/100mb/zho_hans_100mb

# zho_hant
if test -f models/100mb/zho_hant_100mb/pytorch_model.bin; then
echo "Model already found: zho_hant_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zho_hant_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zho_hant_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=735 --save_steps=999999999 \
--max_steps=14706 \
--warmup_steps=1470 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zho_hant_100mb.txt \
--seed=43 \
--override_n_examples=47061 \
--output_dir=models/100mb/zho_hant_100mb
cp tokenizers/monolingual/zho_hant_100mb/* models/100mb/zho_hant_100mb

# zsm_latn
if test -f models/100mb/zsm_latn_100mb/pytorch_model.bin; then
echo "Model already found: zsm_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zsm_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zsm_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=659 --save_steps=999999999 \
--max_steps=13198 \
--warmup_steps=1319 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zsm_latn_100mb.txt \
--seed=43 \
--override_n_examples=42236 \
--output_dir=models/100mb/zsm_latn_100mb
cp tokenizers/monolingual/zsm_latn_100mb/* models/100mb/zsm_latn_100mb

# zul_latn
if test -f models/100mb/zul_latn_100mb/pytorch_model.bin; then
echo "Model already found: zul_latn_100mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zul_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zul_latn_100mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=4 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=no \
--eval_steps=798 --save_steps=999999999 \
--max_steps=15968 \
--warmup_steps=1596 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zul_latn_100mb.txt \
--seed=43 \
--override_n_examples=51099 \
--output_dir=models/100mb/zul_latn_100mb
cp tokenizers/monolingual/zul_latn_100mb/* models/100mb/zul_latn_100mb
