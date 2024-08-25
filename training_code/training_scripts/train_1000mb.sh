export CUDA_VISIBLE_DEVICES=0

# afr_latn
if test -f models/1000mb/afr_latn_1000mb/pytorch_model.bin; then
echo "Model already found: afr_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/afr_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/afr_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3657 --save_steps=3657 \
--max_steps=73145 \
--warmup_steps=7314 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/afr_latn_1000mb.txt \
--seed=43 \
--override_n_examples=468129 \
--output_dir=models/1000mb/afr_latn_1000mb
cp tokenizers/monolingual/afr_latn_100mb/* models/1000mb/afr_latn_1000mb

# amh_ethi
if test -f models/1000mb/amh_ethi_1000mb/pytorch_model.bin; then
echo "Model already found: amh_ethi_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/amh_ethi_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/amh_ethi_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3231 --save_steps=3231 \
--max_steps=64626 \
--warmup_steps=6462 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/amh_ethi_1000mb.txt \
--seed=43 \
--override_n_examples=413609 \
--output_dir=models/1000mb/amh_ethi_1000mb
cp tokenizers/monolingual/amh_ethi_100mb/* models/1000mb/amh_ethi_1000mb

# arb_arab
if test -f models/1000mb/arb_arab_1000mb/pytorch_model.bin; then
echo "Model already found: arb_arab_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/arb_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/arb_arab_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2993 --save_steps=2993 \
--max_steps=59874 \
--warmup_steps=5987 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/arb_arab_1000mb.txt \
--seed=43 \
--override_n_examples=383198 \
--output_dir=models/1000mb/arb_arab_1000mb
cp tokenizers/monolingual/arb_arab_100mb/* models/1000mb/arb_arab_1000mb

# aze_latn
if test -f models/1000mb/aze_latn_1000mb/pytorch_model.bin; then
echo "Model already found: aze_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/aze_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/aze_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3556 --save_steps=3556 \
--max_steps=71133 \
--warmup_steps=7113 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/aze_latn_1000mb.txt \
--seed=43 \
--override_n_examples=455257 \
--output_dir=models/1000mb/aze_latn_1000mb
cp tokenizers/monolingual/aze_latn_100mb/* models/1000mb/aze_latn_1000mb

# bel_cyrl
if test -f models/1000mb/bel_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: bel_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bel_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bel_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3877 --save_steps=3877 \
--max_steps=77556 \
--warmup_steps=7755 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bel_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=496364 \
--output_dir=models/1000mb/bel_cyrl_1000mb
cp tokenizers/monolingual/bel_cyrl_100mb/* models/1000mb/bel_cyrl_1000mb

# ben_beng
if test -f models/1000mb/ben_beng_1000mb/pytorch_model.bin; then
echo "Model already found: ben_beng_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ben_beng_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ben_beng_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2971 --save_steps=2971 \
--max_steps=59429 \
--warmup_steps=5942 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ben_beng_1000mb.txt \
--seed=43 \
--override_n_examples=380346 \
--output_dir=models/1000mb/ben_beng_1000mb
cp tokenizers/monolingual/ben_beng_100mb/* models/1000mb/ben_beng_1000mb

# bos_cyrl
if test -f models/1000mb/bos_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: bos_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bos_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bos_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3547 --save_steps=3547 \
--max_steps=70953 \
--warmup_steps=7095 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bos_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=454105 \
--output_dir=models/1000mb/bos_cyrl_1000mb
cp tokenizers/monolingual/bos_cyrl_100mb/* models/1000mb/bos_cyrl_1000mb

# bos_latn
if test -f models/1000mb/bos_latn_1000mb/pytorch_model.bin; then
echo "Model already found: bos_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bos_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bos_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3483 --save_steps=3483 \
--max_steps=69661 \
--warmup_steps=6966 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bos_latn_1000mb.txt \
--seed=43 \
--override_n_examples=445833 \
--output_dir=models/1000mb/bos_latn_1000mb
cp tokenizers/monolingual/bos_latn_100mb/* models/1000mb/bos_latn_1000mb

# bul_cyrl
if test -f models/1000mb/bul_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: bul_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/bul_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/bul_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3423 --save_steps=3423 \
--max_steps=68465 \
--warmup_steps=6846 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/bul_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=438176 \
--output_dir=models/1000mb/bul_cyrl_1000mb
cp tokenizers/monolingual/bul_cyrl_100mb/* models/1000mb/bul_cyrl_1000mb

# cat_latn
if test -f models/1000mb/cat_latn_1000mb/pytorch_model.bin; then
echo "Model already found: cat_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cat_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cat_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3645 --save_steps=3645 \
--max_steps=72911 \
--warmup_steps=7291 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cat_latn_1000mb.txt \
--seed=43 \
--override_n_examples=466631 \
--output_dir=models/1000mb/cat_latn_1000mb
cp tokenizers/monolingual/cat_latn_100mb/* models/1000mb/cat_latn_1000mb

# ces_latn
if test -f models/1000mb/ces_latn_1000mb/pytorch_model.bin; then
echo "Model already found: ces_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ces_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ces_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3145 --save_steps=3145 \
--max_steps=62900 \
--warmup_steps=6290 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ces_latn_1000mb.txt \
--seed=43 \
--override_n_examples=402565 \
--output_dir=models/1000mb/ces_latn_1000mb
cp tokenizers/monolingual/ces_latn_100mb/* models/1000mb/ces_latn_1000mb

# cym_latn
if test -f models/1000mb/cym_latn_1000mb/pytorch_model.bin; then
echo "Model already found: cym_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/cym_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/cym_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3604 --save_steps=3604 \
--max_steps=72091 \
--warmup_steps=7209 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/cym_latn_1000mb.txt \
--seed=43 \
--override_n_examples=461387 \
--output_dir=models/1000mb/cym_latn_1000mb
cp tokenizers/monolingual/cym_latn_100mb/* models/1000mb/cym_latn_1000mb

# dan_latn
if test -f models/1000mb/dan_latn_1000mb/pytorch_model.bin; then
echo "Model already found: dan_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/dan_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/dan_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3175 --save_steps=3175 \
--max_steps=63502 \
--warmup_steps=6350 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/dan_latn_1000mb.txt \
--seed=43 \
--override_n_examples=406417 \
--output_dir=models/1000mb/dan_latn_1000mb
cp tokenizers/monolingual/dan_latn_100mb/* models/1000mb/dan_latn_1000mb

# deu_latn
if test -f models/1000mb/deu_latn_1000mb/pytorch_model.bin; then
echo "Model already found: deu_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/deu_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/deu_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3216 --save_steps=3216 \
--max_steps=64336 \
--warmup_steps=6433 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/deu_latn_1000mb.txt \
--seed=43 \
--override_n_examples=411752 \
--output_dir=models/1000mb/deu_latn_1000mb
cp tokenizers/monolingual/deu_latn_100mb/* models/1000mb/deu_latn_1000mb

# ell_grek
if test -f models/1000mb/ell_grek_1000mb/pytorch_model.bin; then
echo "Model already found: ell_grek_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ell_grek_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ell_grek_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3642 --save_steps=3642 \
--max_steps=72846 \
--warmup_steps=7284 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ell_grek_1000mb.txt \
--seed=43 \
--override_n_examples=466219 \
--output_dir=models/1000mb/ell_grek_1000mb
cp tokenizers/monolingual/ell_grek_100mb/* models/1000mb/ell_grek_1000mb

# eng_latn
if test -f models/1000mb/eng_latn_1000mb/pytorch_model.bin; then
echo "Model already found: eng_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/eng_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/eng_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3265 --save_steps=3265 \
--max_steps=65300 \
--warmup_steps=6530 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/eng_latn_1000mb.txt \
--seed=43 \
--override_n_examples=417924 \
--output_dir=models/1000mb/eng_latn_1000mb
cp tokenizers/monolingual/eng_latn_100mb/* models/1000mb/eng_latn_1000mb

# epo_latn
if test -f models/1000mb/epo_latn_1000mb/pytorch_model.bin; then
echo "Model already found: epo_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/epo_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/epo_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3530 --save_steps=3530 \
--max_steps=70612 \
--warmup_steps=7061 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/epo_latn_1000mb.txt \
--seed=43 \
--override_n_examples=451923 \
--output_dir=models/1000mb/epo_latn_1000mb
cp tokenizers/monolingual/epo_latn_100mb/* models/1000mb/epo_latn_1000mb

# est_latn
if test -f models/1000mb/est_latn_1000mb/pytorch_model.bin; then
echo "Model already found: est_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/est_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/est_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2891 --save_steps=2891 \
--max_steps=57836 \
--warmup_steps=5783 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/est_latn_1000mb.txt \
--seed=43 \
--override_n_examples=370153 \
--output_dir=models/1000mb/est_latn_1000mb
cp tokenizers/monolingual/est_latn_100mb/* models/1000mb/est_latn_1000mb

# eus_latn
if test -f models/1000mb/eus_latn_1000mb/pytorch_model.bin; then
echo "Model already found: eus_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/eus_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/eus_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3203 --save_steps=3203 \
--max_steps=64062 \
--warmup_steps=6406 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/eus_latn_1000mb.txt \
--seed=43 \
--override_n_examples=410003 \
--output_dir=models/1000mb/eus_latn_1000mb
cp tokenizers/monolingual/eus_latn_100mb/* models/1000mb/eus_latn_1000mb

# fas_arab
if test -f models/1000mb/fas_arab_1000mb/pytorch_model.bin; then
echo "Model already found: fas_arab_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fas_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fas_arab_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3728 --save_steps=3728 \
--max_steps=74572 \
--warmup_steps=7457 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fas_arab_1000mb.txt \
--seed=43 \
--override_n_examples=477265 \
--output_dir=models/1000mb/fas_arab_1000mb
cp tokenizers/monolingual/fas_arab_100mb/* models/1000mb/fas_arab_1000mb

# fil_latn
if test -f models/1000mb/fil_latn_1000mb/pytorch_model.bin; then
echo "Model already found: fil_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fil_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fil_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=4195 --save_steps=4195 \
--max_steps=83909 \
--warmup_steps=8390 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fil_latn_1000mb.txt \
--seed=43 \
--override_n_examples=537023 \
--output_dir=models/1000mb/fil_latn_1000mb
cp tokenizers/monolingual/fil_latn_100mb/* models/1000mb/fil_latn_1000mb

# fin_latn
if test -f models/1000mb/fin_latn_1000mb/pytorch_model.bin; then
echo "Model already found: fin_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fin_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fin_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2838 --save_steps=2838 \
--max_steps=56778 \
--warmup_steps=5677 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fin_latn_1000mb.txt \
--seed=43 \
--override_n_examples=363380 \
--output_dir=models/1000mb/fin_latn_1000mb
cp tokenizers/monolingual/fin_latn_100mb/* models/1000mb/fin_latn_1000mb

# fra_latn
if test -f models/1000mb/fra_latn_1000mb/pytorch_model.bin; then
echo "Model already found: fra_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/fra_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/fra_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3836 --save_steps=3836 \
--max_steps=76725 \
--warmup_steps=7672 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/fra_latn_1000mb.txt \
--seed=43 \
--override_n_examples=491046 \
--output_dir=models/1000mb/fra_latn_1000mb
cp tokenizers/monolingual/fra_latn_100mb/* models/1000mb/fra_latn_1000mb

# glg_latn
if test -f models/1000mb/glg_latn_1000mb/pytorch_model.bin; then
echo "Model already found: glg_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/glg_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/glg_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3388 --save_steps=3388 \
--max_steps=67773 \
--warmup_steps=6777 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/glg_latn_1000mb.txt \
--seed=43 \
--override_n_examples=433750 \
--output_dir=models/1000mb/glg_latn_1000mb
cp tokenizers/monolingual/glg_latn_100mb/* models/1000mb/glg_latn_1000mb

# guj_gujr
if test -f models/1000mb/guj_gujr_1000mb/pytorch_model.bin; then
echo "Model already found: guj_gujr_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/guj_gujr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/guj_gujr_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2957 --save_steps=2957 \
--max_steps=59141 \
--warmup_steps=5914 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/guj_gujr_1000mb.txt \
--seed=43 \
--override_n_examples=378505 \
--output_dir=models/1000mb/guj_gujr_1000mb
cp tokenizers/monolingual/guj_gujr_100mb/* models/1000mb/guj_gujr_1000mb

# hau_latn
if test -f models/1000mb/hau_latn_1000mb/pytorch_model.bin; then
echo "Model already found: hau_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hau_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hau_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=4233 --save_steps=4233 \
--max_steps=84660 \
--warmup_steps=8466 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hau_latn_1000mb.txt \
--seed=43 \
--override_n_examples=541829 \
--output_dir=models/1000mb/hau_latn_1000mb
cp tokenizers/monolingual/hau_latn_100mb/* models/1000mb/hau_latn_1000mb

# heb_hebr
if test -f models/1000mb/heb_hebr_1000mb/pytorch_model.bin; then
echo "Model already found: heb_hebr_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/heb_hebr_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/heb_hebr_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2943 --save_steps=2943 \
--max_steps=58869 \
--warmup_steps=5886 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/heb_hebr_1000mb.txt \
--seed=43 \
--override_n_examples=376767 \
--output_dir=models/1000mb/heb_hebr_1000mb
cp tokenizers/monolingual/heb_hebr_100mb/* models/1000mb/heb_hebr_1000mb

# hin_deva
if test -f models/1000mb/hin_deva_1000mb/pytorch_model.bin; then
echo "Model already found: hin_deva_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hin_deva_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hin_deva_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3479 --save_steps=3479 \
--max_steps=69586 \
--warmup_steps=6958 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hin_deva_1000mb.txt \
--seed=43 \
--override_n_examples=445353 \
--output_dir=models/1000mb/hin_deva_1000mb
cp tokenizers/monolingual/hin_deva_100mb/* models/1000mb/hin_deva_1000mb

# hrv_latn
if test -f models/1000mb/hrv_latn_1000mb/pytorch_model.bin; then
echo "Model already found: hrv_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hrv_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hrv_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3348 --save_steps=3348 \
--max_steps=66962 \
--warmup_steps=6696 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hrv_latn_1000mb.txt \
--seed=43 \
--override_n_examples=428559 \
--output_dir=models/1000mb/hrv_latn_1000mb
cp tokenizers/monolingual/hrv_latn_100mb/* models/1000mb/hrv_latn_1000mb

# hun_latn
if test -f models/1000mb/hun_latn_1000mb/pytorch_model.bin; then
echo "Model already found: hun_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hun_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hun_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2915 --save_steps=2915 \
--max_steps=58315 \
--warmup_steps=5831 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hun_latn_1000mb.txt \
--seed=43 \
--override_n_examples=373222 \
--output_dir=models/1000mb/hun_latn_1000mb
cp tokenizers/monolingual/hun_latn_100mb/* models/1000mb/hun_latn_1000mb

# hye_armn
if test -f models/1000mb/hye_armn_1000mb/pytorch_model.bin; then
echo "Model already found: hye_armn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/hye_armn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/hye_armn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3107 --save_steps=3107 \
--max_steps=62143 \
--warmup_steps=6214 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/hye_armn_1000mb.txt \
--seed=43 \
--override_n_examples=397716 \
--output_dir=models/1000mb/hye_armn_1000mb
cp tokenizers/monolingual/hye_armn_100mb/* models/1000mb/hye_armn_1000mb

# ind_latn
if test -f models/1000mb/ind_latn_1000mb/pytorch_model.bin; then
echo "Model already found: ind_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ind_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ind_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3210 --save_steps=3210 \
--max_steps=64218 \
--warmup_steps=6421 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ind_latn_1000mb.txt \
--seed=43 \
--override_n_examples=411000 \
--output_dir=models/1000mb/ind_latn_1000mb
cp tokenizers/monolingual/ind_latn_100mb/* models/1000mb/ind_latn_1000mb

# isl_latn
if test -f models/1000mb/isl_latn_1000mb/pytorch_model.bin; then
echo "Model already found: isl_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/isl_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/isl_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3614 --save_steps=3614 \
--max_steps=72287 \
--warmup_steps=7228 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/isl_latn_1000mb.txt \
--seed=43 \
--override_n_examples=462642 \
--output_dir=models/1000mb/isl_latn_1000mb
cp tokenizers/monolingual/isl_latn_100mb/* models/1000mb/isl_latn_1000mb

# ita_latn
if test -f models/1000mb/ita_latn_1000mb/pytorch_model.bin; then
echo "Model already found: ita_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ita_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ita_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3297 --save_steps=3297 \
--max_steps=65948 \
--warmup_steps=6594 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ita_latn_1000mb.txt \
--seed=43 \
--override_n_examples=422070 \
--output_dir=models/1000mb/ita_latn_1000mb
cp tokenizers/monolingual/ita_latn_100mb/* models/1000mb/ita_latn_1000mb

# jpn_jpan
if test -f models/1000mb/jpn_jpan_1000mb/pytorch_model.bin; then
echo "Model already found: jpn_jpan_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/jpn_jpan_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/jpn_jpan_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3342 --save_steps=3342 \
--max_steps=66852 \
--warmup_steps=6685 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/jpn_jpan_1000mb.txt \
--seed=43 \
--override_n_examples=427858 \
--output_dir=models/1000mb/jpn_jpan_1000mb
cp tokenizers/monolingual/jpn_jpan_100mb/* models/1000mb/jpn_jpan_1000mb

# kaa_cyrl
if test -f models/1000mb/kaa_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: kaa_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaa_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaa_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3236 --save_steps=3236 \
--max_steps=64727 \
--warmup_steps=6472 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaa_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=414259 \
--output_dir=models/1000mb/kaa_cyrl_1000mb
cp tokenizers/monolingual/kaa_cyrl_100mb/* models/1000mb/kaa_cyrl_1000mb

# kan_knda
if test -f models/1000mb/kan_knda_1000mb/pytorch_model.bin; then
echo "Model already found: kan_knda_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kan_knda_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kan_knda_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3245 --save_steps=3245 \
--max_steps=64905 \
--warmup_steps=6490 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kan_knda_1000mb.txt \
--seed=43 \
--override_n_examples=415397 \
--output_dir=models/1000mb/kan_knda_1000mb
cp tokenizers/monolingual/kan_knda_100mb/* models/1000mb/kan_knda_1000mb

# kat_geor
if test -f models/1000mb/kat_geor_1000mb/pytorch_model.bin; then
echo "Model already found: kat_geor_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kat_geor_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kat_geor_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=5413 --save_steps=5413 \
--max_steps=108265 \
--warmup_steps=10826 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kat_geor_1000mb.txt \
--seed=43 \
--override_n_examples=692896 \
--output_dir=models/1000mb/kat_geor_1000mb
cp tokenizers/monolingual/kat_geor_100mb/* models/1000mb/kat_geor_1000mb

# kaz_cyrl
if test -f models/1000mb/kaz_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: kaz_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kaz_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kaz_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3051 --save_steps=3051 \
--max_steps=61026 \
--warmup_steps=6102 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kaz_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=390567 \
--output_dir=models/1000mb/kaz_cyrl_1000mb
cp tokenizers/monolingual/kaz_cyrl_100mb/* models/1000mb/kaz_cyrl_1000mb

# kir_cyrl
if test -f models/1000mb/kir_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: kir_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kir_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kir_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3403 --save_steps=3403 \
--max_steps=68074 \
--warmup_steps=6807 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kir_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=435676 \
--output_dir=models/1000mb/kir_cyrl_1000mb
cp tokenizers/monolingual/kir_cyrl_100mb/* models/1000mb/kir_cyrl_1000mb

# kor_hang
if test -f models/1000mb/kor_hang_1000mb/pytorch_model.bin; then
echo "Model already found: kor_hang_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/kor_hang_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/kor_hang_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3464 --save_steps=3464 \
--max_steps=69281 \
--warmup_steps=6928 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/kor_hang_1000mb.txt \
--seed=43 \
--override_n_examples=443402 \
--output_dir=models/1000mb/kor_hang_1000mb
cp tokenizers/monolingual/kor_hang_100mb/* models/1000mb/kor_hang_1000mb

# lat_latn
if test -f models/1000mb/lat_latn_1000mb/pytorch_model.bin; then
echo "Model already found: lat_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lat_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lat_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2880 --save_steps=2880 \
--max_steps=57609 \
--warmup_steps=5760 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lat_latn_1000mb.txt \
--seed=43 \
--override_n_examples=368701 \
--output_dir=models/1000mb/lat_latn_1000mb
cp tokenizers/monolingual/lat_latn_100mb/* models/1000mb/lat_latn_1000mb

# lav_latn
if test -f models/1000mb/lav_latn_1000mb/pytorch_model.bin; then
echo "Model already found: lav_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lav_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lav_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3714 --save_steps=3714 \
--max_steps=74280 \
--warmup_steps=7428 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lav_latn_1000mb.txt \
--seed=43 \
--override_n_examples=475394 \
--output_dir=models/1000mb/lav_latn_1000mb
cp tokenizers/monolingual/lav_latn_100mb/* models/1000mb/lav_latn_1000mb

# lit_latn
if test -f models/1000mb/lit_latn_1000mb/pytorch_model.bin; then
echo "Model already found: lit_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/lit_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/lit_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3070 --save_steps=3070 \
--max_steps=61410 \
--warmup_steps=6141 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/lit_latn_1000mb.txt \
--seed=43 \
--override_n_examples=393025 \
--output_dir=models/1000mb/lit_latn_1000mb
cp tokenizers/monolingual/lit_latn_100mb/* models/1000mb/lit_latn_1000mb

# mal_mlym
if test -f models/1000mb/mal_mlym_1000mb/pytorch_model.bin; then
echo "Model already found: mal_mlym_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mal_mlym_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mal_mlym_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3733 --save_steps=3733 \
--max_steps=74679 \
--warmup_steps=7467 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mal_mlym_1000mb.txt \
--seed=43 \
--override_n_examples=477947 \
--output_dir=models/1000mb/mal_mlym_1000mb
cp tokenizers/monolingual/mal_mlym_100mb/* models/1000mb/mal_mlym_1000mb

# mar_deva
if test -f models/1000mb/mar_deva_1000mb/pytorch_model.bin; then
echo "Model already found: mar_deva_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mar_deva_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mar_deva_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3152 --save_steps=3152 \
--max_steps=63058 \
--warmup_steps=6305 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mar_deva_1000mb.txt \
--seed=43 \
--override_n_examples=403575 \
--output_dir=models/1000mb/mar_deva_1000mb
cp tokenizers/monolingual/mar_deva_100mb/* models/1000mb/mar_deva_1000mb

# mkd_cyrl
if test -f models/1000mb/mkd_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: mkd_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mkd_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mkd_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3377 --save_steps=3377 \
--max_steps=67549 \
--warmup_steps=6754 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mkd_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=432317 \
--output_dir=models/1000mb/mkd_cyrl_1000mb
cp tokenizers/monolingual/mkd_cyrl_100mb/* models/1000mb/mkd_cyrl_1000mb

# mlt_latn
if test -f models/1000mb/mlt_latn_1000mb/pytorch_model.bin; then
echo "Model already found: mlt_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mlt_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mlt_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=4320 --save_steps=4320 \
--max_steps=86413 \
--warmup_steps=8641 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mlt_latn_1000mb.txt \
--seed=43 \
--override_n_examples=553044 \
--output_dir=models/1000mb/mlt_latn_1000mb
cp tokenizers/monolingual/mlt_latn_100mb/* models/1000mb/mlt_latn_1000mb

# mon_cyrl
if test -f models/1000mb/mon_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: mon_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/mon_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/mon_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3139 --save_steps=3139 \
--max_steps=62786 \
--warmup_steps=6278 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/mon_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=401831 \
--output_dir=models/1000mb/mon_cyrl_1000mb
cp tokenizers/monolingual/mon_cyrl_100mb/* models/1000mb/mon_cyrl_1000mb

# msa_latn
if test -f models/1000mb/msa_latn_1000mb/pytorch_model.bin; then
echo "Model already found: msa_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/msa_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/msa_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3606 --save_steps=3606 \
--max_steps=72134 \
--warmup_steps=7213 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/msa_latn_1000mb.txt \
--seed=43 \
--override_n_examples=461663 \
--output_dir=models/1000mb/msa_latn_1000mb
cp tokenizers/monolingual/msa_latn_100mb/* models/1000mb/msa_latn_1000mb

# nep_deva
if test -f models/1000mb/nep_deva_1000mb/pytorch_model.bin; then
echo "Model already found: nep_deva_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nep_deva_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nep_deva_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3286 --save_steps=3286 \
--max_steps=65725 \
--warmup_steps=6572 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nep_deva_1000mb.txt \
--seed=43 \
--override_n_examples=420641 \
--output_dir=models/1000mb/nep_deva_1000mb
cp tokenizers/monolingual/nep_deva_100mb/* models/1000mb/nep_deva_1000mb

# nld_latn
if test -f models/1000mb/nld_latn_1000mb/pytorch_model.bin; then
echo "Model already found: nld_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nld_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nld_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3310 --save_steps=3310 \
--max_steps=66216 \
--warmup_steps=6621 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nld_latn_1000mb.txt \
--seed=43 \
--override_n_examples=423786 \
--output_dir=models/1000mb/nld_latn_1000mb
cp tokenizers/monolingual/nld_latn_100mb/* models/1000mb/nld_latn_1000mb

# nob_latn
if test -f models/1000mb/nob_latn_1000mb/pytorch_model.bin; then
echo "Model already found: nob_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nob_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nob_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3142 --save_steps=3142 \
--max_steps=62850 \
--warmup_steps=6285 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nob_latn_1000mb.txt \
--seed=43 \
--override_n_examples=402246 \
--output_dir=models/1000mb/nob_latn_1000mb
cp tokenizers/monolingual/nob_latn_100mb/* models/1000mb/nob_latn_1000mb

# nor_latn
if test -f models/1000mb/nor_latn_1000mb/pytorch_model.bin; then
echo "Model already found: nor_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/nor_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/nor_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3898 --save_steps=3898 \
--max_steps=77967 \
--warmup_steps=7796 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/nor_latn_1000mb.txt \
--seed=43 \
--override_n_examples=498990 \
--output_dir=models/1000mb/nor_latn_1000mb
cp tokenizers/monolingual/nor_latn_100mb/* models/1000mb/nor_latn_1000mb

# pan_guru
if test -f models/1000mb/pan_guru_1000mb/pytorch_model.bin; then
echo "Model already found: pan_guru_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pan_guru_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pan_guru_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3292 --save_steps=3292 \
--max_steps=65849 \
--warmup_steps=6584 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pan_guru_1000mb.txt \
--seed=43 \
--override_n_examples=421436 \
--output_dir=models/1000mb/pan_guru_1000mb
cp tokenizers/monolingual/pan_guru_100mb/* models/1000mb/pan_guru_1000mb

# pes_arab
if test -f models/1000mb/pes_arab_1000mb/pytorch_model.bin; then
echo "Model already found: pes_arab_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pes_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pes_arab_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3295 --save_steps=3295 \
--max_steps=65901 \
--warmup_steps=6590 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pes_arab_1000mb.txt \
--seed=43 \
--override_n_examples=421770 \
--output_dir=models/1000mb/pes_arab_1000mb
cp tokenizers/monolingual/pes_arab_100mb/* models/1000mb/pes_arab_1000mb

# pol_latn
if test -f models/1000mb/pol_latn_1000mb/pytorch_model.bin; then
echo "Model already found: pol_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pol_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pol_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3299 --save_steps=3299 \
--max_steps=65989 \
--warmup_steps=6598 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pol_latn_1000mb.txt \
--seed=43 \
--override_n_examples=422334 \
--output_dir=models/1000mb/pol_latn_1000mb
cp tokenizers/monolingual/pol_latn_100mb/* models/1000mb/pol_latn_1000mb

# por_latn
if test -f models/1000mb/por_latn_1000mb/pytorch_model.bin; then
echo "Model already found: por_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/por_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/por_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3436 --save_steps=3436 \
--max_steps=68738 \
--warmup_steps=6873 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/por_latn_1000mb.txt \
--seed=43 \
--override_n_examples=439926 \
--output_dir=models/1000mb/por_latn_1000mb
cp tokenizers/monolingual/por_latn_100mb/* models/1000mb/por_latn_1000mb

# pus_arab
if test -f models/1000mb/pus_arab_1000mb/pytorch_model.bin; then
echo "Model already found: pus_arab_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/pus_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/pus_arab_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3629 --save_steps=3629 \
--max_steps=72592 \
--warmup_steps=7259 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/pus_arab_1000mb.txt \
--seed=43 \
--override_n_examples=464593 \
--output_dir=models/1000mb/pus_arab_1000mb
cp tokenizers/monolingual/pus_arab_100mb/* models/1000mb/pus_arab_1000mb

# ron_latn
if test -f models/1000mb/ron_latn_1000mb/pytorch_model.bin; then
echo "Model already found: ron_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ron_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ron_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3518 --save_steps=3518 \
--max_steps=70367 \
--warmup_steps=7036 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ron_latn_1000mb.txt \
--seed=43 \
--override_n_examples=450352 \
--output_dir=models/1000mb/ron_latn_1000mb
cp tokenizers/monolingual/ron_latn_100mb/* models/1000mb/ron_latn_1000mb

# rus_cyrl
if test -f models/1000mb/rus_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: rus_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/rus_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/rus_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3364 --save_steps=3364 \
--max_steps=67281 \
--warmup_steps=6728 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/rus_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=430601 \
--output_dir=models/1000mb/rus_cyrl_1000mb
cp tokenizers/monolingual/rus_cyrl_100mb/* models/1000mb/rus_cyrl_1000mb

# sin_sinh
if test -f models/1000mb/sin_sinh_1000mb/pytorch_model.bin; then
echo "Model already found: sin_sinh_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sin_sinh_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sin_sinh_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3556 --save_steps=3556 \
--max_steps=71136 \
--warmup_steps=7113 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sin_sinh_1000mb.txt \
--seed=43 \
--override_n_examples=455271 \
--output_dir=models/1000mb/sin_sinh_1000mb
cp tokenizers/monolingual/sin_sinh_100mb/* models/1000mb/sin_sinh_1000mb

# slk_latn
if test -f models/1000mb/slk_latn_1000mb/pytorch_model.bin; then
echo "Model already found: slk_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/slk_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/slk_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3222 --save_steps=3222 \
--max_steps=64455 \
--warmup_steps=6445 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/slk_latn_1000mb.txt \
--seed=43 \
--override_n_examples=412512 \
--output_dir=models/1000mb/slk_latn_1000mb
cp tokenizers/monolingual/slk_latn_100mb/* models/1000mb/slk_latn_1000mb

# slv_latn
if test -f models/1000mb/slv_latn_1000mb/pytorch_model.bin; then
echo "Model already found: slv_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/slv_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/slv_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3022 --save_steps=3022 \
--max_steps=60440 \
--warmup_steps=6044 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/slv_latn_1000mb.txt \
--seed=43 \
--override_n_examples=386822 \
--output_dir=models/1000mb/slv_latn_1000mb
cp tokenizers/monolingual/slv_latn_100mb/* models/1000mb/slv_latn_1000mb

# som_latn
if test -f models/1000mb/som_latn_1000mb/pytorch_model.bin; then
echo "Model already found: som_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/som_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/som_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=4618 --save_steps=4618 \
--max_steps=92362 \
--warmup_steps=9236 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/som_latn_1000mb.txt \
--seed=43 \
--override_n_examples=591119 \
--output_dir=models/1000mb/som_latn_1000mb
cp tokenizers/monolingual/som_latn_100mb/* models/1000mb/som_latn_1000mb

# spa_latn
if test -f models/1000mb/spa_latn_1000mb/pytorch_model.bin; then
echo "Model already found: spa_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/spa_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/spa_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3384 --save_steps=3384 \
--max_steps=67685 \
--warmup_steps=6768 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/spa_latn_1000mb.txt \
--seed=43 \
--override_n_examples=433185 \
--output_dir=models/1000mb/spa_latn_1000mb
cp tokenizers/monolingual/spa_latn_100mb/* models/1000mb/spa_latn_1000mb

# sqi_latn
if test -f models/1000mb/sqi_latn_1000mb/pytorch_model.bin; then
echo "Model already found: sqi_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/sqi_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/sqi_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=4191 --save_steps=4191 \
--max_steps=83820 \
--warmup_steps=8382 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/sqi_latn_1000mb.txt \
--seed=43 \
--override_n_examples=536454 \
--output_dir=models/1000mb/sqi_latn_1000mb
cp tokenizers/monolingual/sqi_latn_100mb/* models/1000mb/sqi_latn_1000mb

# srp_cyrl
if test -f models/1000mb/srp_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: srp_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srp_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srp_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2814 --save_steps=2814 \
--max_steps=56281 \
--warmup_steps=5628 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srp_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=360202 \
--output_dir=models/1000mb/srp_cyrl_1000mb
cp tokenizers/monolingual/srp_cyrl_100mb/* models/1000mb/srp_cyrl_1000mb

# srp_latn
if test -f models/1000mb/srp_latn_1000mb/pytorch_model.bin; then
echo "Model already found: srp_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/srp_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/srp_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3165 --save_steps=3165 \
--max_steps=63318 \
--warmup_steps=6331 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/srp_latn_1000mb.txt \
--seed=43 \
--override_n_examples=405239 \
--output_dir=models/1000mb/srp_latn_1000mb
cp tokenizers/monolingual/srp_latn_100mb/* models/1000mb/srp_latn_1000mb

# swa_latn
if test -f models/1000mb/swa_latn_1000mb/pytorch_model.bin; then
echo "Model already found: swa_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/swa_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/swa_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3967 --save_steps=3967 \
--max_steps=79355 \
--warmup_steps=7935 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/swa_latn_1000mb.txt \
--seed=43 \
--override_n_examples=507877 \
--output_dir=models/1000mb/swa_latn_1000mb
cp tokenizers/monolingual/swa_latn_100mb/* models/1000mb/swa_latn_1000mb

# swe_latn
if test -f models/1000mb/swe_latn_1000mb/pytorch_model.bin; then
echo "Model already found: swe_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/swe_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/swe_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3148 --save_steps=3148 \
--max_steps=62975 \
--warmup_steps=6297 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/swe_latn_1000mb.txt \
--seed=43 \
--override_n_examples=403046 \
--output_dir=models/1000mb/swe_latn_1000mb
cp tokenizers/monolingual/swe_latn_100mb/* models/1000mb/swe_latn_1000mb

# tam_taml
if test -f models/1000mb/tam_taml_1000mb/pytorch_model.bin; then
echo "Model already found: tam_taml_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tam_taml_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tam_taml_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3059 --save_steps=3059 \
--max_steps=61194 \
--warmup_steps=6119 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tam_taml_1000mb.txt \
--seed=43 \
--override_n_examples=391647 \
--output_dir=models/1000mb/tam_taml_1000mb
cp tokenizers/monolingual/tam_taml_100mb/* models/1000mb/tam_taml_1000mb

# tat_cyrl
if test -f models/1000mb/tat_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: tat_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tat_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tat_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3554 --save_steps=3554 \
--max_steps=71085 \
--warmup_steps=7108 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tat_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=454949 \
--output_dir=models/1000mb/tat_cyrl_1000mb
cp tokenizers/monolingual/tat_cyrl_100mb/* models/1000mb/tat_cyrl_1000mb

# tel_telu
if test -f models/1000mb/tel_telu_1000mb/pytorch_model.bin; then
echo "Model already found: tel_telu_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tel_telu_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tel_telu_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3194 --save_steps=3194 \
--max_steps=63893 \
--warmup_steps=6389 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tel_telu_1000mb.txt \
--seed=43 \
--override_n_examples=408917 \
--output_dir=models/1000mb/tel_telu_1000mb
cp tokenizers/monolingual/tel_telu_100mb/* models/1000mb/tel_telu_1000mb

# tgk_cyrl
if test -f models/1000mb/tgk_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: tgk_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tgk_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tgk_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3311 --save_steps=3311 \
--max_steps=66220 \
--warmup_steps=6622 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tgk_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=423809 \
--output_dir=models/1000mb/tgk_cyrl_1000mb
cp tokenizers/monolingual/tgk_cyrl_100mb/* models/1000mb/tgk_cyrl_1000mb

# tgl_latn
if test -f models/1000mb/tgl_latn_1000mb/pytorch_model.bin; then
echo "Model already found: tgl_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tgl_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tgl_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3744 --save_steps=3744 \
--max_steps=74881 \
--warmup_steps=7488 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tgl_latn_1000mb.txt \
--seed=43 \
--override_n_examples=479240 \
--output_dir=models/1000mb/tgl_latn_1000mb
cp tokenizers/monolingual/tgl_latn_100mb/* models/1000mb/tgl_latn_1000mb

# tha_thai
if test -f models/1000mb/tha_thai_1000mb/pytorch_model.bin; then
echo "Model already found: tha_thai_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tha_thai_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tha_thai_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3141 --save_steps=3141 \
--max_steps=62827 \
--warmup_steps=6282 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tha_thai_1000mb.txt \
--seed=43 \
--override_n_examples=402095 \
--output_dir=models/1000mb/tha_thai_1000mb
cp tokenizers/monolingual/tha_thai_100mb/* models/1000mb/tha_thai_1000mb

# tur_latn
if test -f models/1000mb/tur_latn_1000mb/pytorch_model.bin; then
echo "Model already found: tur_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/tur_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/tur_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=2851 --save_steps=2851 \
--max_steps=57021 \
--warmup_steps=5702 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/tur_latn_1000mb.txt \
--seed=43 \
--override_n_examples=364939 \
--output_dir=models/1000mb/tur_latn_1000mb
cp tokenizers/monolingual/tur_latn_100mb/* models/1000mb/tur_latn_1000mb

# ukr_cyrl
if test -f models/1000mb/ukr_cyrl_1000mb/pytorch_model.bin; then
echo "Model already found: ukr_cyrl_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/ukr_cyrl_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/ukr_cyrl_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3286 --save_steps=3286 \
--max_steps=65732 \
--warmup_steps=6573 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/ukr_cyrl_1000mb.txt \
--seed=43 \
--override_n_examples=420689 \
--output_dir=models/1000mb/ukr_cyrl_1000mb
cp tokenizers/monolingual/ukr_cyrl_100mb/* models/1000mb/ukr_cyrl_1000mb

# urd_arab
if test -f models/1000mb/urd_arab_1000mb/pytorch_model.bin; then
echo "Model already found: urd_arab_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/urd_arab_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/urd_arab_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3782 --save_steps=3782 \
--max_steps=75652 \
--warmup_steps=7565 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/urd_arab_1000mb.txt \
--seed=43 \
--override_n_examples=484179 \
--output_dir=models/1000mb/urd_arab_1000mb
cp tokenizers/monolingual/urd_arab_100mb/* models/1000mb/urd_arab_1000mb

# uzb_latn
if test -f models/1000mb/uzb_latn_1000mb/pytorch_model.bin; then
echo "Model already found: uzb_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/uzb_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/uzb_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3983 --save_steps=3983 \
--max_steps=79668 \
--warmup_steps=7966 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/uzb_latn_1000mb.txt \
--seed=43 \
--override_n_examples=509880 \
--output_dir=models/1000mb/uzb_latn_1000mb
cp tokenizers/monolingual/uzb_latn_100mb/* models/1000mb/uzb_latn_1000mb

# vie_latn
if test -f models/1000mb/vie_latn_1000mb/pytorch_model.bin; then
echo "Model already found: vie_latn_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/vie_latn_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/vie_latn_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=4002 --save_steps=4002 \
--max_steps=80049 \
--warmup_steps=8004 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/vie_latn_1000mb.txt \
--seed=43 \
--override_n_examples=512317 \
--output_dir=models/1000mb/vie_latn_1000mb
cp tokenizers/monolingual/vie_latn_100mb/* models/1000mb/vie_latn_1000mb

# zho_hans
if test -f models/1000mb/zho_hans_1000mb/pytorch_model.bin; then
echo "Model already found: zho_hans_1000mb."
fi
python3 word-acquisition-language-models/lm_code/run_transformer_language_modeling.py \
--tokenizer_name=tokenizers/monolingual/zho_hans_100mb \
--config_name="gpt_base_config.json" \
--do_train --train_iterable --eval_iterable \
--eval_data_file=tokenized_data_split/zho_hans_1000mb_eval2k.txt \
--per_device_train_batch_size=8 \
--gradient_accumulation_steps=8 \
--per_device_eval_batch_size=8 \
--evaluation_strategy=steps --save_strategy=steps \
--eval_steps=3146 --save_steps=3146 \
--max_steps=62928 \
--warmup_steps=6292 \
--learning_rate=0.0001 --adam_epsilon=1e-6 --weight_decay=0.01 \
--train_data_file=tokenized_data_split/zho_hans_1000mb.txt \
--seed=43 \
--override_n_examples=402743 \
--output_dir=models/1000mb/zho_hans_1000mb
cp tokenizers/monolingual/zho_hans_100mb/* models/1000mb/zho_hans_1000mb
