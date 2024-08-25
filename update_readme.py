"""
Before running this, log into goldfish Hugging Face account:
transformers-cli login

Update model readme files.

Requires:
constants.py (for LANG_SETS)
model_details.json


"""

import codecs
import json

from huggingface_hub import upload_file

from constants import LANG_SETS


# All macrolangs, including non-Goldfish.
ALL_MACROLANGS = {'aka': {'twi', 'aka', 'fat'}, 'alb': {'als', 'sqi', 'aae', 'aln', 'aat', 'alb'}, 'ara': {'acx', 'avl', 'apc', 'pga', 'abv', 'ara', 'aao', 'ayl', 'ayh', 'shu', 'apd', 'ayn', 'acy', 'afb', 'ary', 'ayp', 'arb', 'aec', 'arq', 'acw', 'acq', 'ars', 'adf', 'arz', 'abh', 'acm', 'aeb', 'auz', 'ssh'}, 'aym': {'aym', 'ayc', 'ayr'}, 'aze': {'azb', 'aze', 'azj'}, 'bal': {'bgn', 'bal', 'bcc', 'bgp'}, 'bik': {'cts', 'rbl', 'ubl', 'bcl', 'fbl', 'bik', 'lbl', 'bln', 'bto'}, 'bnc': {'bnc', 'vbk', 'rbk', 'obk', 'lbk', 'ebk'}, 'bua': {'bxu', 'bxm', 'bxr', 'bua'}, 'chi': {'zho', 'cpx', 'cdo', 'czh', 'mnp', 'nan', 'wuu', 'yue', 'cmn', 'cjy', 'czo', 'cnp', 'csp', 'hsn', 'gan', 'lzh', 'chi', 'hak'}, 'chm': {'mhr', 'mrj', 'chm'}, 'cre': {'crm', 'crk', 'cre', 'csw', 'cwd', 'crl', 'crj'}, 'del': {'del', 'unm', 'umu'}, 'den': {'scs', 'den', 'xsl'}, 'din': {'diw', 'din', 'dik', 'dks', 'dib', 'dip'}, 'doi': {'doi', 'xnr', 'dgo'}, 'est': {'vro', 'ekk', 'est'}, 'fas': {'fas', 'pes', 'prs'}, 'ful': {'fub', 'fuq', 'fuf', 'fuc', 'fui', 'ffm', 'fue', 'ful', 'fuv', 'fuh'}, 'gba': {'bdt', 'gba', 'gmm', 'gso', 'gbp', 'gya', 'gbq'}, 'gon': {'esg', 'gon', 'wsg', 'gno'}, 'grb': {'gry', 'gbo', 'grv', 'gec', 'grb', 'grj'}, 'grn': {'gug', 'gun', 'gnw', 'gui', 'nhd', 'grn'}, 'hai': {'hai', 'hax', 'hdn'}, 'hbs': {'srp', 'bos', 'cnr', 'hrv', 'hbs'}, 'hmn': {'hma', 'hmm', 'hmj', 'hea', 'hmw', 'hmq', 'hmp', 'mww', 'hms', 'muq', 'hmg', 'hmz', 'hme', 'hnj', 'huj', 'hmi', 'hml', 'hmy', 'hmn', 'hmd', 'hmh', 'mmr', 'hmc', 'cqd', 'hrm', 'sfm'}, 'iku': {'ikt', 'iku', 'ike'}, 'ipk': {'esi', 'ipk', 'esk'}, 'jrb': {'aeb', 'jye', 'aju', 'yud', 'jrb', 'yhd'}, 'kau': {'knc', 'kby', 'kau', 'krt'}, 'kln': {'enb', 'tuy', 'spy', 'eyo', 'tec', 'niq', 'kln', 'sgc', 'oki', 'pko'}, 'kok': {'gom', 'kok', 'knn'}, 'kom': {'kpv', 'kom', 'koi'}, 'kon': {'ldi', 'kng', 'kon', 'kwy'}, 'kpe': {'kpe', 'gkp', 'xpe'}, 'kur': {'kur', 'ckb', 'sdh', 'kmr'}, 'lah': {'pnb', 'jat', 'phr', 'hnd', 'hno', 'xhe', 'lah', 'skr'}, 'lav': {'lvs', 'lav', 'ltg'}, 'luy': {'ida', 'nyd', 'rag', 'lto', 'lri', 'lwg', 'luy', 'bxk', 'lkb', 'lrm', 'lko', 'lts', 'lsm', 'lks', 'nle'}, 'man': {'mku', 'man', 'mwk', 'msc', 'mlq', 'mnk', 'emk'}, 'may': {'max', 'dup', 'ind', 'pse', 'coa', 'lcf', 'min', 'xmm', 'kxd', 'bjn', 'ors', 'liw', 'zsm', 'mfb', 'kvr', 'jak', 'vkt', 'pel', 'urk', 'orn', 'msi', 'bve', 'mfa', 'jax', 'tmw', 'lce', 'may', 'kvb', 'mqg', 'zlm', 'vkk', 'zmi', 'bvu', 'meo', 'mui', 'msa', 'btj', 'hji'}, 'mlg': {'tkg', 'msh', 'xmw', 'bmm', 'bhr', 'mlg', 'bzc', 'plt', 'xmv', 'tdx', 'skg', 'txy'}, 'mon': {'mvf', 'khk', 'mon'}, 'msa': {'max', 'dup', 'ind', 'pse', 'coa', 'lcf', 'min', 'xmm', 'kxd', 'bjn', 'ors', 'liw', 'zsm', 'mfb', 'kvr', 'jak', 'vkt', 'pel', 'urk', 'orn', 'msi', 'bve', 'mfa', 'jax', 'tmw', 'lce', 'kvb', 'mqg', 'zlm', 'vkk', 'zmi', 'bvu', 'meo', 'msa', 'mui', 'btj', 'hji'}, 'mwr': {'mve', 'swv', 'rwr', 'dhd', 'mtr', 'wry', 'mwr'}, 'nep': {'dty', 'nep', 'npi'}, 'nor': {'nno', 'nob', 'nor'}, 'oji': {'oji', 'ojw', 'ciw', 'ojc', 'ojs', 'ojg', 'otw', 'ojb'}, 'ori': {'spv', 'ori', 'ory'}, 'orm': {'hae', 'orc', 'gax', 'gaz', 'orm'}, 'per': {'fas', 'pes', 'prs', 'per'}, 'pus': {'pus', 'pbu', 'pbt', 'pst'}, 'que': {'quw', 'qxo', 'qvw', 'qub', 'qul', 'qxu', 'quk', 'qvm', 'qvi', 'qvj', 'qvh', 'qve', 'qwc', 'qws', 'qug', 'qwh', 'qxc', 'qxl', 'qxt', 'qvs', 'qxh', 'qvn', 'qxr', 'quy', 'quf', 'quh', 'qxp', 'qvc', 'quz', 'qvo', 'qud', 'qvl', 'qvz', 'qup', 'qva', 'qxn', 'qwa', 'qux', 'qxw', 'qxa', 'qus', 'que', 'qur', 'qvp'}, 'raj': {'mup', 'raj', 'hoj', 'bgq', 'gju', 'wbr', 'gda'}, 'rom': {'rmy', 'rmo', 'rmf', 'rom', 'rmw', 'rmc', 'rmn', 'rml'}, 'san': {'san', 'vsn', 'cls'}, 'sqi': {'als', 'sqi', 'aae', 'aln', 'aat'}, 'srd': {'sro', 'sdn', 'sdc', 'src', 'srd'}, 'swa': {'swa', 'swh', 'swc'}, 'syr': {'aii', 'syr', 'cld'}, 'tmh': {'taq', 'ttq', 'tmh', 'thv', 'thz'}, 'uzb': {'uzs', 'uzn', 'uzb'}, 'yid': {'yih', 'ydd', 'yid'}, 'zap': {'zpz', 'ztn', 'zts', 'ztx', 'zae', 'zpn', 'zpr', 'zao', 'zpu', 'zty', 'zpe', 'zpf', 'zat', 'zpa', 'zpw', 'ztp', 'zte', 'zpd', 'zpm', 'zpg', 'ztg', 'zpc', 'zam', 'zaf', 'zcd', 'zca', 'zac', 'zps', 'zai', 'zpi', 'zpb', 'zpq', 'zaa', 'ztm', 'zap', 'zpk', 'zsr', 'zaq', 'zav', 'ztq', 'zar', 'zpy', 'ztt', 'zaw', 'zas', 'zpl', 'zpo', 'zpx', 'zph', 'zpt', 'zpp', 'zax', 'zab', 'zpv', 'zad', 'ztl', 'zpj', 'zoo', 'ztu'}, 'zha': {'zeh', 'zln', 'zyg', 'zgb', 'zgm', 'zhd', 'zha', 'zch', 'zlj', 'zlq', 'zyn', 'zzj', 'zhn', 'zyj', 'zqe', 'zgn', 'zyb'}, 'zho': {'zho', 'cpx', 'cdo', 'czh', 'mnp', 'nan', 'wuu', 'yue', 'cmn', 'cjy', 'czo', 'cnp', 'csp', 'hsn', 'gan', 'lzh', 'hak'}, 'zza': {'kiu', 'diq', 'zza'}}


INPATH = 'model_details.json'
with codecs.open(INPATH, 'rb', encoding='utf-8') as f_in:
    MODEL_DICTS = json.load(f_in)


# Readme file:
README_TEMPLATE = r"""
---
license: apache-2.0
language:
[[lang_tags_str]]
[[hf_dataset_tags_str]]
library_name: transformers
pipeline_tag: text-generation
tags:
- goldfish
[[arxiv_paper_tag_str]]
---

# [[model_name]]

Goldfish is a suite of monolingual language models trained for 350 languages.
This model is the <b>[[language_name]]</b> ([[language_script]] script) model trained on [[dataset_size_str]], after accounting for an estimated byte premium of [[byte_premium]]; content-matched text in [[language_name]] takes on average [[byte_premium]]x as many UTF-8 bytes to encode as English.
The Goldfish models are trained primarily for comparability across languages and for low-resource languages; Goldfish performance for high-resource languages is not designed to be comparable with modern large language models (LLMs).
[[optional_script_note]]
[[language_code_note]]

All training and hyperparameter details are in our paper, [Goldfish: Monolingual Language Models for 350 Languages (Chang et al., 2024)]([[paper_link]]).

Training code and sample usage: https://github.com/tylerachang/goldfish

Sample usage also in this Google Colab: [link](https://colab.research.google.com/drive/1rHFpnQsyXJ32ONwCosWZ7frjOYjbGCXG?usp=sharing)

## Model details:

To access all Goldfish model details programmatically, see https://github.com/tylerachang/goldfish/blob/main/model_details.json.
All models are trained with a [CLS] (same as [BOS]) token prepended, and a [SEP] (same as [EOS]) token separating sequences.
For best results, make sure that [CLS] is prepended to your input sequence (see sample usage linked above)!
Details for this model specifically:

* Architecture: [[model_architecture]]
* Parameters: [[model_parameters]]
* Maximum sequence length: [[model_sequence_length]] tokens
* Training text data (raw): [[dataset_raw_mb]]MB
* Training text data (byte premium scaled): [[dataset_scaled_mb]]5MB
* Training tokens: [[dataset_tokens]] (x[[train_epochs]] epochs)
* Vocabulary size: [[tokenizer_vocab_size]]
* Compute cost: [[train_compute_flops]] FLOPs or ~[[train_compute_hours]] NVIDIA A6000 GPU hours

Training datasets (percentages prior to deduplication):
[[dataset_readme_str]]

## Citation

If you use this model, please cite:

```
@article{chang-etal-2024-goldfish,
  title={Goldfish: Monolingual Language Models for 350 Languages},
  author={Chang, Tyler A. and Arnett, Catherine and Tu, Zhuowen and Bergen, Benjamin K.},
  journal={Preprint},
  year={2024},
  url={https://www.arxiv.org/abs/2408.10441},
}
```
"""


# ARXIV_PAPER_TAG_STR = '- arxiv:<PAPER ID>'
ARXIV_PAPER_TAG_STR = '- arxiv:2408.10441'
PAPER_LINK = 'https://www.arxiv.org/abs/2408.10441'


def generate_readme(lang, dataset_size):
    model_dict = MODEL_DICTS[f'{lang}_{dataset_size}']
    replacements = dict()
    # Tags.
    replacements['arxiv_paper_tag_str'] = ARXIV_PAPER_TAG_STR
    # Language code; include macrolanguage if possible.
    langcodes = set([macro for macro, individuals in ALL_MACROLANGS.items() if lang[:3] in individuals])
    langcodes.add(lang[:3])  # And the language itself.
    if 'nep' in langcodes: langcodes.add('npi')  # Custom inclusion.
    if 'ori' in langcodes: langcodes.add('ory')  # Custom inclusion.
    if 'swa' in langcodes: langcodes.add('swh')  # Custom inclusion.
    if lang == 'zho_hans': langcodes.add('cmn')  # Custom inclusion.
    replacements['lang_tags_str'] = '\n'.join([f'- {x}' for x in langcodes])
    # Other tags.
    replacements['hf_dataset_tags_str'] = '\n'.join([f'- {x}' for x in model_dict['dataset_hugging_face']])
    if replacements['hf_dataset_tags_str'].strip():
        replacements['hf_dataset_tags_str'] = 'datasets:\n' + replacements['hf_dataset_tags_str']
    # Paper link
    replacements['paper_link'] = PAPER_LINK

    # Model intro.
    replacements['model_name'] = lang + '_' + dataset_size
    replacements['language_name'] = model_dict['language_name']
    replacements['language_script'] = model_dict['language_script']
    if dataset_size == 'full':
        mb_str = '%d' % model_dict['dataset_scaled_mb']
        replacements['dataset_size_str'] = f'{mb_str}MB of data (all our data in the language)'
    else:
        replacements['dataset_size_str'] = f'{dataset_size.upper()} of data'
    replacements['byte_premium'] = '%.2f' % model_dict['language_byte_premium']

    # Optional script note.
    same_lang = [l for l in LANG_SETS['5mb'] if l[:3] == lang[:3]]
    same_lang.remove(lang)
    if len(same_lang) == 0:
        optional_script_note = ''
    else:
        same_lang_str = ', '.join(same_lang)
        optional_script_note = f'\nNote: This language is available in Goldfish with other scripts (writing systems). See: {same_lang_str}.\n'
    replacements['optional_script_note'] = optional_script_note

    # Language code note.
    def langs_to_str(langs_dict):
        # Construct a natural language string based on the language codes mapped
        # to language names.
        if len(langs_dict) <= 2:
            return ' and '.join([f'{code} ({name})' for code, name in langs_dict.items()])
        else:
            construct_str = ''
            for i, t in enumerate(langs_dict.items()):
                code, name = t
                if i == len(langs_dict)-1:
                    construct_str += f' and {code} ({name})'
                construct_str += f'{code} ({name}), '
            return construct_str
    language_iso15924 = lang[4:]
    if model_dict['language_code_type'] == 'collective':
        replacements['language_code_note'] = f'Note: {lang} is a [collective language](https://iso639-3.sil.org/code_tables/639/data) code, so it may comprise several distinct languages.'
    elif model_dict['language_code_type'] == 'macrolanguage':
        replacements['language_code_note'] = f'Note: {lang} is a [macrolanguage](https://iso639-3.sil.org/code_tables/639/data) code.'
        individuals = dict()
        for model_dict2 in MODEL_DICTS.values():
            # Individual language with same script.
            if (model_dict2['language_iso6393'] in model_dict['language_code_individuals']) and (model_dict2['language_iso15924'] == model_dict['language_iso15924']):
                individual = model_dict2['language_iso6393'] + '_' + model_dict2['language_iso15924']
                individuals[individual] = model_dict2['language_name']
            individual_langs_str = langs_to_str(individuals)
        if len(individuals) == 0:
            replacements['language_code_note'] = replacements['language_code_note'] + f' None of its contained individual languages are included in Goldfish (for script {language_iso15924}).'
        elif len(individuals) == 1:
            replacements['language_code_note'] = replacements['language_code_note'] + f' Individual language code {individual_langs_str} is included in Goldfish, although with less data.'
        else:
            replacements['language_code_note'] = replacements['language_code_note'] + f' Individual language codes {individual_langs_str} are included in Goldfish, although with less data.'
    elif model_dict['language_code_type'] == 'individual':
        replacements['language_code_note'] = f'Note: {lang} is an [individual language](https://iso639-3.sil.org/code_tables/639/data) code.'
        if ('ara' in model_dict['language_code_macrolangs']) and (model_dict['language_iso15924'] == 'arab'):
            # Special handling for Arabic.
            if lang == 'arb_arab':
                replacements['language_code_note'] = replacements['language_code_note'] + ' However, this model may also be useful for other Arabic dialects with less data.'
            else:
                replacements['language_code_note'] = replacements['language_code_note'] + ' However, you may also want to consider the arb_arab (Standard Arabic) models which are trained on more data.'
        else:
            # Non-Arabic.
            macrolangs = dict()
            for model_dict2 in MODEL_DICTS.values():
                # Individual language with same script.
                if (model_dict2['language_iso6393'] in model_dict['language_code_macrolangs']) and (model_dict2['language_iso15924'] == model_dict['language_iso15924']):
                    macrolang = model_dict2['language_iso6393'] + '_' + model_dict2['language_iso15924']
                    macrolangs[macrolang] = model_dict2['language_name']
            macrolangs_str = langs_to_str(macrolangs)
            assert len(macrolangs) <= 1  # Not multiple macrolanguages for a language.
            if len(macrolangs) == 0:
                replacements['language_code_note'] = replacements['language_code_note'] + f' It is not contained in any macrolanguage codes contained in Goldfish (for script {language_iso15924}).'
            else:
                replacements['language_code_note'] = replacements['language_code_note'] + f' Macrolanguage code {macrolangs_str} is included in Goldfish. Consider using that model depending on your use case.'

    # Model details.
    replacements['model_architecture'] = model_dict['model_architecture']
    replacements['model_sequence_length'] = str(int(model_dict['model_sequence_length']))
    replacements['model_parameters'] = str(int(model_dict['model_parameters']))
    replacements['train_compute_flops'] = str(model_dict['train_compute_flops'])
    replacements['train_compute_hours'] = '%.1f' % model_dict['train_compute_hours']
    replacements['tokenizer_vocab_size'] = str(int(model_dict['tokenizer_vocab_size']))
    replacements['dataset_readme_str'] = model_dict['dataset_readme_str']
    replacements['train_epochs'] = str(int(model_dict['train_epochs']))
    replacements['dataset_raw_mb'] = '%.2f' % model_dict['dataset_raw_mb']
    replacements['dataset_scaled_mb'] = '%.2f' % model_dict['dataset_scaled_mb']
    replacements['dataset_tokens'] = str(int(model_dict['dataset_tokens']))

    # Generate readme.
    readme = README_TEMPLATE
    for k, v in replacements.items():
        readme = readme.replace(f'[[{k}]]', v)
    return readme


# Update README files!
tmp_path = 'tmp_readme.md'
for dataset_size in ['5mb', '10mb', '100mb', 'full', '1000mb']:
    langs = LANG_SETS['5mb'].difference(LANG_SETS['1000mb']) if dataset_size == 'full' else LANG_SETS[dataset_size]
    for lang in sorted(langs):
        model_name = f'{lang}_{dataset_size}'
        print(f'Model name: {model_name}')
        readme = generate_readme(lang, dataset_size)
        with codecs.open(tmp_path, 'wb', encoding='utf-8') as f_out:
            f_out.write(readme)
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo='README.md',
            repo_id=f'goldfish-models/{model_name}',
        )
