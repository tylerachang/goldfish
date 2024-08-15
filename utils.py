"""
Utility functions for Goldfish models.
"""

from constants import LANG_SETS


# Map ISO 639-3 to name for Goldfish languages.
LANG_NAMES = {'abk': 'Abkhazian', 'ace': 'Achinese', 'ady': 'Adyghe', 'afb': 'Gulf Arabic', 'afr': 'Afrikaans', 'aka': 'Akan', 'als': 'Tosk Albanian', 'alt': 'Southern Altai', 'amh': 'Amharic', 'ang': 'Old English', 'apc': 'Levantine Arabic', 'arb': 'Standard Arabic', 'arg': 'Aragonese', 'arz': 'Egyptian Arabic', 'asm': 'Assamese', 'ast': 'Asturian', 'ava': 'Avaric', 'aym': 'Aymara', 'ayr': 'Central Aymara', 'azb': 'South Azerbaijani', 'aze': 'Azerbaijani', 'azj': 'North Azerbaijani', 'bak': 'Bashkir', 'bam': 'Bambara', 'ban': 'Balinese', 'bar': 'Bavarian', 'bbc': 'Batak Toba', 'bcl': 'Central Bikol', 'bel': 'Belarusian', 'bem': 'Bemba', 'ben': 'Bengali', 'bew': 'Betawi', 'bgp': 'Eastern Balochi', 'bho': 'Bhojpuri', 'bik': 'Bikol', 'bjn': 'Banjar', 'bod': 'Tibetan', 'bos': 'Bosnian', 'bpy': 'Bishnupriya', 'bqc': 'Boko', 'bre': 'Breton', 'bsb': 'Brunei Bisaya', 'bua': 'Buriat', 'bug': 'Buginese', 'bul': 'Bulgarian', 'bxr': 'Russia Buriat', 'cak': 'Kaqchikel', 'cat': 'Catalan', 'ceb': 'Cebuano', 'ces': 'Czech', 'cfm': 'Falam Chin', 'che': 'Chechen', 'chm': 'Mari', 'chv': 'Chuvash', 'cjk': 'Chokwe', 'ckb': 'Central Kurdish', 'cnh': 'Hakha Chin', 'cor': 'Cornish', 'cos': 'Corsican', 'crh': 'Crimean Tatar', 'ctd': 'Tedim Chin', 'cym': 'Welsh', 'dan': 'Danish', 'dar': 'Dargwa', 'deu': 'German', 'dik': 'Southwestern Dinka', 'din': 'Dinka', 'diq': 'Dimli', 'div': 'Dhivehi', 'dov': 'Dombe', 'dyu': 'Dyula', 'dzo': 'Dzongkha', 'ekk': 'Standard Estonian', 'ell': 'Modern Greek', 'eng': 'English', 'epo': 'Esperanto', 'est': 'Estonian', 'eus': 'Basque', 'ewe': 'Ewe', 'fao': 'Faroese', 'fas': 'Persian', 'fij': 'Fijian', 'fil': 'Filipino', 'fin': 'Finnish', 'fon': 'Fon', 'fra': 'French', 'frr': 'Northern Frisian', 'fry': 'Western Frisian', 'ful': 'Fulah', 'fur': 'Friulian', 'fuv': 'Nigerian Fulfulde', 'gaz': 'West Central Oromo', 'gla': 'Scottish Gaelic', 'gle': 'Irish', 'glg': 'Galician', 'glk': 'Gilaki', 'glv': 'Manx', 'gom': 'Goan Konkani', 'grc': 'Ancient Greek', 'grn': 'Guarani', 'gsw': 'Swiss German', 'guj': 'Gujarati', 'hat': 'Haitian', 'hau': 'Hausa', 'haw': 'Hawaiian', 'heb': 'Hebrew', 'hif': 'Fiji Hindi', 'hil': 'Hiligaynon', 'hin': 'Hindi', 'hmn': 'Hmong', 'hne': 'Chhattisgarhi', 'hrv': 'Croatian', 'hsb': 'Upper Sorbian', 'hun': 'Hungarian', 'hye': 'Armenian', 'iba': 'Iban', 'ibo': 'Igbo', 'ido': 'Ido', 'iku': 'Inuktitut', 'ilo': 'Iloko', 'ina': 'Interlingua', 'ind': 'Indonesian', 'inh': 'Ingush', 'isl': 'Icelandic', 'iso': 'Isoko', 'ita': 'Italian', 'jav': 'Javanese', 'jpn': 'Japanese', 'kaa': 'Kara-Kalpak', 'kab': 'Kabyle', 'kac': 'Kachin', 'kal': 'Kalaallisut', 'kan': 'Kannada', 'kas': 'Kashmiri', 'kat': 'Georgian', 'kaz': 'Kazakh', 'kbd': 'Kabardian', 'kbp': 'Kabiye', 'kea': 'Kabuverdianu', 'kha': 'Khasi', 'khk': 'Halh Mongolian', 'khm': 'Central Khmer', 'kik': 'Kikuyu', 'kin': 'Kinyarwanda', 'kir': 'Kirghiz', 'kjh': 'Khakas', 'kmb': 'Kimbundu', 'kmr': 'Northern Kurdish', 'knc': 'Central Kanuri', 'kom': 'Komi', 'kon': 'Kongo', 'kor': 'Korean', 'kpv': 'Komi-Zyrian', 'krc': 'Karachay-Balkar', 'kum': 'Kumyk', 'kur': 'Kurdish', 'lao': 'Lao', 'lat': 'Latin', 'lav': 'Latvian', 'lbe': 'Lak', 'lez': 'Lezghian', 'lfn': 'Lingua Franca Nova', 'lij': 'Ligurian', 'lim': 'Limburgan', 'lin': 'Lingala', 'lit': 'Lithuanian', 'lmo': 'Lombard', 'ltg': 'Latgalian', 'ltz': 'Luxembourgish', 'lua': 'Luba-Lulua', 'lub': 'Luba-Katanga', 'lug': 'Ganda', 'luo': 'Luo', 'lus': 'Lushai', 'lvs': 'Standard Latvian', 'lzh': 'Literary Chinese', 'mad': 'Madurese', 'mag': 'Magahi', 'mai': 'Maithili', 'mal': 'Malayalam', 'mam': 'Mam', 'mar': 'Marathi', 'mdf': 'Moksha', 'meo': 'Kedah Malay', 'mgh': 'Makhuwa-Meetto', 'mhr': 'Eastern Mari', 'min': 'Minangkabau', 'mkd': 'Macedonian', 'mkw': 'Kituba', 'mlg': 'Malagasy', 'mlt': 'Maltese', 'mon': 'Mongolian', 'mos': 'Mossi', 'mri': 'Maori', 'mrj': 'Western Mari', 'msa': 'Malay', 'mwl': 'Mirandese', 'mya': 'Burmese', 'myv': 'Erzya', 'nan': 'Min Nan Chinese', 'nap': 'Neapolitan', 'nde': 'North Ndebele', 'nds': 'Low German', 'nep': 'Nepali', 'new': 'Newari', 'ngu': 'Guerrero Nahuatl', 'nhe': 'Eastern Huasteca Nahuatl', 'nld': 'Dutch', 'nnb': 'Nande', 'nno': 'Norwegian Nynorsk', 'nob': 'Norwegian Bokmal', 'nor': 'Norwegian', 'nso': 'Pedi', 'nya': 'Nyanja', 'nzi': 'Nzima', 'oci': 'Occitan', 'ori': 'Odia', 'orm': 'Oromo', 'oss': 'Ossetian', 'otq': 'Queretaro Otomi', 'pag': 'Pangasinan', 'pam': 'Pampanga', 'pan': 'Panjabi', 'pap': 'Papiamento', 'pbt': 'Southern Pashto', 'pck': 'Paite Chin', 'pcm': 'Nigerian Pidgin', 'pes': 'Iranian Persian', 'plt': 'Plateau Malagasy', 'pms': 'Piemontese', 'pnb': 'Western Panjabi', 'pol': 'Polish', 'pon': 'Pohnpeian', 'por': 'Portuguese', 'prs': 'Dari', 'pus': 'Pushto', 'que': 'Quechua', 'quy': 'Ayacucho Quechua', 'quz': 'Cusco Quechua', 'rmc': 'Carpathian Romani', 'roh': 'Romansh', 'ron': 'Romanian', 'rue': 'Rusyn', 'run': 'Rundi', 'rus': 'Russian', 'sag': 'Sango', 'sah': 'Yakut', 'san': 'Sanskrit', 'sat': 'Santali', 'scn': 'Sicilian', 'sco': 'Scots', 'shn': 'Shan', 'sin': 'Sinhala', 'slk': 'Slovak', 'slv': 'Slovenian', 'sme': 'Northern Sami', 'smo': 'Samoan', 'sna': 'Shona', 'snd': 'Sindhi', 'som': 'Somali', 'sot': 'Southern Sotho', 'spa': 'Spanish', 'sqi': 'Albanian', 'srd': 'Sardinian', 'srn': 'Sranan Tongo', 'srp': 'Serbian', 'ssw': 'Swati', 'sun': 'Sundanese', 'swa': 'Swahili', 'swe': 'Swedish', 'syr': 'Syriac', 'szl': 'Silesian', 'tam': 'Tamil', 'tat': 'Tatar', 'tbz': 'Ditammari', 'tcy': 'Tulu', 'tdx': 'Tandroy-Mahafaly Malagasy', 'tel': 'Telugu', 'tet': 'Tetum', 'tgk': 'Tajik', 'tgl': 'Tagalog', 'tha': 'Thai', 'tir': 'Tigrinya', 'tiv': 'Tiv', 'tlh': 'Klingon', 'ton': 'Tonga', 'tpi': 'Tok Pisin', 'tsn': 'Tswana', 'tso': 'Tsonga', 'tuk': 'Turkmen', 'tum': 'Tumbuka', 'tur': 'Turkish', 'twi': 'Twi', 'tyv': 'Tuvinian', 'tzo': 'Tzotzil', 'udm': 'Udmurt', 'uig': 'Uighur', 'ukr': 'Ukrainian', 'umb': 'Umbundu', 'urd': 'Urdu', 'uzb': 'Uzbek', 'uzn': 'Northern Uzbek', 'vec': 'Venetian', 'ven': 'Venda', 'vep': 'Veps', 'vie': 'Vietnamese', 'vls': 'Vlaams', 'vol': 'Volapuk', 'war': 'Waray', 'wln': 'Walloon', 'wol': 'Wolof', 'wuu': 'Wu Chinese', 'xal': 'Kalmyk', 'xho': 'Xhosa', 'xmf': 'Mingrelian', 'ydd': 'Eastern Yiddish', 'yid': 'Yiddish', 'yor': 'Yoruba', 'yua': 'Yucateco', 'yue': 'Yue Chinese', 'zap': 'Zapotec', 'zho': 'Chinese', 'zsm': 'Standard Malay', 'zul': 'Zulu', 'zza': 'Zaza'}

# Map ISO 639-3 to other possible codes for Goldfish languages.
LANG_CODES = {'abk': {'ab', 'abk'}, 'ace': {'ace'}, 'ady': {'ady'}, 'afb': {'afb'}, 'afr': {'afr', 'af'}, 'aka': {'ak', 'aka'}, 'als': {'als'}, 'alt': {'alt'}, 'amh': {'amh', 'am'}, 'ang': {'ang'}, 'apc': {'apc', 'ajp'}, 'arb': {'arb'}, 'arg': {'arg', 'an'}, 'arz': {'arz'}, 'asm': {'as', 'asm'}, 'ast': {'ast'}, 'ava': {'ava', 'av'}, 'aym': {'aym', 'ay'}, 'ayr': {'ayr'}, 'azb': {'azb'}, 'aze': {'aze', 'az'}, 'azj': {'azj'}, 'bak': {'bak', 'ba'}, 'bam': {'bm', 'bam'}, 'ban': {'ban'}, 'bar': {'bar'}, 'bbc': {'bbc'}, 'bcl': {'bcl'}, 'bel': {'be', 'bel'}, 'bem': {'bem'}, 'ben': {'bn', 'ben'}, 'bew': {'bew'}, 'bgp': {'bgp'}, 'bho': {'bho'}, 'bik': {'bik'}, 'bjn': {'bjn'}, 'bod': {'bod', 'tib', 'bo'}, 'bos': {'bos', 'bs'}, 'bpy': {'bpy'}, 'bqc': {'bqc'}, 'bre': {'bre', 'br'}, 'bsb': {'bsb'}, 'bua': {'bua'}, 'bug': {'bug'}, 'bul': {'bg', 'bul'}, 'bxr': {'bxr'}, 'cak': {'cak'}, 'cat': {'cat', 'ca'}, 'ceb': {'ceb'}, 'ces': {'cs', 'ces', 'cze'}, 'cfm': {'cfm'}, 'che': {'che', 'ce'}, 'chm': {'chm'}, 'chv': {'chv', 'cv'}, 'cjk': {'cjk'}, 'ckb': {'ckb'}, 'cnh': {'cnh'}, 'cor': {'kw', 'cor'}, 'cos': {'cos', 'co'}, 'crh': {'crh'}, 'ctd': {'ctd'}, 'cym': {'wel', 'cym', 'cy'}, 'dan': {'dan', 'da'}, 'dar': {'dar'}, 'deu': {'de', 'deu', 'ger'}, 'dik': {'dik'}, 'din': {'din'}, 'diq': {'diq'}, 'div': {'dv', 'div'}, 'dov': {'dov'}, 'dyu': {'dyu'}, 'dzo': {'dz', 'dzo'}, 'ekk': {'ekk'}, 'ell': {'gre', 'ell', 'el'}, 'eng': {'en', 'eng'}, 'epo': {'eo', 'epo'}, 'est': {'est', 'et'}, 'eus': {'baq', 'eu', 'eus'}, 'ewe': {'ewe', 'ee'}, 'fao': {'fao', 'fo'}, 'fas': {'fa', 'per', 'fas'}, 'fij': {'fij', 'fj'}, 'fil': {'fil'}, 'fin': {'fin', 'fi'}, 'fon': {'fon'}, 'fra': {'fre', 'fra', 'fr'}, 'frr': {'frr'}, 'fry': {'fy', 'fry'}, 'ful': {'ful', 'ff'}, 'fur': {'fur'}, 'fuv': {'fuv'}, 'gaz': {'gaz'}, 'gla': {'gd', 'gla'}, 'gle': {'ga', 'gle'}, 'glg': {'gl', 'glg'}, 'glk': {'glk'}, 'glv': {'gv', 'glv'}, 'gom': {'gom'}, 'grc': {'grc'}, 'grn': {'gn', 'grn'}, 'gsw': {'gsw'}, 'guj': {'guj', 'gu'}, 'hat': {'hat', 'ht'}, 'hau': {'ha', 'hau'}, 'haw': {'haw'}, 'heb': {'heb', 'he'}, 'hif': {'hif'}, 'hil': {'hil'}, 'hin': {'hi', 'hin'}, 'hmn': {'hmn'}, 'hne': {'hne'}, 'hrv': {'hrv', 'hr'}, 'hsb': {'hsb'}, 'hun': {'hun', 'hu'}, 'hye': {'arm', 'hye', 'hy'}, 'iba': {'iba'}, 'ibo': {'ibo', 'ig'}, 'ido': {'ido', 'io'}, 'iku': {'iu', 'iku'}, 'ilo': {'ilo'}, 'ina': {'ina', 'ia'}, 'ind': {'id', 'ind'}, 'inh': {'inh'}, 'isl': {'ice', 'is', 'isl'}, 'iso': {'iso'}, 'ita': {'it', 'ita'}, 'jav': {'jv', 'jav'}, 'jpn': {'ja', 'jpn'}, 'kaa': {'kaa'}, 'kab': {'kab'}, 'kac': {'kac'}, 'kal': {'kal', 'kl'}, 'kan': {'kan', 'kn'}, 'kas': {'ks', 'kas'}, 'kat': {'kat', 'geo', 'ka'}, 'kaz': {'kaz', 'kk'}, 'kbd': {'kbd'}, 'kbp': {'kbp'}, 'kea': {'kea'}, 'kha': {'kha'}, 'khk': {'khk'}, 'khm': {'khm', 'km'}, 'kik': {'ki', 'kik'}, 'kin': {'kin', 'rw'}, 'kir': {'ky', 'kir'}, 'kjh': {'kjh'}, 'kmb': {'kmb'}, 'kmr': {'kmr'}, 'knc': {'knc'}, 'kom': {'kom', 'kv'}, 'kon': {'kg', 'kon'}, 'kor': {'ko', 'kor'}, 'kpv': {'kpv'}, 'krc': {'krc'}, 'kum': {'kum'}, 'kur': {'kur', 'ku'}, 'lao': {'lao', 'lo'}, 'lat': {'lat', 'la'}, 'lav': {'lav', 'lv'}, 'lbe': {'lbe'}, 'lez': {'lez'}, 'lfn': {'lfn'}, 'lij': {'lij'}, 'lim': {'li', 'lim'}, 'lin': {'ln', 'lin'}, 'lit': {'lit', 'lt'}, 'lmo': {'lmo'}, 'ltg': {'ltg'}, 'ltz': {'ltz', 'lb'}, 'lua': {'lua'}, 'lub': {'lu', 'lub'}, 'lug': {'lug', 'lg'}, 'luo': {'luo'}, 'lus': {'lus'}, 'lvs': {'lvs'}, 'lzh': {'lzh'}, 'mad': {'mad'}, 'mag': {'mag'}, 'mai': {'mai'}, 'mal': {'mal', 'ml'}, 'mam': {'mam'}, 'mar': {'mar', 'mr'}, 'mdf': {'mdf'}, 'meo': {'meo'}, 'mgh': {'mgh'}, 'mhr': {'mhr'}, 'min': {'min'}, 'mkd': {'mkd', 'mk', 'mac'}, 'mkw': {'mkw'}, 'mlg': {'mg', 'mlg'}, 'mlt': {'mt', 'mlt'}, 'mon': {'mon', 'mn'}, 'mos': {'mos'}, 'mri': {'mri', 'mao', 'mi'}, 'mrj': {'mrj'}, 'msa': {'ms', 'may', 'msa'}, 'mwl': {'mwl'}, 'mya': {'my', 'bur', 'mya'}, 'myv': {'myv'}, 'nan': {'nan'}, 'nap': {'nap'}, 'nde': {'nd', 'nde'}, 'nds': {'nds'}, 'nep': {'nep', 'ne'}, 'new': {'new'}, 'ngu': {'ngu'}, 'nhe': {'nhe'}, 'nld': {'nl', 'nld', 'dut'}, 'nnb': {'nnb'}, 'nno': {'nn', 'nno'}, 'nob': {'nb', 'nob'}, 'nor': {'no', 'nor'}, 'nso': {'nso'}, 'nya': {'ny', 'nya'}, 'nzi': {'nzi'}, 'oci': {'oci', 'oc'}, 'ori': {'ori', 'or'}, 'orm': {'orm', 'om'}, 'oss': {'oss', 'os'}, 'otq': {'otq'}, 'pag': {'pag'}, 'pam': {'pam'}, 'pan': {'pan', 'pa'}, 'pap': {'pap'}, 'pbt': {'pbt'}, 'pck': {'pck'}, 'pcm': {'pcm'}, 'pes': {'pes'}, 'plt': {'plt'}, 'pms': {'pms'}, 'pnb': {'pnb'}, 'pol': {'pl', 'pol'}, 'pon': {'pon'}, 'por': {'por', 'pt'}, 'prs': {'prs'}, 'pus': {'ps', 'pus'}, 'que': {'que', 'qu'}, 'quy': {'quy'}, 'quz': {'quz'}, 'rmc': {'rmc'}, 'roh': {'roh', 'rm'}, 'ron': {'ro', 'ron', 'rum'}, 'rue': {'rue'}, 'run': {'run', 'rn'}, 'rus': {'ru', 'rus'}, 'sag': {'sag', 'sg'}, 'sah': {'sah'}, 'san': {'sa', 'san'}, 'sat': {'sat'}, 'scn': {'scn'}, 'sco': {'sco'}, 'shn': {'shn'}, 'sin': {'si', 'sin'}, 'slk': {'slk', 'sk', 'slo'}, 'slv': {'sl', 'slv'}, 'sme': {'se', 'sme'}, 'smo': {'sm', 'smo'}, 'sna': {'sn', 'sna'}, 'snd': {'snd', 'sd'}, 'som': {'som', 'so'}, 'sot': {'sot', 'st'}, 'spa': {'spa', 'es'}, 'sqi': {'sqi', 'sq', 'alb'}, 'srd': {'srd', 'sc'}, 'srn': {'srn'}, 'srp': {'sr', 'srp'}, 'ssw': {'ss', 'ssw'}, 'sun': {'su', 'sun'}, 'swa': {'sw', 'swa'}, 'swe': {'swe', 'sv'}, 'syr': {'syr'}, 'szl': {'szl'}, 'tam': {'tam', 'ta'}, 'tat': {'tat', 'tt'}, 'tbz': {'tbz'}, 'tcy': {'tcy'}, 'tdx': {'tdx'}, 'tel': {'tel', 'te'}, 'tet': {'tet'}, 'tgk': {'tgk', 'tg'}, 'tgl': {'tgl', 'tl'}, 'tha': {'th', 'tha'}, 'tir': {'ti', 'tir'}, 'tiv': {'tiv'}, 'tlh': {'tlh'}, 'ton': {'to', 'ton'}, 'tpi': {'tpi'}, 'tsn': {'tn', 'tsn'}, 'tso': {'tso', 'ts'}, 'tuk': {'tk', 'tuk'}, 'tum': {'tum'}, 'tur': {'tur', 'tr'}, 'twi': {'tw', 'twi'}, 'tyv': {'tyv'}, 'tzo': {'tzo'}, 'udm': {'udm'}, 'uig': {'uig', 'ug'}, 'ukr': {'uk', 'ukr'}, 'umb': {'umb'}, 'urd': {'urd', 'ur'}, 'uzb': {'uz', 'uzb'}, 'uzn': {'uzn'}, 'vec': {'vec'}, 'ven': {'ven', 've'}, 'vep': {'vep'}, 'vie': {'vie', 'vi'}, 'vls': {'vls'}, 'vol': {'vol', 'vo'}, 'war': {'war'}, 'wln': {'wa', 'wln'}, 'wol': {'wo', 'wol'}, 'wuu': {'wuu'}, 'xal': {'xal'}, 'xho': {'xho', 'xh'}, 'xmf': {'xmf'}, 'ydd': {'ydd'}, 'yid': {'yi', 'yid'}, 'yor': {'yor', 'yo'}, 'yua': {'yua'}, 'yue': {'yue'}, 'zap': {'zap'}, 'zho': {'chi', 'zho', 'zh'}, 'zsm': {'zsm'}, 'zul': {'zu', 'zul'}, 'zza': {'zza'}}

# Map known macrolanguages to individual language codes (all ISO 639-3).
MACROLANG_MAPPING = {'aka': {'fat', 'aka', 'twi'}, 'alb': {'aat', 'alb', 'als', 'sqi', 'aln', 'aae'}, 'ara': {'avl', 'abh', 'ayn', 'ssh', 'acm', 'acq', 'afb', 'adf', 'acy', 'aec', 'abv', 'arb', 'arz', 'auz', 'ayh', 'ayp', 'ars', 'ayl', 'shu', 'arq', 'apd', 'aao', 'aeb', 'pga', 'acx', 'ara', 'apc', 'acw', 'ary'}, 'aym': {'ayr', 'ayc', 'aym'}, 'aze': {'azb', 'azj', 'aze'}, 'bal': {'bcc', 'bgp', 'bgn', 'bal'}, 'bik': {'lbl', 'fbl', 'cts', 'rbl', 'ubl', 'bln', 'bcl', 'bik', 'bto'}, 'bnc': {'obk', 'ebk', 'lbk', 'rbk', 'bnc', 'vbk'}, 'bua': {'bxu', 'bua', 'bxr', 'bxm'}, 'chi': {'csp', 'wuu', 'hsn', 'cjy', 'gan', 'cnp', 'nan', 'zho', 'cmn', 'mnp', 'cpx', 'cdo', 'czo', 'hak', 'lzh', 'czh', 'chi', 'yue'}, 'chm': {'mhr', 'chm', 'mrj'}, 'cre': {'crj', 'csw', 'cwd', 'cre', 'crk', 'crm', 'crl'}, 'del': {'unm', 'del', 'umu'}, 'den': {'xsl', 'scs', 'den'}, 'din': {'dik', 'dib', 'diw', 'dks', 'dip', 'din'}, 'doi': {'dgo', 'xnr', 'doi'}, 'est': {'est', 'ekk', 'vro'}, 'fas': {'prs', 'pes', 'fas'}, 'ful': {'fuv', 'fue', 'ful', 'ffm', 'fuc', 'fui', 'fuf', 'fub', 'fuh', 'fuq'}, 'gba': {'gbp', 'gmm', 'gba', 'bdt', 'gso', 'gbq', 'gya'}, 'gon': {'gon', 'gno', 'wsg', 'esg'}, 'grb': {'grj', 'gry', 'gbo', 'grb', 'grv', 'gec'}, 'grn': {'gug', 'gnw', 'grn', 'gun', 'nhd', 'gui'}, 'hai': {'hax', 'hai', 'hdn'}, 'hbs': {'hbs', 'hrv', 'cnr', 'srp', 'bos'}, 'hmn': {'hmd', 'mmr', 'sfm', 'hmj', 'hmy', 'hme', 'mww', 'hmn', 'hmp', 'hmw', 'hmc', 'hea', 'hrm', 'hmm', 'hmh', 'hmq', 'huj', 'muq', 'hma', 'hml', 'cqd', 'hmz', 'hnj', 'hmi', 'hms', 'hmg'}, 'iku': {'ikt', 'iku', 'ike'}, 'ipk': {'esi', 'esk', 'ipk'}, 'jrb': {'yhd', 'jye', 'yud', 'aju', 'jrb', 'aeb'}, 'kau': {'knc', 'kau', 'kby', 'krt'}, 'kln': {'oki', 'pko', 'enb', 'sgc', 'kln', 'eyo', 'spy', 'tec', 'tuy', 'niq'}, 'kok': {'knn', 'gom', 'kok'}, 'kom': {'koi', 'kpv', 'kom'}, 'kon': {'ldi', 'kng', 'kwy', 'kon'}, 'kpe': {'gkp', 'kpe', 'xpe'}, 'kur': {'kur', 'ckb', 'kmr', 'sdh'}, 'lah': {'phr', 'hno', 'pnb', 'skr', 'hnd', 'jat', 'xhe', 'lah'}, 'lav': {'lav', 'ltg', 'lvs'}, 'luy': {'lks', 'nyd', 'lrm', 'nle', 'lkb', 'lto', 'lwg', 'rag', 'lri', 'lts', 'ida', 'luy', 'bxk', 'lsm', 'lko'}, 'man': {'mlq', 'emk', 'mwk', 'mnk', 'mku', 'man', 'msc'}, 'may': {'may', 'jax', 'lcf', 'bjn', 'urk', 'zmi', 'vkk', 'min', 'vkt', 'dup', 'max', 'mui', 'meo', 'lce', 'kxd', 'bve', 'kvr', 'tmw', 'msa', 'jak', 'liw', 'hji', 'zlm', 'kvb', 'mfb', 'bvu', 'orn', 'msi', 'ors', 'ind', 'pel', 'xmm', 'pse', 'coa', 'mqg', 'btj', 'mfa', 'zsm'}, 'mlg': {'tkg', 'xmv', 'bmm', 'tdx', 'skg', 'txy', 'xmw', 'msh', 'bzc', 'bhr', 'mlg', 'plt'}, 'mon': {'mvf', 'khk', 'mon'}, 'msa': {'jax', 'lcf', 'bjn', 'urk', 'zmi', 'vkk', 'min', 'vkt', 'dup', 'max', 'mui', 'meo', 'lce', 'kxd', 'bve', 'kvr', 'tmw', 'msa', 'jak', 'liw', 'hji', 'zlm', 'kvb', 'mfb', 'bvu', 'orn', 'msi', 'ors', 'ind', 'pel', 'xmm', 'pse', 'coa', 'mqg', 'btj', 'mfa', 'zsm'}, 'mwr': {'mtr', 'mve', 'swv', 'wry', 'dhd', 'mwr', 'rwr'}, 'nep': {'nep', 'npi', 'dty'}, 'nor': {'nob', 'nno', 'nor'}, 'oji': {'ojs', 'ojg', 'ojb', 'ciw', 'ojc', 'otw', 'ojw', 'oji'}, 'ori': {'ori', 'ory', 'spv'}, 'orm': {'orm', 'gaz', 'orc', 'gax', 'hae'}, 'per': {'per', 'prs', 'pes', 'fas'}, 'pus': {'pus', 'pst', 'pbu', 'pbt'}, 'que': {'quy', 'qub', 'qur', 'quh', 'qus', 'qxr', 'qup', 'quz', 'qxu', 'qxw', 'qxt', 'qxo', 'qwa', 'qvm', 'quk', 'qws', 'qvl', 'qvc', 'quw', 'quf', 'qva', 'qvs', 'qvo', 'qvj', 'qxl', 'qxc', 'qxp', 'qvw', 'qxn', 'qvp', 'qvn', 'qve', 'qug', 'qvz', 'que', 'qxh', 'qux', 'qud', 'qvi', 'qwh', 'qvh', 'qxa', 'qul', 'qwc'}, 'raj': {'mup', 'wbr', 'hoj', 'raj', 'bgq', 'gda', 'gju'}, 'rom': {'rml', 'rom', 'rmw', 'rmc', 'rmf', 'rmy', 'rmo', 'rmn'}, 'san': {'vsn', 'cls', 'san'}, 'sqi': {'aat', 'als', 'sqi', 'aln', 'aae'}, 'srd': {'srd', 'sdn', 'src', 'sdc', 'sro'}, 'swa': {'swh', 'swc', 'swa'}, 'syr': {'syr', 'cld', 'aii'}, 'tmh': {'tmh', 'thz', 'thv', 'taq', 'ttq'}, 'uzb': {'uzn', 'uzb', 'uzs'}, 'yid': {'ydd', 'yid', 'yih'}, 'zap': {'zpr', 'zpe', 'zat', 'zpg', 'zpp', 'zax', 'ztp', 'ztt', 'zsr', 'zam', 'zts', 'zaf', 'ztm', 'ztq', 'zas', 'zpo', 'zpd', 'zpn', 'zpy', 'zpw', 'zpl', 'zpj', 'zpt', 'zaa', 'zab', 'ztg', 'zpm', 'zte', 'zpz', 'zpx', 'zpi', 'zap', 'zty', 'zpa', 'zae', 'zad', 'zpc', 'zpv', 'zph', 'ztu', 'zac', 'zar', 'zaw', 'zcd', 'zpb', 'zpq', 'zpk', 'zps', 'zai', 'ztx', 'zao', 'zca', 'zpf', 'zpu', 'ztl', 'ztn', 'zav', 'zoo', 'zaq'}, 'zha': {'zgb', 'zzj', 'zyb', 'zhn', 'zyn', 'zyg', 'zgm', 'zln', 'zhd', 'zqe', 'zyj', 'zlj', 'zeh', 'zha', 'zch', 'zlq', 'zgn'}, 'zho': {'csp', 'wuu', 'hsn', 'cjy', 'gan', 'cnp', 'zho', 'nan', 'cmn', 'mnp', 'cpx', 'cdo', 'czo', 'hak', 'lzh', 'czh', 'yue'}, 'zza': {'kiu', 'zza', 'diq'}}
MACROLANG_MAPPING['arb'] = MACROLANG_MAPPING['ara']

# Map known macrolanguages individual language codes and names.
MACROLANG_MAPPING_ALIASES = {'aka': {'Twi', 'fat', 'ak', 'Akan', 'tw', 'Fanti', 'twi', 'aka'}, 'alb': {'Arbereshe Albanian', 'aat', 'Tosk Albanian', 'sq', 'Albanian', 'als', 'alb', 'Arvanitika Albanian', 'Gheg Albanian', 'sqi', 'aln', 'aae'}, 'ara': {'avl', 'Mesopotamian Arabic', 'Sudanese Creole Arabic', 'abh', 'Libyan Arabic', 'Hijazi Arabic', 'ayn', 'ssh', 'acm', 'Egyptian Arabic', 'Tunisian Arabic', 'acq', 'Saidi Arabic', 'Omani Arabic', 'afb', 'adf', 'Algerian Saharan Arabic', 'acy', 'Eastern Egyptian Bedawi Arabic', 'aec', 'abv', "Ta'izzi-Adeni Arabic", 'arb', 'arz', 'Tajiki Arabic', 'auz', 'ayh', 'ayp', 'ars', 'Arabic', 'ar', 'Moroccan Arabic', 'ayl', 'Shihhi Arabic', 'Sanaani Arabic', 'shu', 'arq', 'Gulf Arabic', 'aao', 'apd', 'North Mesopotamian Arabic', 'Algerian Arabic', 'aeb', 'Standard Arabic', 'Chadian Arabic', 'pga', 'Levantine Arabic', 'acx', 'Cypriot Arabic', 'ara', 'apc', 'Najdi Arabic', 'Uzbeki Arabic', 'acw', 'Hadrami Arabic', 'Sudanese Arabic', 'Baharna Arabic', 'ary', 'Dhofari Arabic'}, 'aym': {'ayc', 'ay', 'Central Aymara', 'ayr', 'aym', 'Southern Aymara', 'Aymara'}, 'aze': {'South Azerbaijani', 'azb', 'North Azerbaijani', 'azj', 'Azerbaijani', 'aze', 'az'}, 'bal': {'Baluchi', 'bcc', 'bgn', 'Southern Balochi', 'Eastern Balochi', 'bgp', 'Western Balochi', 'bal'}, 'bik': {'Southern Catanduanes Bikol', 'Bikol', 'lbl', 'fbl', 'Central Bikol', 'cts', 'Northern Catanduanes Bikol', 'rbl', 'bln', 'Miraya Bikol', "Buhi'non Bikol", 'Libon Bikol', 'bcl', 'ubl', 'bik', 'Rinconada Bikol', 'bto', 'West Albay Bikol'}, 'bnc': {'obk', 'ebk', 'lbk', 'Northern Bontok', 'Bontok', 'Central Bontok', 'rbk', 'Southwestern Bontok', 'Southern Bontok', 'bnc', 'vbk', 'Eastern Bontok'}, 'bua': {'bxu', 'Russia Buriat', 'bua', 'Buriat', 'Mongolia Buriat', 'bxr', 'China Buriat', 'bxm'}, 'chi': {'Huizhou Chinese', 'Min Zhong Chinese', 'nan', 'Jinyu Chinese', 'Northern Ping Chinese', 'zh', 'Yue Chinese', 'chi', 'wuu', 'hsn', 'cnp', 'Mandarin Chinese', 'mnp', 'czo', 'czh', 'Southern Ping Chinese', 'Chinese', 'Min Bei Chinese', 'cjy', 'Literary Chinese', 'Hakka Chinese', 'Gan Chinese', 'cpx', 'yue', 'csp', 'gan', 'zho', 'cmn', 'Pu-Xian Chinese', 'cdo', 'hak', 'lzh', 'Wu Chinese', 'Xiang Chinese', 'Min Nan Chinese', 'Min Dong Chinese'}, 'chm': {'Western Mari', 'mhr', 'Eastern Mari', 'chm', 'mrj', 'Mari'}, 'cre': {'crj', 'Plains Cree', 'Southern East Cree', 'Northern East Cree', 'csw', 'Swampy Cree', 'cwd', 'Cree', 'cr', 'cre', 'crk', 'Moose Cree', 'crm', 'crl', 'Woods Cree'}, 'del': {'unm', 'Unami', 'umu', 'del', 'Munsee', 'Delaware'}, 'den': {'South Slavey', 'scs', 'Athapascan', 'Slavey', 'xsl', 'North Slavey', 'den'}, 'din': {'Southwestern Dinka', 'Southeastern Dinka', 'dik', 'din', 'dib', 'diw', 'dks', 'South Central Dinka', 'dip', 'Northwestern Dinka', 'Northeastern Dinka', 'Dinka'}, 'doi': {'xnr', 'Kangri', 'doi', 'dgo', 'Dogri'}, 'est': {'Standard Estonian', 'est', 'et', 'Voro', 'ekk', 'Estonian', 'vro'}, 'fas': {'pes', 'Iranian Persian', 'Persian', 'per', 'prs', 'Dari', 'fa', 'fas'}, 'ful': {'Pulaar', 'fuf', 'Fulah', 'fub', 'Central-Eastern Niger Fulfulde', 'Bagirmi Fulfulde', 'fuc', 'fuh', 'Maasina Fulfulde', 'Borgu Fulfulde', 'fui', 'Adamawa Fulfulde', 'Western Niger Fulfulde', 'fuq', 'Pular', 'fuv', 'fue', 'ful', 'ffm', 'ff', 'Nigerian Fulfulde'}, 'gba': {'gbp', 'Gbaya-Bossangoa', 'gmm', 'Southwest Gbaya', 'bdt', 'gso', 'Northwest Gbaya', 'gba', 'gbq', 'Gbaya', 'gya', 'Bokoto', 'Gbaya-Mbodomo', 'Gbaya-Bozoum'}, 'gon': {'Northern Gondi', 'gno', 'Aheri Gondi', 'Adilabad Gondi', 'esg', 'Gondi', 'gon', 'wsg'}, 'grb': {'grj', 'Grebo', 'Barclayville Grebo', 'gry', 'gbo', 'grv', 'grb', 'gec', 'Southern Grebo', 'Gboloo Grebo', 'Central Grebo', 'Northern Grebo'}, 'grn': {'Chiripa', 'nhd', 'gug', 'Guarani', 'gnw', 'Mbya Guarani', 'grn', 'Paraguayan Guarani', 'gun', 'Eastern Bolivian Guarani', 'Western Bolivian Guarani', 'gui', 'gn'}, 'hai': {'hax', 'Haida', 'hdn', 'Southern Haida', 'hai', 'Northern Haida'}, 'hbs': {'hbs', 'hrv', 'cnr', 'hr', 'sh', 'srp', 'Croatian', 'Serbo-Croatian', 'bos', 'Montenegrin', 'bs', 'Serbian', 'sr', 'Bosnian'}, 'hmn': {'Western Xiangxi Miao', 'hmd', 'Eastern Huishui Hmong', 'Northern Qiandong Miao', 'mmr', 'sfm', 'hmj', 'hmy', 'Luopohe Hmong', 'hme', 'mww', 'Southern Guiyang Hmong', 'Horned Miao', 'hmn', 'Central Mashan Hmong', 'Eastern Qiandong Miao', 'Southern Qiandong Miao', 'hmp', 'Hmong', 'hmw', 'hmc', 'Hmong Shua', 'Northern Huishui Hmong', 'Northern Mashan Hmong', 'Western Mashan Hmong', 'Chuanqiandian Cluster Miao', 'Hmong Daw', 'hea', 'Southern Mashan Hmong', 'Ge', 'Southwestern Guiyang Hmong', 'hrm', 'Northern Guiyang Hmong', 'hmh', 'hmm', 'hmq', 'huj', 'Large Flowery Miao', 'muq', 'Small Flowery Miao', 'hma', 'Southwestern Huishui Hmong', 'cqd', 'hml', 'hmz', 'Hmong Njua', 'hnj', 'Eastern Xiangxi Miao', 'Central Huishui Hmong', 'hmi', 'hms', 'hmg'}, 'iku': {'Eastern Canadian Inuktitut', 'ike', 'ikt', 'iku', 'Inuinnaqtun', 'iu', 'Inuktitut'}, 'ipk': {'ipk', 'Inupiaq', 'North Alaskan Inupiatun', 'ik', 'esi', 'Northwest Alaska Inupiatun', 'esk'}, 'jrb': {'Judeo-Yemeni Arabic', 'yhd', 'yud', 'jrb', 'Judeo-Arabic', 'Judeo-Iraqi Arabic', 'Judeo-Moroccan Arabic', 'Tunisian Arabic', 'Judeo-Tripolitanian Arabic', 'aju', 'jye', 'aeb'}, 'kau': {'krt', 'knc', 'kby', 'kau', 'kr', 'Central Kanuri', 'Manga Kanuri', 'Tumari Kanuri', 'Kanuri'}, 'kln': {'oki', 'Tugen', 'Terik', 'Okiek', 'tec', 'Kalenjin', 'enb', 'Sabaot', 'niq', 'Keiyo', 'pko', 'Nandi', 'tuy', 'Kipsigis', 'Pokoot', 'Markweeta', 'spy', 'eyo', 'kln', 'sgc'}, 'kok': {'knn', 'Konkani', 'kok', 'Goan Konkani', 'gom'}, 'kom': {'kom', 'kv', 'koi', 'Komi-Permyak', 'Komi', 'kpv', 'Komi-Zyrian'}, 'kon': {'kng', 'kon', 'ldi', 'San Salvador Kongo', 'Koongo', 'kwy', 'Kongo', 'Laari', 'kg'}, 'kpe': {'xpe', 'Kpelle', 'gkp', 'Guinea Kpelle', 'kpe', 'Liberia Kpelle'}, 'kur': {'kur', 'kmr', 'Central Kurdish', 'ku', 'Northern Kurdish', 'ckb', 'Southern Kurdish', 'Kurdish', 'sdh'}, 'lah': {'hno', 'phr', 'pnb', 'Western Panjabi', 'skr', 'lah', 'Jakati', 'hnd', 'Southern Hindko', 'Lahnda', 'jat', 'xhe', 'Northern Hindko', 'Saraiki', 'Khetrani', 'Pahari-Potwari'}, 'lav': {'lvs', 'Latvian', 'Standard Latvian', 'lav', 'ltg', 'Latgalian', 'lv'}, 'luy': {'Marachi', 'lrm', 'Tsotso', 'rag', 'East Nyala', 'Marama', 'nyd', 'lts', 'ida', 'Kisa', 'nle', 'Bukusu', 'lks', 'lto', 'lwg', 'Kabras', 'Idakho-Isukha-Tiriki', 'Wanga', 'Tachoni', 'luy', 'Logooli', 'Nyore', 'lsm', 'lko', 'Saamia', 'Luyia', 'lkb', 'Khayo', 'lri', 'bxk'}, 'man': {'Kita Maninkakan', 'Mandingo', 'Western Maninkakan', 'mlq', 'emk', 'Mandinka', 'mwk', 'Eastern Maninkakan', 'Konyanka Maninka', 'mku', 'mnk', 'Sankaran Maninka', 'man', 'msc'}, 'may': {'jax', 'Kubu', 'may', 'North Moluccan Malay', 'Haji', 'Standard Malay', 'lcf', 'bjn', 'Col', 'mfa', 'Brunei', 'urk', 'Kota Bangun Kutai Malay', 'Negeri Sembilan Malay', 'zmi', 'vkk', 'min', 'vkt', 'dup', 'Loncong', 'Bangka', 'Banjar', 'Lubu', 'max', 'Duano', 'Pattani Malay', 'Indonesian', 'Sabah Malay', 'meo', 'mui', 'Berau Malay', 'Orang Seletar', 'Malay', 'Minangkabau', 'kxd', 'bve', 'Jambi Malay', 'kvr', 'lce', 'Tenggarong Kutai Malay', 'tmw', 'Manado Malay', 'Kedah Malay', 'msa', 'jak', 'liw', 'id', 'Orang Kanaq', 'hji', 'zlm', 'kvb', 'mfb', 'bvu', 'orn', 'Jakun', 'msi', 'Pekal', 'Temuan', 'Bukit Malay', 'Musi', 'ors', 'ind', 'Central Malay', 'pel', 'Kaur', 'xmm', 'ms', 'pse', 'coa', 'Kerinci', 'mqg', 'btj', "Urak Lawoi'", 'Bacanese Malay', 'Cocos Islands Malay', 'zsm'}, 'mlg': {'tkg', 'bmm', 'Sakalava Malagasy', 'mg', 'plt', 'Antankarana Malagasy', 'xmv', 'Masikoro Malagasy', 'Northern Betsimisaraka Malagasy', 'xmw', 'Bara Malagasy', 'Tsimihety Malagasy', 'Southern Betsimisaraka Malagasy', 'Tanosy Malagasy', 'skg', 'Tandroy-Mahafaly Malagasy', 'txy', 'Tesaka Malagasy', 'bzc', 'bhr', 'Plateau Malagasy', 'tdx', 'Malagasy', 'msh', 'mlg'}, 'mon': {'Mongolian', 'khk', 'mvf', 'Halh Mongolian', 'mon', 'Peripheral Mongolian', 'mn'}, 'msa': {'jax', 'Kubu', 'may', 'North Moluccan Malay', 'Haji', 'Standard Malay', 'lcf', 'bjn', 'Col', 'mfa', 'Brunei', 'urk', 'Kota Bangun Kutai Malay', 'Negeri Sembilan Malay', 'zmi', 'vkk', 'min', 'vkt', 'dup', 'Loncong', 'Bangka', 'Banjar', 'Lubu', 'max', 'Duano', 'Pattani Malay', 'Indonesian', 'Sabah Malay', 'meo', 'mui', 'Berau Malay', 'Orang Seletar', 'Malay', 'Minangkabau', 'kxd', 'bve', 'Jambi Malay', 'kvr', 'lce', 'Tenggarong Kutai Malay', 'tmw', 'Manado Malay', 'Kedah Malay', 'msa', 'jak', 'liw', 'id', 'Orang Kanaq', 'hji', 'zlm', 'kvb', 'mfb', 'bvu', 'orn', 'Jakun', 'msi', 'Pekal', 'Temuan', 'Bukit Malay', 'Musi', 'ors', 'ind', 'Central Malay', 'pel', 'Kaur', 'xmm', 'ms', 'pse', 'coa', 'Kerinci', 'mqg', 'btj', "Urak Lawoi'", 'Bacanese Malay', 'Cocos Islands Malay', 'zsm'}, 'mwr': {'Merwari', 'mtr', 'mve', 'Dhundari', 'Marwari', 'Shekhawati', 'swv', 'wry', 'dhd', 'mwr', 'Mewari', 'rwr'}, 'nep': {'Dotyali', 'dty', 'ne', 'nep', 'npi', 'Nepali'}, 'nor': {'Norwegian Nynorsk', 'nb', 'Norwegian Bokmal', 'no', 'Norwegian', 'nor', 'nn', 'nno', 'nob'}, 'oji': {'Chippewa', 'oji', 'Northwestern Ojibwa', 'ojs', 'Severn Ojibwa', 'Ojibwa', 'ojg', 'Western Ojibwa', 'ojb', 'ciw', 'ojc', 'Eastern Ojibwa', 'Central Ojibwa', 'oj', 'ojw', 'Ottawa', 'otw'}, 'ori': {'ory', 'Sambalpuri', 'ori', 'Odia', 'Oriya', 'spv', 'or'}, 'orm': {'Orma', 'Eastern Oromo', 'West Central Oromo', 'gaz', 'orm', 'Oromo', 'orc', 'Borana-Arsi-Guji Oromo', 'gax', 'hae', 'om'}, 'per': {'pes', 'Iranian Persian', 'Persian', 'per', 'prs', 'Dari', 'fa', 'fas'}, 'pus': {'Northern Pashto', 'pbt', 'ps', 'pbu', 'Southern Pashto', 'Pushto', 'pus', 'pst', 'Central Pashto'}, 'que': {'qub', 'qur', 'Ayacucho Quechua', 'Huamalies-Dos de Mayo Huanuco Quechua', 'qxw', 'qxt', 'Jauja Wanca Quechua', 'qxo', 'qvm', 'Northern Pastaza Quichua', 'quf', 'Canar Highland Quichua', 'qvj', 'qxn', 'qvn', 'Sihuas Ancash Quechua', 'qug', 'Huaylla Wanca Quechua', 'qwh', 'qul', 'Chiquian Ancash Quechua', 'Chimborazo Highland Quichua', 'Yanahuanca Pasco Quechua', 'Cusco Quechua', 'Huallaga Huanuco Quechua', 'Southern Conchucos Ancash Quechua', 'qxr', 'Quechua', 'San Martin Quechua', 'qvl', 'qvc', 'Chachapoyas Quechua', 'Pacaraos Quechua', 'quw', 'Yauyos Quechua', 'qxl', 'Ambo-Pasco Quechua', 'qxp', 'Arequipa-La Union Quechua', 'qvw', 'qve', 'qu', 'qud', 'qxa', 'Santa Ana de Tusi Pasco Quechua', 'qus', 'Cajatambo North Lima Quechua', 'Napo Lowland Quechua', 'qup', 'qwa', 'Southern Pastaza Quechua', 'qva', 'qvo', 'Eastern Apurimac Quechua', 'qxc', 'North Bolivian Quechua', 'qvz', 'que', 'qux', 'Cajamarca Quechua', 'qvh', 'quy', 'Panao Huanuco Quechua', 'quh', 'Northern Conchucos Ancash Quechua', 'quz', 'qxu', 'Santiago del Estero Quichua', 'Salasaca Highland Quichua', 'quk', 'South Bolivian Quechua', 'North Junin Quechua', 'qws', 'Lambayeque Quechua', 'qvs', 'Imbabura Highland Quichua', 'Puno Quechua', 'Calderon Highland Quichua', 'qvp', 'Classical Quechua', 'Margos-Yarowilca-Lauricocha Quechua', 'qxh', 'Tena Lowland Quichua', 'qvi', 'Corongo Ancash Quechua', 'Loja Highland Quichua', 'qwc', 'Chincha Quechua', 'Huaylas Ancash Quechua'}, 'raj': {'Bagri', 'mup', 'wbr', 'Rajasthani', 'Hadothi', 'Gujari', 'hoj', 'raj', 'bgq', 'Wagdi', 'Gade Lohar', 'Malvi', 'gda', 'gju'}, 'rom': {'Balkan Romani', 'Romany', 'Sinte Romani', 'rom', 'Baltic Romani', 'rmn', 'rmw', 'rmc', 'rmf', 'Vlax Romani', 'Welsh Romani', 'rmo', 'Carpathian Romani', 'rmy', 'Kalo Finnish Romani', 'rml'}, 'san': {'sa', 'vsn', 'san', 'Sanskrit', 'Classical Sanskrit', 'Vedic Sanskrit', 'cls'}, 'sqi': {'Arbereshe Albanian', 'aat', 'Tosk Albanian', 'sq', 'Albanian', 'als', 'alb', 'Arvanitika Albanian', 'Gheg Albanian', 'sqi', 'aln', 'aae'}, 'srd': {'Logudorese Sardinian', 'srd', 'Gallurese Sardinian', 'Sassarese Sardinian', 'sdn', 'sdc', 'src', 'sro', 'Campidanese Sardinian', 'sc', 'Sardinian'}, 'swa': {'swh', 'Congo Swahili', 'swc', 'Swahili', 'swa', 'sw'}, 'syr': {'aii', 'cld', 'Chaldean Neo-Aramaic', 'syr', 'Syriac', 'Assyrian Neo-Aramaic'}, 'tmh': {'tmh', 'Tamashek', 'Tahaggart Tamahaq', 'Tayart Tamajeq', 'thz', 'thv', 'Tawallammat Tamajaq', 'Tamasheq', 'taq', 'ttq'}, 'uzb': {'uz', 'uzs', 'Northern Uzbek', 'Southern Uzbek', 'uzb', 'Uzbek', 'uzn'}, 'yid': {'Eastern Yiddish', 'yi', 'yih', 'Yiddish', 'yid', 'Western Yiddish', 'ydd'}, 'zap': {'San Baltazar Loxicha Zapotec', 'Quioquitani-Quieri Zapotec', 'Western Tlacolula Valley Zapotec', 'zpg', 'Mazaltepec Zapotec', 'zax', 'ztt', 'Loxicha Zapotec', 'Santo Domingo Albarradas Zapotec', 'zts', 'Quiavicuzas Zapotec', 'Chichicapan Zapotec', 'zpy', 'Ocotlan Zapotec', 'ztg', 'zte', 'Texmelucan Zapotec', 'Petapa Zapotec', 'zpi', 'zap', 'zty', 'Lachixio Zapotec', 'zph', 'zaw', 'Yalalag Zapotec', 'zpk', 'Santa Ines Yatzechi Zapotec', 'zai', 'zao', 'ztn', 'zaq', 'zat', 'Ozolotepec Zapotec', 'Zoogocho Zapotec', 'zsr', 'zam', 'Tilquiapan Zapotec', 'zpd', 'Rincon Zapotec', 'zpn', 'Zaniza Zapotec', 'Amatlan Zapotec', 'Yatzachi Zapotec', 'zpl', 'zpj', 'zpt', 'zab', 'zpx', 'Cajonos Zapotec', 'zad', 'Sierra de Juarez Zapotec', 'Isthmus Zapotec', 'Lapaguia-Guivini Zapotec', 'Tejalapan Zapotec', 'zca', 'zpf', 'San Pedro Quiatoni Zapotec', 'ztl', 'ztp', 'zav', 'ztm', 'Xadani Zapotec', 'zpp', 'zaf', 'ztq', 'zpo', 'zpw', 'Zaachila Zapotec', 'zpm', 'zpz', 'San Vicente Coatlan Zapotec', 'Tabaa Zapotec', 'zpa', 'Choapan Zapotec', 'Yatee Zapotec', 'zpc', 'zpv', 'San Agustin Mixtepec Zapotec', 'Southern Rincon Zapotec', 'Xanaguia Zapotec', 'ztx', 'zpu', 'Tlacolulita Zapotec', 'zoo', 'zpr', 'Ayoquesco Zapotec', 'zpe', 'Guevea De Humboldt Zapotec', 'Santa Catarina Albarradas Zapotec', 'Yautepec Zapotec', 'El Alto Zapotec', 'Guila Zapotec', 'Santa Maria Quiegolani Zapotec', 'zas', 'Aloapam Zapotec', 'Coatlan Zapotec', 'Zapotec', 'Las Delicias Zapotec', 'Yareni Zapotec', 'Mitla Zapotec', 'zaa', 'Totomachapan Zapotec', 'Coatecas Altas Zapotec', 'Asuncion Mixtepec Zapotec', 'Mixtepec Zapotec', 'zae', 'ztu', 'zac', 'Miahuatlan Zapotec', 'zar', 'zcd', 'zpb', 'zpq', 'zps', 'Southeastern Ixtlan Zapotec', 'Elotepec Zapotec', 'Santiago Xanica Zapotec', 'Lachiguiri Zapotec'}, 'zha': {'zgb', 'zhn', 'zqe', 'Zuojiang Zhuang', 'zeh', 'zha', 'Liuqian Zhuang', 'zzj', 'Youjiang Zhuang', 'zgm', 'Yongbei Zhuang', 'Minz Zhuang', 'zln', 'Guibian Zhuang', 'Qiubei Zhuang', 'Liujiang Zhuang', 'zlj', 'Central Hongshuihe Zhuang', 'Nong Zhuang', 'Lianshan Zhuang', 'za', 'Yongnan Zhuang', 'zyg', 'zhd', 'zgn', 'zyb', 'zyn', 'Yang Zhuang', 'Eastern Hongshuihe Zhuang', 'Zhuang', 'zyj', 'Guibei Zhuang', 'Dai Zhuang', 'zlq', 'zch'}, 'zho': {'Huizhou Chinese', 'Min Zhong Chinese', 'nan', 'Jinyu Chinese', 'Northern Ping Chinese', 'zh', 'Yue Chinese', 'chi', 'wuu', 'hsn', 'cnp', 'Mandarin Chinese', 'mnp', 'czo', 'czh', 'Southern Ping Chinese', 'Chinese', 'Min Bei Chinese', 'cjy', 'Literary Chinese', 'Hakka Chinese', 'Gan Chinese', 'cpx', 'yue', 'csp', 'gan', 'zho', 'cmn', 'Pu-Xian Chinese', 'cdo', 'hak', 'lzh', 'Wu Chinese', 'Xiang Chinese', 'Min Nan Chinese', 'Min Dong Chinese'}, 'zza': {'zza', 'kiu', 'Dimli', 'Kirmanjki', 'Zaza', 'diq'}}
MACROLANG_MAPPING_ALIASES['arb'] = MACROLANG_MAPPING_ALIASES['ara']


lang_to_queries = dict()
for lang in [l[:3] for l in LANG_SETS['5mb']]:
    queries = set()
    if lang in LANG_NAMES: queries.add(LANG_NAMES[lang].lower())
    if lang in LANG_CODES: queries.update(LANG_CODES[lang])
    if lang in MACROLANG_MAPPING: queries.update(MACROLANG_MAPPING[lang])
    if lang in MACROLANG_MAPPING_ALIASES: queries.update(
            [q.lower() for q in MACROLANG_MAPPING_ALIASES[lang]])
    lang_to_queries[lang] = queries


# Get available Goldfish models from input language name, two-letter language
# code, or three-letter language code.
def get_available_goldfish(input_query):
    # Process query.
    input_query = input_query.lower()
    if (len(input_query) == 8) and (input_query[3] == '_'):
        query_lang = input_query[:3]
        query_script = input_query[4:]
    else:
        query_lang = input_query
        query_script = None
    # Search.
    # For queries length <= 3 (e.g. language codes), require exact match.
    # For longer queries, only has to end with the query.
    if len(query_lang) <= 3:
        matching_langs = set([lang for lang, queries in lang_to_queries.items()
                              if query_lang in queries])
    else:
        matching_langs = set([lang for lang, queries in lang_to_queries.items()
                              if any(q.endswith(query_lang) for q in queries)])
    # Get available models.
    possible_models = []
    for possible_lang in LANG_SETS['5mb']:
        if query_script and (possible_lang[4:] != query_script): continue
        if possible_lang[:3] in matching_langs:
            # Match!
            possible_models.append(f'{possible_lang}_5mb')
            if possible_lang in LANG_SETS['10mb']:
                possible_models.append(f'{possible_lang}_10mb')
            if possible_lang in LANG_SETS['100mb']:
                possible_models.append(f'{possible_lang}_100mb')
            if possible_lang in LANG_SETS['1000mb']:
                possible_models.append(f'{possible_lang}_1000mb')
            else:
                possible_models.append(f'{possible_lang}_full')
    return sorted(possible_models)