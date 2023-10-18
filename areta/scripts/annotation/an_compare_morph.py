import json
import os

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

dirname = os.path.dirname(__file__)

filename_config = os.path.join(dirname, '../../config.json')
config_file = open(filename_config)
config_data = json.load(config_file)
mode = config_data['mode']
mle_top = config_data['mle_top']

global mle
if mle_top != "":
    mle = MLEDisambiguator.pretrained(top=mle_top)
db = MorphologyDB.builtin_db()


def init_params():
    global mle
    global mode
    filename_config = os.path.join(dirname, '../../config.json')
    config_file = open(filename_config)
    config_file = open(config_file)
    config_data = json.load(config_file)
    mode = config_data['mode']
    mle_top = config_data['mle_top']
    mle = MLEDisambiguator.pretrained(top=mle_top)


def morph_error_type(word, correct_word):
    global mle
    word = [word]
    correct_word = [correct_word]
    disambig1 = mle.disambiguate(word)
    disambig2 = mle.disambiguate(correct_word)

    lex_word = [d.analyses[0].analysis['lex'] for d in disambig1]
    lex_correct_word = [d.analyses[0].analysis['lex'] for d in disambig2]

    d_word_analysis = disambig1[0].analyses[0].analysis
    d_correct_word_analysis = disambig2[0].analyses[0].analysis

    d_word_lex = disambig1[0].analyses[0].analysis['lex']
    d_correct_word_lex = disambig2[0].analyses[0].analysis['lex']

    d_word_pos = disambig1[0].analyses[0].analysis['pos']
    d_correct_word_pos = disambig2[0].analyses[0].analysis['pos']

    d_word_source = disambig1[0].analyses[0].analysis['source']
    d_correct_source = disambig2[0].analyses[0].analysis['source']

    if d_word_lex == d_correct_word_lex and d_word_pos == d_correct_word_pos and d_correct_source == "lex" and d_word_source == "lex":
        return ["LEMMA_MATCH", _get_features_differences(d_word_analysis, d_correct_word_analysis)]
    return ["LEMMA_MISMATCH", []]


def _reformat_disambig_analysis(scored_analyses):
    an_list = []
    for an in scored_analyses:
        an_list.append(an.analysis)
    return an_list


def _has_analysis(word, correct_word):
    global db

    # Create analyzer with no backoff
    analyzer = Analyzer(db)

    analyses_word = analyzer.analyze(word)
    analyses_correct_word = analyzer.analyze(correct_word)

    if len(analyses_word) > 0 and len(analyses_correct_word) > 0:
        return True
    return False


def morph_error_type_calimastar(word, correct_word):
    global mode
    if mode == "analyser":
        return morph_error_type_calimastar_analyser(word, correct_word)
    else:
        return morph_error_type_calimastar_mle(word, correct_word)


def morph_error_type_calimastar_analyser(word, correct_word):
    global db

    if word == correct_word:
        return ["SAME_WORD", []]

    # Create analyzer with no backoff
    analyzer = Analyzer(db)

    # Create analyzer with NOAN_PROP backoff
    # analyzer = Analyzer(db, 'NOAN_PROP')

    # or
    # analyzer = Analyzer(db, backoff='NOAN_PROP')

    # To analyze a word, we can use the analyze() method
    analyses_word = analyzer.analyze(word)
    analyses_word = expand_analysis_mod_all_list(analyses_word)
    analyses_word = expand_analysis_gen_all_list(analyses_word)
    # analyses_word = hack_pgn_all(analyses_word)

    analyses_correct_word = analyzer.analyze(correct_word)
    analyses_correct_word = expand_analysis_mod_all_list(analyses_correct_word)
    analyses_correct_word = expand_analysis_gen_all_list(analyses_correct_word)

    list_morph_sub_paths = []
    for analysis_w in analyses_word:
        for analysis_cw in analyses_correct_word:
            if analysis_w["lex"] == analysis_cw["lex"] and analysis_w["pos"] == analysis_cw["pos"] and analysis_w[
                "source"] == "lex" and analysis_cw["source"] == "lex":
                list_morph_sub_paths.append(_get_features_differences(analysis_w, analysis_cw))
    if len(list_morph_sub_paths) > 0:
        return ["LEMMA_MATCH", list_morph_sub_paths]
    return ["LEMMA_MISMATCH", []]


def morph_error_type_calimastar_mle(word, correct_word):
    global db

    if word == correct_word:
        return ["SAME_WORD", []]

    global mle
    word = [word]
    correct_word = [correct_word]

    # Create analyzer with no backoff
    analyzer = Analyzer(db)

    # Create analyzer with NOAN_PROP backoff
    # analyzer = Analyzer(db, 'NOAN_PROP')

    # or
    # analyzer = Analyzer(db, backoff='NOAN_PROP')

    # To analyze a word, we can use the analyze() method

    analyses_word = mle.disambiguate(word)[0].analyses
    analyses_word = _reformat_disambig_analysis(analyses_word)
    analyses_word = expand_analysis_mod_all_list(analyses_word)
    analyses_word = expand_analysis_gen_all_list(analyses_word)
    # analyses_word = hack_pgn_all(analyses_word)

    analyses_correct_word = mle.disambiguate(correct_word)[0].analyses
    analyses_correct_word = _reformat_disambig_analysis(analyses_correct_word)
    analyses_correct_word = expand_analysis_mod_all_list(analyses_correct_word)
    analyses_correct_word = expand_analysis_gen_all_list(analyses_correct_word)
    # analyses_correct_word = hack_pgn_all(analyses_correct_word)

    list_morph_sub_paths = []
    for analysis_w in analyses_word:
        for analysis_cw in analyses_correct_word:
            # if d_word_lex == d_correct_word_lex and d_word_pos == d_correct_word_pos and d_correct_source == "lex" and d_word_source == "lex":
            if analysis_w["lex"] == analysis_cw["lex"] and analysis_w["pos"] == analysis_cw["pos"] and analysis_w[
                "source"] == "lex" and analysis_cw["source"] == "lex":
                list_morph_sub_paths.append(_get_features_differences(analysis_w, analysis_cw))
    if len(list_morph_sub_paths) > 0:
        return ["LEMMA_MATCH", list_morph_sub_paths]
    return ["LEMMA_MISMATCH", []]


def _expand_analysis_mod(analysis):
    exp_list = []
    valid = False
    for k, v in analysis.items():
        if k == 'mod' and v == 'u':
            valid = True
            break
    if valid:
        for i in range(3):
            d = {}
            for k, v in analysis.items():
                if k != 'mod':
                    d[k] = v
                else:
                    if i == 0:
                        d[k] = 'i'
                    if i == 1:
                        d[k] = 'j'
                    if i == 2:
                        d[k] = 's'
            exp_list.append(d)
    else:
        exp_list.append(analysis)
    return exp_list


def _expand_analysis_gen(analysis):
    exp_list = []
    valid = False
    for k, v in analysis.items():
        if k == 'gen' and v == 'u':
            valid = True
            break
    if valid:
        for i in range(3):
            d = {}
            for k, v in analysis.items():
                if k != 'gen':
                    d[k] = v
                else:
                    if i == 0:
                        d[k] = 'i'
                    if i == 1:
                        d[k] = 'j'
                    if i == 2:
                        d[k] = 's'
            exp_list.append(d)
    else:
        exp_list.append(analysis)
    return exp_list


def expand_analysis_mod_all_list(analyses):
    list_expan = []
    for an in analyses:
        list_expan.extend(_expand_analysis_mod(an))
    return list_expan


def expand_analysis_gen_all_list(analyses):
    list_expan = []
    for an in analyses:
        list_expan.extend(_expand_analysis_gen(an))
    return list_expan


def _hack_pgn(analysis):
    if analysis['per'] == '1' and analysis['gen'] == 'm' and analysis['num'] == 's':
        d = {}
        for k, v in analysis.items():
            if k != 'gen':
                d[k] = v
            else:
                d[k] = 'u'
        return [d]
    return [analysis]


def _hack_pgn_all(analyses):
    pgn_new_list = []
    for an in analyses:
        pgn_new_list.extend(_hack_pgn(an))
    return pgn_new_list


def _get_features_differences(analysis1, analysis2):
    target_features = ['num', 'gen', 'per', 'asp', 'vox', 'mod', 'stt', 'cas', 'enc0', 'prc0', 'prc1', 'prc2', 'prc3',
                       'pos', 'rat']
    diff_list = []
    for feature in target_features:
        if analysis1[feature] != analysis2[feature]:
            diff_list.append({feature: (analysis1[feature], analysis2[feature])})
    return diff_list
