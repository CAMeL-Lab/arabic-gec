import codecs
from scripts.explainability.ex_get_explanation_raw_correct import explain_error


def _normalize_punct(s):
    # OLD:
    # from unicodedata import category
    # if len(s) > 2 and category(s[0])[0] == 'P' and s[1] == ' ':
    #     return s[2:] + s[0]
    # elif len(s) == 2 and category(s[0])[0] == 'P' and s[1] == ' ':
    #     return s[0]
    # return s

    #NEW:
    #The above function takes out spaces in cases of inserting
    #puncs before or after words. ARETA inserts white spaces for
    #punc subtituation errors, we just need to strip those out.
    from unicodedata import category
    if len(s.strip()) == 1 and category(s.strip()[0])[0] == 'P':
        return s.strip()
    return s


def annotate(aligned_file, annot_file_out, show_paths=False):
    i = 0
    lines = []
    fw = codecs.open(annot_file_out, "w", "utf8")
    with codecs.open(aligned_file, "r", "utf8") as f:
        for l in f:
            if l == "\n":
                fw.write("\n")
                lines.append("\n")
                continue
            if i > 0:
                raw_word = l.split("\t")[0]
                raw_word = raw_word.replace("\r", "")
                correct_word = l.split("\t")[1].replace("\n", "").replace("\r", " ")
                correct_word = _normalize_punct(correct_word)
                # import pdb; pdb.set_trace()
                if correct_word.startswith(" "):
                    correct_word = correct_word[1:]
                try:
                    explain, path = explain_error(raw_word, correct_word)
                    if not explain: explain = 'UNK'
                    line = "\t".join([raw_word, correct_word, "+".join(sorted(list(set(explain.split("+")))))])
                    if show_paths and path:
                        orth_path, morph_path = path[0]
                        line = line + f"\tOrth Path: {orth_path}\tMorph Path: {str(morph_path)}\n"
                    else:
                        line += "\n"
                    lines.append(line)
                    fw.write(line)
                except:
                    line = "\t".join([raw_word, correct_word, "UNK"]) + "\n"
                    lines.append(line)
                    fw.write(line)
            i += 1
    fw.close()
    return "".join(lines)
