import codecs, os
import json

from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')


def _get_score_compare(reference, predicted):
    count_ones = 0
    for v in reference:
        if v == 1:
            count_ones += 1
    prc = 0
    for i in range(len(reference)):
        if reference[i] == predicted[i] == 1:
            prc += 1
    return prc / count_ones


def _get_binary_form(err_types, dict_labels):
    new_dict = dict(dict_labels)
    for e in err_types.split("+"):
        if e.replace("\n", "") in new_dict:
            new_dict[e.replace("\n", "")] = 1

    return list(new_dict.values())


def eval_multi_label_subclasses(uc, output_path, extension):
    mapped_err_type = []
    predicted_err_type = []

    dirname = os.path.dirname(__file__)
    conf_path = os.path.join(dirname, "../utils/config.json")

    config_file = open(conf_path)
    config_data = json.load(config_file)
    dict_ = config_data['subcat_tags_orig_merge']
    config_file.close()

    if not uc:
        dict_.pop("UC")
    dict_labels = dict_.keys()
    list_labels_err_type = list(dict_labels)

    with codecs.open(f"{output_path}/annot_input_ref.tsv", "r", "utf8") as f:
        for l in f:
            mapped_err_type.append(_get_binary_form(l.split("\t")[2], dict_))

    with codecs.open(f"{output_path}/annot_input_sys.tsv", "r", "utf8") as f:
        for l in f:
            predicted_err_type.append(_get_binary_form(l.split("\t")[2], dict_))

    results = classification_report(mapped_err_type, predicted_err_type, output_dict=True)
    fw = codecs.open(f"{output_path}/subclasses_results_" + str(extension) + ".tsv", "w", "utf8")
    fw.write("\t".join(["CLASS", "PRECISION", "RECALL", "F1-Score", "SUPPORT"]) + "\n")
    i = 0
    new_sec = False
    for k, v in results.items():
        line = [str(k)]
        for measure, val in v.items():
            if measure in ["precision", "recall", "f1-score", "support"]:
                line.append(str(val))
        if i < len(list_labels_err_type):
            fw.write(list_labels_err_type[i] + "\t" + "\t".join(line[1:]) + "\n")
        else:
            if not new_sec:
                fw.write("\n")
                new_sec = True
            if line[0] != "samples avg":
                fw.write("\t".join(line) + "\n")
        # print("\t".join(line))
        i += 1
    fw.close()
    print("Results saved to: " + f"{output_path}/subclasses_results_" + str(extension))
