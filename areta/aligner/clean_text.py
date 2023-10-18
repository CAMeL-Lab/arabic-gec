import codecs


def clean_file(file_name):
    fw_clmb = codecs.open("sample/" + file_name + "-clean", "w", "utf8")
    with codecs.open("sample/" + file_name, "r", "utf8") as f:
        for l in f:

            nl = " ".join(l.split()[1:])
            new_clean_line = []
            is_waw = False
            for w in l.split()[1:]:
                if w == "و":
                    n_w = w + "#"
                    is_waw = True
                    continue
                if is_waw == False:
                    new_clean_line.append(w)
                else:
                    new_clean_line.append(n_w + w)
                    is_waw = False

            fw_clmb.write(" ".join(new_clean_line) + "\n")
    fw_clmb.close()


f_name1 = "CLMB-1"
f_name2 = "QALB-Test2014.sent"
clean_file(f_name1)
clean_file(f_name2)
