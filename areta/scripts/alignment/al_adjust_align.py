import codecs


def _prepare_item(tmp, k):
    valid = False
    els = []
    for e in tmp:
        els.append(e[2])
        valid = True
    if k[2] == '' and len(els) == 0:
        k[2] = 'Null'
    k[2] = " ".join(els) + " " + k[2]
    k[1] = "|"
    nk = [k[0], k[2]]
    return nk, valid


def adjust_align(output_file_path, input_file_path):
    fw = codecs.open(output_file_path, "w", "utf8")
    tmp = []
    id = 1
    i = 0
    with codecs.open(input_file_path, "r", "utf8") as f:
        for l in f:
            if l.strip():
                k = l.split("\t")
                if k[0] == '':
                    tmp.append(k)
                else:
                    pr, valid = _prepare_item(tmp, k)
                    print(i)
                    fw.write("\t".join(pr[0:2]) + "\n")
                    if valid:
                        tmp = []
                    i = i + 1
            else:
                id = id + 1

    fw.close()
