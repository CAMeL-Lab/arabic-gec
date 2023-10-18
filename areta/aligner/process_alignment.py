import codecs

fw = codecs.open("faifi_all_pairs_auto_align.tsv", "w", "utf8")

tags = []
with codecs.open("sample/fafi_tags.tsv", "r", "utf8") as f:
    for l in f:
        tags.append(l.replace("\n", ""))


def prepare_item(tmp, k):
    valid = False
    els = []
    for e in tmp:
        els.append(e[2])
        valid = True
    # return k, valid
    if k[2] == '' and len(els) == 0:
        k[2] = 'Null'
    k[2] = " ".join(els) + " " + k[2]
    k[1] = "|"
    nk = [k[0], k[2]]
    return nk, valid


tmp = []
id = 1
i = 0
with codecs.open("sample/faifi_basic.ar.basic", "r", "utf8") as f:
    for l in f:
        if l.strip():
            k = l.split("\t")
            # if len(l.split("\t")) != 4:
            if k[0] == '':
                tmp.append(k)
            else:
                pr, valid = prepare_item(tmp, k)
                # fw.write(str(id) + "\t" + "\t".join(pr[0:3]) + "\n")
                print(i)
                fw.write(tags[i] + "\t" +
                         "\t".join(pr[0:2]) + "\n")
                if valid:
                    tmp = []
                # print(len(l.split("\t")))
                i = i + 1
        else:
            id = id + 1

fw.close()
