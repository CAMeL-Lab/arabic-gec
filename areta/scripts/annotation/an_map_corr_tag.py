import os
import pandas as pd

global map_dict
dirname = os.path.dirname(__file__)

filename_map = os.path.join(dirname, '../../charmapping_new_format.csv')

data = pd.read_csv(filename_map)
map_dict = dict(zip(data.Result, data.Description))


def get_single_operation_label(w1, w2, operation):
    global map_dict
    if operation[0] == "replace":
        return "replace: " + str(map_dict[w1[int(operation[1])]]) + "-->" + str(map_dict[w2[int(operation[2])]])
    if operation[0] == "delete":
        return "delete: " + str(map_dict[w1[int(operation[1])]])
    if operation[0] == "insert":
        return "insert: " + str(map_dict[w2[int(operation[2])]])


def get_all_operations_labels(operations, w1, w2):
    operations_labels = []
    for op in operations:
        operations_labels.append(get_single_operation_label(w1, w2, op))
    return operations_labels
