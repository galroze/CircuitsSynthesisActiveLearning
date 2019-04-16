import pandas
import numpy as np
import git


class OA:
    def __init__(self, array, num_of_att, is_optimal):
        self.array = array
        self.num_of_att = num_of_att
        self.is_optimal = is_optimal


def read_oa(path):
    with open(path) as oa_file:
        oa_array = oa_file.read().split('\n').copy()
    oa_num_of_att = len(oa_array[0])
    splitted_path = path.split('\\')
    new_oa = OA(oa_array, oa_num_of_att, 0 if splitted_path[len(splitted_path) - 1].startswith('GEN') else 1)
    return new_oa


def trim_oa_to_fit_inputs(oa, input_size):
    trimmed_oa_array = [oa_line[:input_size] for oa_line in oa.array.copy()]
    return OA(trimmed_oa_array, input_size, oa.is_optimal)


def get_transformed_att_value(data, att_names, gate):
    first_value = True
    for att in att_names:
        if not isinstance(att, str):
            att = att.to_string()
        if first_value:
            att_val_column = "gate.f(" + data[att].replace(False, "False").replace(True, "True")
            first_value = False
        else:
            att_val_column += "," + data[att].replace(False, "False").replace(True, "True")
    att_val_column += ")"

    transformed_column = []
    for att_val in att_val_column:
        tmp = eval(att_val)
        if tmp:
            transformed_column.append(True)
        else:
            transformed_column.append(False)
    return transformed_column


def get_input_names(data):
    return data.columns[pandas.Series(data.columns).str.startswith('i')]


def get_output_names(data):
    return data.columns[pandas.Series(data.columns).str.startswith('o')]


def get_nearest_oa(oa_list, value):
    possible_att_values = [oa.num_of_att for oa in oa_list]
    possible_att_values = np.asarray(possible_att_values)

    diff_array = possible_att_values - value
    diff_array[diff_array < 0] = 2147483647 # get only att higher than value
    nearest_value_index = diff_array.argmin()
    if nearest_value_index == 2147483647:
        raise ValueError('no OA found for data')
    return oa_list[nearest_value_index]


def get_strength(file_name):
    split_name = file_name.split('_')
    return split_name[len(split_name) - 1].split('.txt')[0]


def get_current_git_version():
    # git version info
    repo = git.Repo(search_parent_directories=True)
    current_version = repo.head.object.hexsha
    version_time = repo.head.object.committed_datetime

    return current_version


def get_metric_to_persist(metric, entry):
    return metric[entry] if ((metric.get(entry) is not None) or ((metric.get(entry) is not None) and (len(metric.get(entry)) > 0))) else -1
