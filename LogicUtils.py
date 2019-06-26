import pandas
import numpy as np
import git
from LogicTypes import *


class OA:
    def __init__(self, name, array, num_of_att, is_optimal):
        self.name = name
        self.array = array
        self.num_of_att = num_of_att
        self.is_optimal = is_optimal


def read_oa(path):
    with open(path) as oa_file:
        oa_array = oa_file.read().split('\n').copy()
    oa_num_of_att = len(oa_array[0])
    splitted_path = path.split('\\')
    new_oa = OA(splitted_path[len(splitted_path) - 1][:len(splitted_path[len(splitted_path) - 1]) - len('.txt')],
                oa_array,
                oa_num_of_att,
                0 if splitted_path[len(splitted_path) - 1].startswith('GEN') else 1)
    return new_oa


def trim_oa_to_fit_inputs(oa, input_size):
    trimmed_oa_array = [oa_line[:input_size] for oa_line in oa.array.copy()]
    return OA(oa.name, list(set(trimmed_oa_array)), input_size, oa.is_optimal)


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


def is_transformed_column_exist(data_inputs, column):
    if data_inputs is not None:
        col_series = pandas.Series(column)
        values = data_inputs.values
        col = col_series.values.reshape(len(col_series), 1)
        eq_df = (values == col).all(axis=0)
        return len(np.where(eq_df == True)[0]) > 0
    return False


def get_transformed_att_value_cache_enabled(orig_data_inputs, orig_cached_full_data, new_gate_feature, data, generate_not_gate_for_cache):
    new_attribute_name = new_gate_feature.to_string()
    if not orig_cached_full_data.__contains__(new_attribute_name):
        transformed_column = get_transformed_att_value(orig_cached_full_data, new_gate_feature.inputs, new_gate_feature.gate)
        if is_transformed_column_exist(orig_data_inputs, transformed_column):
            return None
        else:
            orig_cached_full_data.insert(len(orig_cached_full_data.columns), new_attribute_name, transformed_column)
            # for caching not gate
            if generate_not_gate_for_cache:
                get_transformed_att_value_cache_enabled(orig_data_inputs, orig_cached_full_data, GateFeature(OneNot, [new_gate_feature]), data, False)
    transformed_column = orig_cached_full_data.iloc[data.index.values][new_attribute_name]
    return transformed_column


def get_input_names(data):
    return [str(col) for col in data.columns[pandas.Series(data.columns).str.startswith('i')]
            if (col[0] == 'i' and col[1].isdigit())]


def get_output_names(data):
    out_col = []
    for col in data.columns:
        if col[0] == 'o' and col[1].isdigit():
            out_col.append(col)
    return out_col


def get_nearest_oa(oa_list, value):
    possible_att_values = [oa.num_of_att for oa in oa_list]
    possible_att_values = np.asarray(possible_att_values)

    diff_array = possible_att_values - value
    diff_array[diff_array < 0] = 2147483647  # get only att higher than value
    nearest_value_index = diff_array.argmin()
    if diff_array[nearest_value_index] == 2147483647:
        print('not enough attributes in current strength. Possible: ' + str(possible_att_values) + ', given: ' + str(value))
        return None
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
    return metric[entry] if (metric.get(entry) is not None) else 0
