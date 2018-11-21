import operator
import pandas
import itertools
import time
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from LogicUtils import *


class OneNot:
    name = "not"
    numInputs = 1
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value):
        return not value


class TwoXor:
    name = "2xor"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return operator.xor(value1, value2)


class TwoAnd:
    name = "2and"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return operator.and_(value1, value2)


class TwoOr:
    name = "2or"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return operator.or_(value1, value2)


class GateFeature:
    def __init__(self, gate, inputs):
        self.gate = gate
        self.inputs = []
        if isinstance(inputs, GateFeature.__class__):
            self.inputs.append(inputs)
        else:
            for input in inputs:
                self.inputs.append(input)

    def to_string(self):
        to_string = ""
        if self.gate is not None:
            to_string += str(self.gate.name) + "("
            for input_index in range(len(self.inputs)):
                if input_index > 0: # add comma for all inputs but the first input
                    to_string += ","
                if isinstance(self.inputs[input_index], str): # only argument
                    to_string += str(self.inputs[input_index])
                else:
                    to_string += str(self.inputs[input_index].to_string())
            to_string += ")"
        else: # only argument
            return self.inputs[0]

        return to_string

    def __eq__(self, other):
        return self.to_string().__eq__(other.to_string())


class OA:
    def __init__(self, array, num_of_att):
        self.array = array
        self.num_of_att = num_of_att


def preprocess(data): # maybe take it off aftre refactor!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    data.replace("F", False, inplace=True)
    data.replace("T", True, inplace=True)


def insert_gate_as_att(data, gate_feature, max_input_index):
    new_attribute_name = gate_feature.to_string()
    new_column = get_transformed_att_value(data, gate_feature.inputs, gate_feature.gate)
    data.insert(max_input_index, new_attribute_name, new_column)
    max_input_index += 1
    return max_input_index


def print_tree_to_code(tree, curr_gate_features):
    tree_ = tree.tree_
    tree_feature_names = get_tree_feature_names(curr_gate_features, tree)
    # print ("def tree({}):".format(", ".join(curr_gate_features))) # should be refactored if needed

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = tree_feature_names[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


def get_tree_feature_names(gate_features, tree):
    tree_ = tree.tree_
    feature_name = [
        gate_features[i].to_string() if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    return feature_name


def get_total_gates(gates_map):
    total_gates = 0
    for gate_usage in gates_map.values():
        total_gates += gate_usage
    return total_gates


def is_better_combination(new_attribute_name, best_combination):
    gates_map = get_gates_map(new_attribute_name)
    return get_total_gates(gates_map) < get_total_gates(best_combination)


def get_gates_map(attribute_name):
    gates_map = {}
    parenthesis_index = attribute_name.find("(")
    while parenthesis_index != -1:
        start_index = 0
        comma_index = attribute_name.find(",",0,parenthesis_index)
        if comma_index != -1: # comma leftover from previous iteration
            start_index = comma_index + 1
        gate = attribute_name[start_index:parenthesis_index]
        if not gates_map.__contains__(gate):
            gates_map[gate] = 0
        gates_map[gate] += 1
        attribute_name = attribute_name[parenthesis_index + 1:]
        parenthesis_index = attribute_name.find("(")
    return gates_map


def matched_oa(data_row, oa_row):
    min_len = min(len(data_row), len(oa_row)) # go over oa and data only as far as the min length of either one
    for column_index in range(min_len):
        if not isinstance(data_row[column_index], str):
            value = "1" if data_row[column_index] else "0"
        else:
            value = data_row[column_index]
        if oa_row[column_index] != value:
            return False
    return True


def process_oa(curr_oa, gate_features_inputs, next_oa, values_to_explore):
    # 1. filter next oa of curr oa because older instances shouldn't be filtered,
    # therefore no use of going over a row in next oa that is already included in curr oa
    if curr_oa is not None:
        updated_next_oa_array = []
        for next_oa_row_index in range(len(next_oa.array)):
            matched = False
            for curr_oa_row_index in range(len(curr_oa)):
                if matched_oa(curr_oa[curr_oa_row_index], next_oa.array[next_oa_row_index]):
                    matched = True
                    break
            if not matched:
                updated_next_oa_array.append(next_oa.array[next_oa_row_index])
    else: # there was no previous OA, first iteration
        curr_oa = []
        updated_next_oa_array = next_oa.array

    # 2. apply values to explore filter over next oa, make sure there are enough att as values,
    # if not use the first values as they are ordered from root to leaf
    # for every value look for its position in the feature gates collection and apply oa filter at the same position
    if len(values_to_explore) > 0:
        if len(values_to_explore) > len(updated_next_oa_array[0]):
            values_to_explore = values_to_explore.copy()[:len(updated_next_oa_array[0])]
            print("******** WARN - Orthogonal array has less attributes than inputs(values to explore) ********")
        # going over each input and placing an explorable value if exist or any 0 or 1 value if it doesn't exist
        explorable_regex = '^'
        #feature_input_index = 0
        max_att = min(len(gate_features_inputs), len(updated_next_oa_array[0]))
        for feature_input_index in range(max_att):
            value_to_explore = get_value_to_explore(gate_features_inputs[feature_input_index], values_to_explore)
            if value_to_explore is not None:
                explorable_regex += str(value_to_explore[1]) # appends the node's value
            else:
                explorable_regex += '[0-1]'
        explorable_regex += '.*$'
        compiled_regex = re.compile(explorable_regex)
        updated_next_oa_array = list(filter(compiled_regex.match, updated_next_oa_array))

    # 3. merge curr oa with processed next oa
    return list(set(curr_oa + updated_next_oa_array))


def get_value_to_explore(value, values_to_explore):
    for value_to_explore in values_to_explore:
        if value.__eq__(value_to_explore[0]):
            return value_to_explore
    return None


# def get_oa(num_of_data_inputs, oa, values_to_explore):
#     num_of_oa_att = min(num_of_data_inputs, oa.num_of_att)
#     values_to_explore = values_to_explore[:num_of_oa_att] # cutting values_to_explore to the current number of OA attributes
#
#     # cleaning current OA from rows not matching the desired values to explore
#     curr_oa = oa.array.copy()
#     if values_to_explore != "":
#         updated_curr_oa = []
#         for oa_row_index in range(len(curr_oa)):
#             if matched_oa(values_to_explore, curr_oa[oa_row_index]):
#                 updated_curr_oa.append(curr_oa[oa_row_index])
#         curr_oa = updated_curr_oa
#     return curr_oa


def get_batch_using_oa(data, max_input_index, oa):
    # if max_input_index > len(oa[0]):
    #     print("******** WARN - Orthogonal array has less attributes than inputs ********")

    data_batch = pandas.DataFrame(columns=data.columns.values, dtype=bool)

    for row_index in range(len(data)):
        for oa_row_index in range(len(oa)):
            if matched_oa(data.loc[row_index][:max_input_index], oa[oa_row_index]):
                data_batch = data_batch.append(pandas.DataFrame([data.loc[row_index]]), ignore_index=True)
                break
    return data_batch


def apply_new_attributes(data, max_input_index, combinations_and_gates, start_index, end_index):
    for combination_gate_tuple_index in range(start_index, end_index):
        max_input_index = insert_gate_as_att(data,combinations_and_gates[combination_gate_tuple_index], max_input_index)
    return data


def generate_gate_features(inputs):
    gate_features = []
    for input in inputs:
        gate_features.append(GateFeature(None,[input]))
    return gate_features

def get_curr_values_to_explore(tree, curr_gate_feature_inputs):
    values_to_explore = []
    tree_ = tree.tree_

    nodes_impurity = tree_.impurity
    max_impurity = max(nodes_impurity)
    if max_impurity != 0:
        gate_feature_input_list = []
        for node_index in range(len(nodes_impurity)):
            # in case there are several nodes with same entropy value
            if max_impurity == nodes_impurity[node_index]:
                explorable_node_index = node_index
                break # don't take all equal entropies, only the first one for consistency

        values_to_explore = extract_path_to_explorable_node(curr_gate_feature_inputs, tree_, node_index)

    return values_to_explore

def extract_path_to_explorable_node(curr_gate_feature_inputs, tree, node_index):
    path = []

    def recursively_find_path(path, curr_gate_feature_inputs, tree, node_index):
        if node_index == 0:
            return path
        parent_index = get_parent_index(tree.children_left, node_index)
        parent_value = 0
        if parent_index == -1:
            parent_index = get_parent_index(tree.children_right, node_index)
            parent_value = 1

        # converting 'not' gates to their original gate and alternating their value for consistency
        feature_gate = curr_gate_feature_inputs[tree.feature[parent_index]]
        while isinstance(feature_gate.gate, OneNot.__class__):
            feature_gate = feature_gate.inputs[0] # Not gate has only one input
            parent_value = (not parent_value).__int__()

        # append parent at the beginning to path
        path.insert(0, (feature_gate, parent_value))
        recursively_find_path(path, curr_gate_feature_inputs, tree, parent_index)

    recursively_find_path(path, curr_gate_feature_inputs, tree, node_index)
    return path

def get_parent_index(children_list, node_index):
    for parent_index in range(len(children_list)):
        if children_list[parent_index] == node_index:
            return parent_index
    return -1


def merge_values(curr_values_to_explore, new_values_to_explore):
    merged_values = curr_values_to_explore.copy()
    for new_value in new_values_to_explore:
        value_exists = False
        for curr_value in curr_values_to_explore:
            if new_value.__eq__(curr_value):
                value_exists = True
                break
        if not value_exists:
            merged_values.append(new_value)
    return merged_values


def init_orthogonal_arrays(use_orthogonal_arrays):

    def put_oa(path, strength_index, oa_by_strength_map):
        with open(path) as oa_file:
            oa_array = oa_file.read().split('\n').copy()
            oa_num_of_att = len(oa_array[0])
            oa_by_strength_map[strength_index] = OA(oa_array, oa_num_of_att)

    oa_by_strength_map = {}
    if use_orthogonal_arrays:
        oa_path = "E:\\ise_masters\\gal_thesis\\data_sets\\oa\\"
        put_oa(oa_path + "strength_2\\11_att.txt", 2, oa_by_strength_map)
        put_oa(oa_path + "strength_3\\12_att.txt", 3, oa_by_strength_map)
    return oa_by_strength_map


def trim_oa_to_fit_inputs(oa, input_size):
    trimmed_oa_array = [oa_line[:input_size] for oa_line in oa.array.copy()]
    return OA(trimmed_oa_array, input_size)


if __name__ == '__main__':

    #######################
    # parameters
    #######################
    # fileName = "E:\\ise_masters\\gal_thesis\\data_sets\\full_adder.tab"
    fileName = TRUTH_TABLE_PATH_74182

    possible_gates = [TwoOr, TwoAnd, OneNot, TwoXor]
    subset_min = 1
    subset_max = 2
    num_of_iterations = 10
    remove_unused_attributes = False  # should be used to boost computational complexity
    use_orthogonal_arrays = True
    random_batch_size = 70
    use_explore_nodes = False
    min_oa_strength = 2
    max_oa_strength = 3
    print("Working on:" + fileName)

    start_time = time.time()
    orig_data = pandas.read_csv(fileName, delimiter='\t', header=0)
    preprocess(orig_data)

    outputs = get_output_names(orig_data)
    number_of_outputs = len(outputs)
    print(str(number_of_outputs) + " outputs: " + str(outputs))

    orig_max_input_index = len(orig_data.columns.values) - number_of_outputs
    inputs = orig_data.columns[:orig_max_input_index]
    oa_by_strength_map = init_orthogonal_arrays(use_orthogonal_arrays)


    print ("Initial data:")
    print (orig_data)
    print ("##################")

    induced = 0
    gate_features_inputs = generate_gate_features(inputs) # init the starting inputs as feature gates for later process in the tree nodes
    curr_oa = None
    values_to_explore = []

    # FIRST ADD THE NOT GATE FOR ALL CURRENT INPUT
    not_gate_arguments = itertools.combinations(gate_features_inputs, 1)
    max_input_index = orig_max_input_index
    for not_gate_argument_tuple in not_gate_arguments:
        gate_feature = GateFeature(OneNot, [not_gate_argument_tuple[0]])
        max_input_index = insert_gate_as_att(orig_data, gate_feature, max_input_index)
        gate_features_inputs.append(gate_feature)

    if not use_orthogonal_arrays:# ??????? needed?
        data = orig_data

    while induced < num_of_iterations:
        best_quality = 10000
        best_attribute_name = ""
        best_attribute_gates_map = {}
        best_gate_feature = None

        print("\n**** iteration #: " + str(induced) + " ****\n")
        if use_orthogonal_arrays:
            next_strength = induced + min_oa_strength
            if oa_by_strength_map.__contains__(next_strength):
                if induced != 0:
                    orig_data = apply_new_attributes(orig_data, orig_max_input_index, gate_features_inputs, orig_max_input_index, orig_max_input_index + 1)
                    orig_max_input_index += 1
                next_oa = trim_oa_to_fit_inputs(oa_by_strength_map[next_strength], orig_max_input_index)
                curr_oa = process_oa(curr_oa, gate_features_inputs, next_oa, values_to_explore)
                data = get_batch_using_oa(orig_data, orig_max_input_index, curr_oa)
            else:
                data = apply_new_attributes(data, orig_max_input_index, gate_features_inputs, orig_max_input_index, orig_max_input_index + 1)
                orig_max_input_index += 1
        else: #Random Batch
            # take new sample out of the rest of the data
            new_sample = orig_data.sample(n=random_batch_size, random_state=2018)
            # apply over the new sample the added features from all previous iterations
            new_sample = apply_new_attributes(new_sample, orig_max_input_index, gate_features_inputs, orig_max_input_index, orig_max_input_index + 1)
            # apply over the previous data the last new added feature
            data = apply_new_attributes(data, orig_max_input_index, gate_features_inputs, orig_max_input_index, orig_max_input_index + 1)
            data = pandas.concat([data, new_sample])

        # ADD NOT GATE !AT THE END! FOR NEW ADDED ARG FROM LAST ITERATION
        if induced != 0:
            max_input_index = len(data.columns.values) - number_of_outputs
            not_gate_new_arg_tuple = (data.columns[orig_max_input_index - 1],)
            gate_feature = GateFeature(OneNot, [gate_features_inputs[orig_max_input_index - 1]])
            max_input_index = insert_gate_as_att(data, gate_feature, max_input_index)
            gate_features_inputs.append(gate_feature)

        print('**** # of instances: ' + str(len(data)) + ' ****')

        # Now working on all other gates
        possible_arguments_combinations = []

        #appends all combinations of different sizes of subsets
        for i in range(max(subset_min, 2), subset_max + 1):
            possible_arguments_combinations += itertools.combinations(gate_features_inputs, i)

        #for each combination
        for curr_arg_combination in possible_arguments_combinations:
            #for each possiible gate
            for possible_gate in possible_gates:
                if possible_gate.numInputs == len(curr_arg_combination):
                    new_gate_feature = GateFeature(possible_gate, curr_arg_combination)
                    new_attribute_name = new_gate_feature.to_string()
                    if (not data.__contains__(new_attribute_name)):
                        new_column = get_transformed_att_value(data, curr_arg_combination, possible_gate)
                        data.insert(orig_max_input_index, new_attribute_name, new_column)

                        tree_quality = 0
                        inputs = data.columns[:max_input_index + 1]
                        fitted_trees = {}
                        for output_index in range(len(outputs)):
                            tree_data = DecisionTreeClassifier(random_state=0, criterion="entropy")
                            tree_data.fit(data[inputs], data[outputs[output_index]])
                            tree_quality += tree_data.tree_.node_count
                            fitted_trees[outputs[output_index]] = tree_data

                        # print("checking "  + str(new_attribute_name) + " complexity over all trees is: " + str(tree_quality))
                        if (tree_quality < best_quality) or ((tree_quality == best_quality) and (is_better_combination(new_attribute_name, best_attribute_gates_map))):
                            best_quality = tree_quality
                            best_gate_feature = new_gate_feature
                            best_attribute_name = new_attribute_name
                            best_attribute_gates_map = get_gates_map(best_attribute_name)
                            #best_trees_input_dump = inputs
                            best_trees_dump = fitted_trees
                        del data[new_attribute_name]

        # Now check if there is something new to add
        if best_quality < 10000:
            induced += 1
            #data = best_attribute_table
            gate_features_inputs.insert(orig_max_input_index, best_gate_feature)
            print ("============ Selected Gate is: %s" % (best_attribute_name))
            values_to_explore = [] # node values ordered from root to leaf
            for output, tree in best_trees_dump.items():
                print("\ntree for " + str(output))
                print_tree_to_code(tree, gate_features_inputs)
                if use_explore_nodes:
                    values_to_explore = merge_values(values_to_explore, get_curr_values_to_explore(tree, gate_features_inputs))

            print ("=============")
            # print ("The new data domain is: %s" % (data.columns.values))
            print("=============")


        if not use_orthogonal_arrays and best_quality <= 3 * number_of_outputs:
            #decision stump
            break

        # should be refactored if needed!!!
        # if remove_unused_attributes:
        #     feature_set = set()
        #     for output, tree in best_trees_dump.items():
        #         feature_set.update(set(get_tree_feature_names(best_trees_input_dump, tree)))
        #     temp_data = data
        #     for curr_att in data.columns.values:
        #         if (not feature_set.__contains__(curr_att)):
        #             if (curr_att[0] != "_") and (curr_att != best_attribute_name):
        #                 del temp_data[curr_att]
        #     data = temp_data
        #     print ("=======> The new ***reduced**** data domain is: %s" % (data.columns.values))

    end_time = time.time()
    print("took: %.5f sec" % (end_time - start_time))
