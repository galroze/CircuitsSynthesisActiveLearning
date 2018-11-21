import operator
import pandas
import itertools
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

class onenot:
    name = "not"
    numInputs = 1
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value):
        return not (value)


class twoxor:
    name = "2xor"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return operator.xor(value1, value2)


class twoand:
    name = "2and"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return operator.and_(value1, value2)


class twoor:
    name = "2or"
    numInputs = 2
    numOutputs = 1
    cost = 1

    @staticmethod
    def f(value1, value2):
        return operator.or_(value1, value2)

def transform_att_value (data, att_names, gate):
    first_value = True
    for att in att_names:
        if first_value:
            att_val_column = "gate.f(" + data[att].replace(False,"False").replace(True,"True")
            first_value = False
        else:
            att_val_column += "," + data[att].replace(False,"False").replace(True,"True")
    att_val_column += ")"

    transformed_column = []
    for att_val in att_val_column:
        tmp = eval(att_val)
        if tmp:
            transformed_column.append(True)
        else:
            transformed_column.append(False)
    return transformed_column


def get_argument_combination_str(argument_combination):
    arg_comb_str = ""
    for i in range(len(argument_combination)):
        if arg_comb_str == "":
            arg_comb_str += "(" + argument_combination[i]
        else:
            arg_comb_str += "," + argument_combination[i]
    arg_comb_str += ")"
    return arg_comb_str

def preprocess(data):
    data.replace("F", False, inplace=True)
    data.replace("T", True, inplace=True)


def insert_arg_not_gate(data, not_gate_new_arg_tuple, max_input_index):
    new_attribute_name = onenot.name + get_argument_combination_str(not_gate_new_arg_tuple)
    new_not_column = transform_att_value(data, not_gate_new_arg_tuple, onenot)
    data.insert(max_input_index, new_attribute_name, new_not_column)
    max_input_index += 1
    return max_input_index

def print_tree_to_code(tree, all_feature_names):
    tree_ = tree.tree_
    tree_feature_names = get_tree_feature_names(all_feature_names, tree)
    print ("def tree({}):".format(", ".join(all_feature_names)))

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


def get_tree_feature_names(feature_names, tree):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    return feature_name

if __name__ == '__main__':

    #######################
    # parameters
    #######################
    fileName = "E:\\ise_masters\\gal_thesis\\data_sets\\substractor_all_no_poss_values.tab"
    # fileName = "C:\\git\\prediction_server\\mish\\substractor_all_no_poss_values.tab"

    possible_gates = [twoor, twoand, onenot, twoxor]
    subset_min = 1
    subset_max = 2
    num_of_terations = 10
    remove_unused_attributes = True  # should be used to boost computational complexity

    print("Working on:" + fileName)

    data=pandas.read_csv(fileName,delimiter='\t',header=0)
    preprocess(data)
    print ("Initial data:")
    print (data)
    print ("##################")

    number_of_outputs = 0
    headers = list(data.columns.values)
    outputs = []
    for att in headers:
        if (att[0] == '_'):
            number_of_outputs += 1
            outputs.append(att)
    print ("number of outputs: " + str(number_of_outputs) + " : " + str(outputs))

    induced = 0
    best_quality = 10000
    best_attribute_name_len = 1000

    # FIRST ADD THE NOT GATE FOR ALL CURRENT INPUT
    max_input_index = len(data.columns.values) - number_of_outputs
    inputs = data.columns[:max_input_index]
    not_gate_arguments = itertools.combinations(inputs, 1)
    for not_gate_argument_tuple in not_gate_arguments:
        max_input_index = insert_arg_not_gate(data, not_gate_argument_tuple, max_input_index)

    while (induced < num_of_terations):

        print("\n**** iteration #: " + str(induced) + " ****\n")
        #ADD NOT GATE FOR NEW ADDED ARG FROM LAST ITERATION
        if (induced != 0):
            max_input_index = len(data.columns.values) - number_of_outputs
            not_gate_new_arg_tuple = (data.columns[max_input_index - 1],)
            max_input_index = insert_arg_not_gate(data, not_gate_new_arg_tuple, max_input_index)

        # Now working on all other gates
        inputs = data.columns[:max_input_index]
        possible_arguments_combinations = []

        #appends all combinations of different sizes of subsets
        for i in range(max(subset_min, 2), subset_max + 1):
            possible_arguments_combinations += itertools.combinations(inputs, i)

        #for each combination
        for curr_arg_combination in possible_arguments_combinations:
            #for each possiible gate
            for possible_gate in possible_gates:
                if possible_gate.numInputs == len(curr_arg_combination):
                    new_attribute_name = possible_gate.name + get_argument_combination_str(curr_arg_combination)
                    if (not data.__contains__(new_attribute_name)):
                        new_column = transform_att_value(data, curr_arg_combination, possible_gate)
                        data.insert(max_input_index, new_attribute_name, new_column)

                        tree_quality = 0
                        inputs = data.columns[:max_input_index + 1]
                        tree_str = ""
                        fitted_trees = {}
                        for output_index in range(len(outputs)):
                            tree_data = DecisionTreeClassifier()
                            tree_data.fit(data[inputs], data[outputs[output_index]])
                            total_nodes = tree_data.tree_.node_count
                            tree_quality += total_nodes
                            fitted_trees[outputs[output_index]] = tree_data

                        # print("checking "  + str(new_attribute_name) + " complexity over all trees is: " + str(tree_quality))
                        if (tree_quality < best_quality) | (
                                (len(new_attribute_name) < best_attribute_name_len) & (tree_quality == best_quality)):
                            best_quality = tree_quality
                            best_attribute_name = new_attribute_name
                            best_attribute_name_len = len(new_attribute_name)
                            best_attribute_table = data.copy()
                            best_trees_input_dump = inputs
                            best_trees_dump = fitted_trees
                        del data[new_attribute_name]

        # Now check if there is something new to add
        if best_quality < 10000:
            induced += 1
            data = best_attribute_table
            print ("============ Selected Gate is: %s" % (best_attribute_name))
            for output, tree in best_trees_dump.items():
                print("tree for " + str(output))
                print_tree_to_code(tree, best_trees_input_dump)
            print ("=============")
            print ("The new data domain is: %s" % (data.columns.values))
            print("=============")
        if best_quality <= 3 * number_of_outputs:
            #decision stump
            break

        if remove_unused_attributes:
            feature_set = set()
            for output, tree in best_trees_dump.items():
                feature_set.update(set(get_tree_feature_names(best_trees_input_dump, tree)))
            temp_data = data
            for curr_att in data.columns.values:
                if (not feature_set.__contains__(curr_att)):
                    if (curr_att[0] != "_") & (curr_att != best_attribute_name):
                        del temp_data[curr_att]
            data = temp_data
            print ("=======> The new ***reduced**** data domain is: %s" % (data.columns.values))