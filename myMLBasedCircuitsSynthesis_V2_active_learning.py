from os import listdir
from os.path import isfile, join
import itertools
import git
import time
import csv
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from LogicUtils import *
from LogicTypes import *
from dataSystemsUtil import generate_expected_gates_map
from Experiments.ExperimentServiceImpl import write_experiment

class ActiveLearningCircuitSynthesisConfiguration:

    def __init__(self, file_name, total_num_of_instances, possible_gates, subset_min, subset_max, max_num_of_iterations,
                 use_orthogonal_arrays, use_explore_nodes, randomize_remaining_data, random_batch_size, min_oa_strength):

        self.file_name = file_name
        self.total_num_of_instances = total_num_of_instances
        self.possible_gates = possible_gates
        self.subset_min = subset_min
        self.subset_max = subset_max
        self.max_num_of_iterations = max_num_of_iterations
        self.use_orthogonal_arrays = use_orthogonal_arrays
        self.use_explore_nodes = use_explore_nodes
        # when using OA and there are no more arrays to use, True for random, False for taking all remaining data
        self.randomize_remaining_data = randomize_remaining_data
        self.random_batch_size = random_batch_size
        self.min_oa_strength = min_oa_strength




def insert_gate_as_att(data, gate_feature, max_input_index):
    new_attribute_name = gate_feature.to_string()
    new_column = get_transformed_att_value(data, gate_feature.inputs, gate_feature.gate)
    data.insert(max_input_index, new_attribute_name, new_column)
    max_input_index += 1
    return max_input_index


def print_tree_to_code(tree, curr_gate_features):
    tree_ = tree.tree_
    tree_feature_names = get_tree_feature_names(curr_gate_features, tree)

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


def get_tree_features(gate_features, tree):
    tree_ = tree.tree_
    features = [
        gate_features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    return features


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


def process_oa(curr_oa, gate_features_inputs, next_oa, values_to_explore_by_tree):
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

    # 2. apply values to explore filter over next oa, make sure there are more or equal att as values to explore,
    # if not use the first values as they are ordered from root to leaf
    trees_joint_next_oa_array = []
    if len(values_to_explore_by_tree) > 0:
        at_least_one_value_exists = False
        for tree_values in values_to_explore_by_tree.values():
            if (len(tree_values) > 0) & (len(updated_next_oa_array) > 0):
                at_least_one_value_exists = True
                if len(tree_values) > len(updated_next_oa_array[0]):
                    tree_values = tree_values.copy()[:len(updated_next_oa_array[0])]
                    print("******** WARN - Orthogonal array has less attributes than inputs(values to explore) ********")
                oa_by_values_to_explore = get_data_by_values_to_explore(tree_values, gate_features_inputs, updated_next_oa_array)
                trees_joint_next_oa_array = list(set(trees_joint_next_oa_array) | set(oa_by_values_to_explore))
                updated_next_oa_array = [x for x in updated_next_oa_array if x not in oa_by_values_to_explore]
        if at_least_one_value_exists:
            updated_next_oa_array = trees_joint_next_oa_array

    # 3. merge curr oa with processed next oa
    return list(set(curr_oa + updated_next_oa_array))


def get_value_to_explore(value, values_to_explore):
    for value_to_explore in values_to_explore:
        if value.__eq__(value_to_explore[0]):
            return value_to_explore
    return None


def get_data_by_values_to_explore(values_to_explore, gate_features_inputs, data_list):
    # going over each input and placing an explorable value if exist or any 0 or 1 value if it doesn't exist
    # for every value look for its position in the feature gates collection and apply oa filter at the same position
    explorable_regex = '^'
    max_att = min(len(gate_features_inputs), len(data_list[0]))
    for feature_input_index in range(max_att):
        value_to_explore = get_value_to_explore(gate_features_inputs[feature_input_index], values_to_explore)
        if value_to_explore is not None:
            explorable_regex += str(value_to_explore[1])  # appends the node's value
        else:
            explorable_regex += '[0-1]'
    explorable_regex += '.*$'
    compiled_regex = re.compile(explorable_regex)
    return list(filter(compiled_regex.match, data_list))


def get_data_frame_by_values_to_explore(values_to_explore_by_tree, gate_features_inputs, data_frame):
    processed_data_frame = pandas.DataFrame(columns=data_frame.columns.values, dtype=bool)

    for tree_values in values_to_explore_by_tree.values():
        tmp_df = data_frame.copy()
        if len(tree_values) > 0:
            for k, v in tree_values:
                tmp_df = tmp_df.loc[tmp_df[k.to_string()] == (False if v == 0 else True)]

            processed_data_frame = pandas.concat([processed_data_frame, tmp_df])
            data_frame = data_frame.drop(tmp_df.index)
    return processed_data_frame


def get_batch_using_oa(data, max_input_index, oa):

    data_batch = pandas.DataFrame(columns=data.columns.values, dtype=bool)

    for row_index in range(len(data)):
        for oa_row_index in range(len(oa)):
            if matched_oa(data.loc[row_index][:max_input_index], oa[oa_row_index]):
                data_batch = data_batch.append(pandas.DataFrame([data.loc[row_index]]), ignore_index=True)
                break
    return data_batch


def apply_new_attributes(induced, data, max_input_index, combinations_and_gates, start_index, end_index):
    if induced > 0:
        for combination_gate_tuple_index in range(start_index, end_index):
            max_input_index = insert_gate_as_att(data,combinations_and_gates[combination_gate_tuple_index], max_input_index)
    return data, max_input_index


def generate_gate_features(inputs):
    gate_features = []
    for input in inputs:
        gate_features.append(GateFeature(None, [input]))
    return gate_features


def get_curr_values_to_explore(tree, curr_gate_feature_inputs):
    values_to_explore = []
    tree_ = tree.tree_

    nodes_impurity = tree_.impurity
    if len(nodes_impurity) > 1:
        max_impurity = max(nodes_impurity[1:]) # get max impurity from inner nodes and not the root
        if max_impurity != 0:
            gate_feature_input_list = []
            for node_index in range(1, len(nodes_impurity)):
                # in case there are several nodes with same entropy value
                if max_impurity == nodes_impurity[node_index]:
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
        while feature_gate.gate == OneNot:
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


def init_orthogonal_arrays(use_orthogonal_arrays):

    def put_oa(path, strength_index, oa_by_strength_map):
        new_oa = read_oa(path)
        if oa_by_strength_map.__contains__(strength_index):
            oa_by_strength_map[strength_index].append(new_oa)
        else:
            oa_by_strength_map[strength_index] = [new_oa]

    oa_by_strength_map = {}
    if use_orthogonal_arrays:
        oa_path = "E:\\ise_masters\\gal_thesis\\data_sets\\oa\\"
        oa_files_list = [f for f in listdir(oa_path) if isfile(join(oa_path, f))]
        for file_name in oa_files_list:
            put_oa(oa_path + file_name, get_strength(file_name), oa_by_strength_map)
    return oa_by_strength_map


def get_random_data(ALCS_conf, orig_data, curr_data, non_not_max_input_index, induced, gate_features_inputs,
                    values_to_explore_by_tree, number_of_outputs, should_randomize_remaining_data):
    if len(orig_data) > 0:
        # apply over the previous data the last new added feature
        curr_data, _ = apply_new_attributes(induced, curr_data, non_not_max_input_index, gate_features_inputs,
                                            non_not_max_input_index, non_not_max_input_index + 1)
        # Add last iteration's not gate
        orig_data, _ = apply_new_attributes(induced - 1, orig_data, len(orig_data.columns.values) - number_of_outputs,
                                            gate_features_inputs, len(gate_features_inputs) - 1, len(gate_features_inputs))
        # Add new added att
        orig_data, non_not_max_input_index = apply_new_attributes(induced, orig_data, non_not_max_input_index, gate_features_inputs,
                                                                  non_not_max_input_index, non_not_max_input_index + 1)
        tmp_data = orig_data
        if ALCS_conf.use_explore_nodes & len(values_to_explore_by_tree) > 0:
            tmp_data = get_data_frame_by_values_to_explore(values_to_explore_by_tree, gate_features_inputs, orig_data)
        # take new sample out of the rest of the data
        if should_randomize_remaining_data:  # Random batch
            batch_size = ALCS_conf.random_batch_size if len(tmp_data) > ALCS_conf.random_batch_size else len(tmp_data)
            if batch_size > 0:
                new_sample = tmp_data.sample(n=batch_size, random_state=2018)
                orig_data = orig_data.drop(new_sample.index)
                data = pandas.concat([curr_data, new_sample])
            else:  # no instances to sample from
                # apply over the previous data the last new added feature
                data = curr_data
        else:  # All
            data = pandas.concat([curr_data, tmp_data])
            orig_data = orig_data.drop(tmp_data.index)
    else:
        # apply over the previous data the last new added feature
        data, non_not_max_input_index = apply_new_attributes(induced, curr_data, non_not_max_input_index,
                                                gate_features_inputs, non_not_max_input_index, non_not_max_input_index + 1)
    return orig_data, data, non_not_max_input_index


def get_data_for_iteration(ALCS_conf, orig_data, curr_data, non_not_max_input_index, max_input_index, curr_oa, induced,
                           oa_by_strength_map, gate_features_inputs, values_to_explore_by_tree, number_of_outputs):
    oa_is_optimal = True
    if ALCS_conf.use_orthogonal_arrays:
        next_strength = str(induced + ALCS_conf.min_oa_strength)
        if oa_by_strength_map.__contains__(next_strength):
            # Add last iteration's not gate
            orig_data, _ = apply_new_attributes(induced - 1, orig_data, len(orig_data.columns.values) - number_of_outputs,
                                                gate_features_inputs, len(gate_features_inputs) - 1, len(gate_features_inputs))
            orig_data, non_not_max_input_index = apply_new_attributes(induced, orig_data, non_not_max_input_index, gate_features_inputs,
                                             non_not_max_input_index, non_not_max_input_index + 1)
            nearest_oa = get_nearest_oa(oa_by_strength_map[next_strength], non_not_max_input_index)
            oa_is_optimal = nearest_oa.is_optimal
            next_oa = trim_oa_to_fit_inputs(nearest_oa, non_not_max_input_index)
            curr_oa = process_oa(curr_oa, gate_features_inputs, next_oa, values_to_explore_by_tree)
            data = get_batch_using_oa(orig_data, non_not_max_input_index, curr_oa)
        else:
            if len(curr_oa) > 0:  # first time to get data without OA
                curr_data = get_batch_using_oa(orig_data, non_not_max_input_index, curr_oa)
                curr_oa = []  # emptying list to not use it in next iterations anymore
                orig_data = orig_data.drop(curr_data.index)
                # Add last iteration's not gate
                curr_data, _ = apply_new_attributes(induced - 1, curr_data,
                                                    len(curr_data.columns.values) - number_of_outputs,
                                                    gate_features_inputs, len(gate_features_inputs) - 1,
                                                    len(gate_features_inputs))

            orig_data, data, non_not_max_input_index = get_random_data(ALCS_conf, orig_data, curr_data,
                            non_not_max_input_index, induced, gate_features_inputs, values_to_explore_by_tree,
                            number_of_outputs, ALCS_conf.randomize_remaining_data)

    else:  # Random Batch
        orig_data, data, non_not_max_input_index = get_random_data(ALCS_conf, orig_data, curr_data, non_not_max_input_index,
                                                       induced, gate_features_inputs, values_to_explore_by_tree, number_of_outputs, False)

    # ADD NOT GATE !AT THE END! FOR NEW ADDED ARG FROM LAST ITERATION
    if induced != 0:
        max_input_index = len(data.columns.values) - number_of_outputs
        gate_feature = GateFeature(OneNot, [gate_features_inputs[non_not_max_input_index - 1]])
        max_input_index = insert_gate_as_att(data, gate_feature, max_input_index)
        gate_features_inputs.append(gate_feature)

    return orig_data, data, non_not_max_input_index, max_input_index, gate_features_inputs, curr_oa, oa_is_optimal


def run_ALCS(ALCS_configuration, orig_data, oa_by_strength_map, expected_gates_map):

    outputs = get_output_names(orig_data)
    number_of_outputs = len(outputs)

    orig_max_input_index = len(orig_data.columns.values) - number_of_outputs
    inputs = orig_data.columns[:orig_max_input_index]

    print(str(orig_max_input_index) + " intputs: " + str(inputs) + ", \n" + str(number_of_outputs) + " outputs: " + str(outputs))

    print("Initial data:")
    print(orig_data)
    print("##################")

    induced = 0
    gate_features_inputs = generate_gate_features(inputs) # init the starting inputs as feature gates for later process in the tree nodes
    curr_oa = None
    values_to_explore_by_tree = {}
    metrics_by_iteration = {}

    # FIRST ADD THE NOT GATE FOR ALL CURRENT INPUT
    not_gate_arguments = itertools.combinations(gate_features_inputs, 1)
    max_input_index = orig_max_input_index
    non_not_max_input_index = orig_max_input_index
    for not_gate_argument_tuple in not_gate_arguments:
        gate_feature = GateFeature(OneNot, [not_gate_argument_tuple[0]])
        max_input_index = insert_gate_as_att(orig_data, gate_feature, max_input_index)
        gate_features_inputs.append(gate_feature)

    data = pandas.DataFrame(columns=orig_data.columns.values, dtype=bool)

    while induced < ALCS_configuration.max_num_of_iterations:
        best_quality = 10000
        best_attribute_name = ""
        best_attribute_gates_map = {}
        best_gate_feature = None

        orig_data, data, non_not_max_input_index, max_input_index, gate_features_inputs, curr_oa, oa_is_optimal = \
            get_data_for_iteration(ALCS_configuration, orig_data, data, non_not_max_input_index, max_input_index, curr_oa,
                                   induced, oa_by_strength_map, gate_features_inputs, values_to_explore_by_tree, number_of_outputs)
        num_of_insances = len(data)
        print("\n**** iteration #: " + str(induced) + ", " + "# of instances: " + str(num_of_insances) + " ****\n")

        # Now working on all other gates
        possible_arguments_combinations = []

        #appends all combinations of different sizes of subsets
        for i in range(max(ALCS_configuration.subset_min, 2), ALCS_configuration.subset_max + 1):
            possible_arguments_combinations += itertools.combinations(gate_features_inputs, i)

        #for each combination
        for curr_arg_combination in possible_arguments_combinations:
            #for each possible gate
            for possible_gate in ALCS_configuration.possible_gates:
                if possible_gate.numInputs == len(curr_arg_combination):
                    new_gate_feature = GateFeature(possible_gate, list(curr_arg_combination))
                    new_attribute_name = new_gate_feature.to_string()
                    if (not data.__contains__(new_attribute_name)):
                        new_column = get_transformed_att_value(data, curr_arg_combination, possible_gate)
                        data.insert(non_not_max_input_index, new_attribute_name, new_column)

                        tree_quality = 0
                        inputs = data.columns[:max_input_index + 1]
                        fitted_trees = {}
                        # fit_start = time.time()
                        for output_index in range(len(outputs)):
                            tree_data = DecisionTreeClassifier(random_state=0, criterion="entropy")
                            tree_data.fit(data[inputs], data[outputs[output_index]])
                            tree_quality += tree_data.tree_.node_count
                            fitted_trees[outputs[output_index]] = tree_data

                        if (tree_quality < best_quality) or ((tree_quality == best_quality) and (is_better_combination(new_attribute_name, best_attribute_gates_map))):
                            best_quality = tree_quality
                            best_gate_feature = new_gate_feature
                            best_attribute_name = new_attribute_name
                            best_attribute_gates_map = get_gates_map(best_attribute_name)
                            best_trees_dump = fitted_trees
                        del data[new_attribute_name]
        # Now check if there is something new to add
        if best_quality < 1000:
            induced += 1
            gate_features_inputs.insert(non_not_max_input_index, best_gate_feature)
            print ("============ Selected Gate is: %s" % (best_attribute_name))
            values_to_explore_dict = {} # node values ordered from root to leaf
            for output, tree in best_trees_dump.items():
                print("\ntree for " + str(output))
                print_tree_to_code(tree, gate_features_inputs)
                if ALCS_configuration.use_explore_nodes:
                    values_to_explore_by_tree[output] = get_curr_values_to_explore(tree, gate_features_inputs)
            print ("=============")
        else:
            raise Exception('Current iteration did not add any new attribute')


        sys_description = create_system_description(best_trees_dump, best_quality, number_of_outputs, gate_features_inputs)

        metrics_by_iteration[induced] = {'num_of_instances': num_of_insances,
                                        'sys_description': sys_description,
                                         'oa_is_optimal': oa_is_optimal}
        # get_component_distribution_metric(expected_gates_map, best_trees_dump, best_quality, number_of_outputs, gate_features_inputs)

    return metrics_by_iteration

def generate_gates_map(trees_dump, gate_features):
    curr_gates_map = {}
    # all trees are decision stumps, merge them all to distinct gates with inputs
    for output, tree in trees_dump.items():
        root_feature = get_tree_features(gate_features, tree)[0]  # there is only one feature in the root
        add_gates(curr_gates_map, root_feature, output)
    return curr_gates_map


def get_gate_from_set(gate, set):
    for item in set:
        if item == gate:
            return item


def add_gates(gates_map, gate, output):

    if gate != "undefined!": # all instances are of the same class - no tree at all
        curr_gate = gate.gate
        curr_gate_name = 'basic_inputs' if curr_gate is None else curr_gate.name

        if not gates_map.__contains__(curr_gate_name):
            gates_map[curr_gate_name] = set()

        if gate in gates_map[curr_gate_name]:
            existing_gate = get_gate_from_set(gate, gates_map[curr_gate_name])
            existing_gate.outputs.add(output)
            gates_map[curr_gate_name].add(existing_gate)
        else:
            gate.outputs.update([output])
            gates_map[curr_gate_name].add(gate)

        if curr_gate_name != 'basic_inputs':
            for curr_input in gate.inputs:
                add_gates(gates_map, curr_input, gate)


def create_system_description(best_trees_dump, best_quality, number_of_outputs, gate_features):

    vertices = 0
    edges = 0
    degree_distribution = {}
    comp_distribution_map = {}
    avg_vertex_degree = 0
    if best_quality <= 3 * number_of_outputs:
        curr_gates_map = generate_gates_map(best_trees_dump, gate_features)
        vertices += number_of_outputs
        degree_distribution[1] = number_of_outputs # init nodes of degree 1 for all outputs
        for gate_type, gate_type_set in curr_gates_map.items():
            vertices += len(gate_type_set)
            for gate in gate_type_set:
                edges += len(gate.outputs)
                curr_degree = len(gate.outputs) + (len(gate.inputs) if gate.gate is not None else 0)
                if not degree_distribution.__contains__(curr_degree):
                    degree_distribution[curr_degree] = 0
                degree_distribution[curr_degree] += 1
            if gate_type != 'basic_inputs':
                comp_distribution_map[gate_type] = len(gate_type_set)

        for degree, degree_count in degree_distribution.items():
            avg_vertex_degree += int(degree) * degree_count
        avg_vertex_degree /= sum(degree_distribution.values())


    return {'edges': edges, 'vertices': vertices, 'comp_distribution_map': comp_distribution_map, 'degree_distribution': degree_distribution, 'avg_vertex_degree': avg_vertex_degree}


def get_component_distribution_metric(curr_gates_map, expected_gates_map):
    comp_dist_metric_map = {}
    for expected_logic_gate, expected_gate_set in expected_gates_map.items():
        curr_gate_set = curr_gates_map.get(expected_logic_gate)
        comp_dist_metric_map[expected_logic_gate] = len(expected_gate_set) / len(
            curr_gate_set) if curr_gate_set is not None else 0

    for curr_gate, curr_gate_set in curr_gates_map.items():  # gate exist in current iteration but not in expected
        if not comp_dist_metric_map.__contains__(curr_gate):
            comp_dist_metric_map[curr_gate] = 0


if __name__ == '__main__':
    circuit_name = "c17"
    file_name = TRUTH_TABLE_PATH + circuit_name + ".tab"
    possible_gates = [OneNot, TwoXor, TwoAnd, TwoOr]

    orig_data = pandas.read_csv(file_name, delimiter='\t', header=0)
    ALCS_configuration = ActiveLearningCircuitSynthesisConfiguration(file_name=file_name, total_num_of_instances=len(orig_data),
                                    possible_gates=possible_gates, subset_min=1, subset_max=2, max_num_of_iterations=10,
                                    use_orthogonal_arrays=True, use_explore_nodes=True, randomize_remaining_data=True,
                                    random_batch_size=int(round(len(orig_data)*0.25)), min_oa_strength=2)

    print("Working on: " + ALCS_configuration.file_name)

    oa_by_strength_map = init_orthogonal_arrays(ALCS_configuration.use_orthogonal_arrays)
    expected_gates_map = generate_expected_gates_map(circuit_name)

    start_time = int(round(time.time() * 1000))
    metrics_by_iteration = run_ALCS(ALCS_configuration, orig_data, oa_by_strength_map, expected_gates_map)
    end_time = int(round(time.time() * 1000))

    run_time = (end_time - start_time)/1000
    print("took: %.5f sec" % run_time)

    # git version info
    repo = git.Repo(search_parent_directories=True)
    current_version = repo.head.object.hexsha
    version_time = repo.head.object.committed_datetime

    write_experiment(current_version, run_time, ALCS_configuration, metrics_by_iteration)