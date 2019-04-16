from os import listdir
from os.path import isfile, join
import itertools
import time
import re
import datetime

from more_itertools import first
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from LogicUtils import *
from LogicTypes import *
from Experiments.ExperimentServiceImpl import write_experiment
from Experiments.ExperimentServiceImpl import write_iterations
from dataSystemsUtil import create_system_description


class ActiveLearningCircuitSynthesisConfiguration:

    def __init__(self, file_name, total_num_of_instances, possible_gates, subset_min, subset_max, max_num_of_iterations,
                 use_orthogonal_arrays, use_explore_nodes, randomize_remaining_data, random_batch_size,
                 pre_defined_random_size_per_iteration, min_oa_strength):

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
        self.pre_defined_random_size_per_iteration = pre_defined_random_size_per_iteration
        self.min_oa_strength = min_oa_strength


    def __str__(self):
        return '\n'.join('%s=%s' % item for item in vars(self).items())


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
            # console_file.write("{}if {} <= {}:".format(indent, name, threshold))
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            # console_file.write("{}else:  # if {} > {}".format(indent, name, threshold))
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            # console_file.write("{}return {}".format(indent, tree_.value[node]))
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
                data_batch = data_batch.append(pandas.DataFrame([data.loc[row_index]]), ignore_index=False)
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
                    values_to_explore_by_tree, number_of_outputs):
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
        if ALCS_conf.randomize_remaining_data:  # Random batch
            if len(ALCS_conf.pre_defined_random_size_per_iteration) > 0:
                if induced == 0:
                    batch_size = ALCS_conf.pre_defined_random_size_per_iteration[induced]
                else:
                    batch_size = ALCS_conf.pre_defined_random_size_per_iteration[induced] - ALCS_conf.pre_defined_random_size_per_iteration[induced - 1]
            else:
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
    oa_is_optimal = -1
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
                            non_not_max_input_index, induced, gate_features_inputs, values_to_explore_by_tree, number_of_outputs)

    else:  # Random Batch
        orig_data, data, non_not_max_input_index = get_random_data(ALCS_conf, orig_data, curr_data, non_not_max_input_index,
                                                       induced, gate_features_inputs, values_to_explore_by_tree, number_of_outputs)

    # ADD NOT GATE !AT THE END! FOR NEW ADDED ARG FROM LAST ITERATION
    if induced != 0:
        max_input_index = len(data.columns.values) - number_of_outputs
        gate_feature = GateFeature(OneNot, [gate_features_inputs[non_not_max_input_index - 1]])
        max_input_index = insert_gate_as_att(data, gate_feature, max_input_index)
        gate_features_inputs.append(gate_feature)

    return orig_data, data, non_not_max_input_index, max_input_index, gate_features_inputs, curr_oa, oa_is_optimal


def get_model_error(best_trees_dump, data, outputs):
    error = 0
    for output, tree in best_trees_dump.items():
        pred_y = tree.predict(data.ix[:, [i for i in data.columns if i not in outputs]])
        row_index = 0
        for index, instance in data.iterrows():
            if instance[output] != pred_y[row_index]:
                error += 1
            row_index += 1
    return error/len(data)


def write_batch(enable_write_experiments_to_DB, ALCS_configuration, induced, experiment_fk, metrics_by_iteration, write_iterations_batch_size, git_version):

    def first_batch(induced, write_iterations_batch_size):
        return induced == write_iterations_batch_size

    if (write_iterations_batch_size == 1) or ((induced > 1) & ((induced  % write_iterations_batch_size) == 0)):
        if enable_write_experiments_to_DB:
            # first batch
            if first_batch(induced, write_iterations_batch_size):
                experiment_fk = write_experiment(git_version, ALCS_configuration, metrics_by_iteration)
            else:
                write_iterations(experiment_fk, metrics_by_iteration)
        else:# write to file
            # first batch
            lines_to_write = []
            if first_batch(induced, write_iterations_batch_size):
                lines_to_write.append("* created: " + str(datetime.datetime.now()))
                lines_to_write.append("\n* git version: " + git_version)
                lines_to_write.append("\n\n* configuration:\n")
                lines_to_write.append(ALCS_configuration.__str__())
                lines_to_write.append("\n\n* Iterations:\n")
                lines_to_write.append("\niteration_number\tnum_of_instances\tedges\tvertices\tcomponent_distribution_and\tcomponent_distribution_or\tcomponent_distribution_not\tcomponent_distribution_xor\tdegree_distribution\tavg_vertex_degree\ttest_set_error\toa_is_optimal\titeration_time\n")

            for iteration, metrics in metrics_by_iteration.items():
                lines_to_write.append(str(iteration))
                lines_to_write.append("\t" + str(metrics["num_of_instances"]))
                lines_to_write.append("\t" + str(metrics["sys_description"]["edges"]))
                lines_to_write.append("\t" + str(metrics["sys_description"]["vertices"]))
                lines_to_write.append("\t" + str(get_metric_to_persist(metrics["sys_description"]["comp_distribution_map"], TwoAnd.name)))
                lines_to_write.append("\t" + str(get_metric_to_persist(metrics["sys_description"]["comp_distribution_map"], TwoOr.name)))
                lines_to_write.append("\t" + str(get_metric_to_persist(metrics["sys_description"]["comp_distribution_map"], OneNot.name)))
                lines_to_write.append("\t" + str(get_metric_to_persist(metrics["sys_description"]["comp_distribution_map"], TwoXor.name)))
                degree_distribution = ",".join([(str(degree) + ':' + str(degree_count)) for degree, degree_count in
                          metrics["sys_description"]['degree_distribution'].items()])
                lines_to_write.append("\t" + ("-1" if len(degree_distribution) == 0 else degree_distribution))
                lines_to_write.append("\t" + str(metrics["sys_description"]["avg_vertex_degree"]))
                lines_to_write.append("\t" + str(metrics["test_set_error"]))
                lines_to_write.append("\t" + str(metrics["oa_is_optimal"]))
                lines_to_write.append("\t" + str(metrics["iteration_time"]))
                lines_to_write.append("\n")

            with open(circuit_name + '_experiments_log.txt', 'a') as experiments_log_file:
                experiments_log_file.writelines(lines_to_write)
        metrics_by_iteration = {}

    return experiment_fk, metrics_by_iteration


def run_ALCS(ALCS_configuration, orig_data, oa_by_strength_map, write_iterations_batch_size, enable_write_experiments_to_DB, git_version):
    outputs = get_output_names(orig_data)
    number_of_outputs = len(outputs)

    orig_max_input_index = len(orig_data.columns.values) - number_of_outputs
    inputs = orig_data.columns[:orig_max_input_index]
    # console_file.write(str(orig_max_input_index) + " intputs: " + str(inputs) + ", \n" + str(number_of_outputs) + " outputs: " + str(outputs))
    print(str(orig_max_input_index) + " intputs: " + str(inputs) + ", \n" + str(number_of_outputs) + " outputs: " + str(outputs))

    print("Initial data:")
    print(orig_data)
    print("##################")

    induced = 0
    gate_features_inputs = generate_gate_features(inputs) # init the starting inputs as feature gates for later process in the tree nodes
    curr_oa = None
    values_to_explore_by_tree = {}
    metrics_by_iteration = {}
    experiment_fk = -1  # dummy experiment

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
        start_time = int(round(time.time() * 1000))
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

        induced += 1
        gate_features_inputs.insert(non_not_max_input_index, best_gate_feature)
        # console_file.write("============ Selected Gate is: %s" % (best_attribute_name))
        print ("============ Selected Gate is: %s" % (best_attribute_name))
        values_to_explore_dict = {} # node values ordered from root to leaf
        for output, tree in best_trees_dump.items():
            # console_file.write("\ntree for " + str(output))
            print("\ntree for " + str(output))
            print_tree_to_code(tree, gate_features_inputs)
            if ALCS_configuration.use_explore_nodes:
                values_to_explore_by_tree[output] = get_curr_values_to_explore(tree, gate_features_inputs)
        # console_file.write("=============")
        print ("=============")

        before = int(round(time.time() * 1000))
        evaluate_metrics(metrics_by_iteration, orig_data, data, induced, outputs, num_of_instances, oa_is_optimal, best_quality, best_trees_dump,
                             gate_features_inputs, number_of_outputs, non_not_max_input_index)
        end_time = int(round(time.time() * 1000))
        metrics_by_iteration[induced]['iteration_time'] = (end_time - start_time) / 1000
        experiment_fk, metrics_by_iteration = write_batch(enable_write_experiments_to_DB, ALCS_configuration, induced, experiment_fk,
                                                          metrics_by_iteration, write_iterations_batch_size, git_version)

    return metrics_by_iteration, induced, experiment_fk


def evaluate_metrics(metrics_by_iteration, orig_data, data, induced, outputs, num_of_insances, oa_is_optimal, best_quality, best_trees_dump, gate_features_inputs, number_of_outputs, non_not_max_input_index):
    sys_description = {'edges': 0, 'vertices': 0, 'comp_distribution_map': {},
                       'degree_distribution': {}, 'avg_vertex_degree': 0}
    test_set_error = -1

    if best_quality <= 3 * number_of_outputs:
        curr_gates_map = generate_gates_map(best_trees_dump, gate_features_inputs)
        sys_description = create_system_description(curr_gates_map, number_of_outputs)

        # train_set_error = get_model_error(best_trees_dump, data_dump)
        # print('train_set_error: ', train_set_error)
        if len(orig_data) > 0:
            delete_data_index_list = [ind for ind in data.index if ind in orig_data.index]
            test_set = orig_data.copy().drop(delete_data_index_list)
            if len(test_set) > 0:
                test_set, _ = apply_new_attributes(induced - 1, test_set,
                                                   len(test_set.columns.values) - number_of_outputs,
                                                   gate_features_inputs, len(gate_features_inputs) - 1,
                                                   len(gate_features_inputs))
                test_set, _ = apply_new_attributes(induced, test_set, non_not_max_input_index,
                                                   gate_features_inputs,
                                                   non_not_max_input_index,
                                                   non_not_max_input_index + 1)
                test_set_error = get_model_error(best_trees_dump, test_set, outputs)
                print('test_set_error: ' + str(test_set_error) + " test size: " + str(len(test_set)))


    metrics_by_iteration[induced] = {'num_of_instances': num_of_insances,
                                 'sys_description': sys_description,
                                 'test_set_error': test_set_error,
                                 'oa_is_optimal': oa_is_optimal}


# get_component_distribution_metric(expected_gates_map, best_trees_dump, best_quality, number_of_outputs, gate_features_inputs)
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
    enable_write_experiments_to_DB = False
    write_iterations_batch_size = 5
    circuit_name = "c17"
    # pre_def_list = [12, 20, 37, 57, 58, 106, 234, 362, 466, 466, 502, 508, 508, 508, 508, 508, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
    # pre_def_list = [12, 23, 47, 60, 64, 116, 244, 360, 427, 468, 468, 468, 485, 485, 485, 485, 485, 485, 485, 485, 485, 485, 485, 485, 485, 485, 485, 485, 485, 485]
    # pre_def_list = [12, 23, 47, 60, 64, 116, 244, 372, 425, 450, 450, 450, 462, 487, 487, 487, 487, 487, 487, 487, 487, 487, 487, 487, 487, 487, 487, 487, 487, 487]
    pre_def_list = [8,12,18,25,25,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]
    file_name = TRUTH_TABLE_PATH + circuit_name + ".tab"
    possible_gates = [OneNot, TwoXor, TwoAnd, TwoOr]

    orig_data = pandas.read_csv(file_name, delimiter='\t', header=0)
    ALCS_configuration = ActiveLearningCircuitSynthesisConfiguration(file_name=file_name, total_num_of_instances=len(orig_data),
                                    possible_gates=possible_gates, subset_min=1, subset_max=2, max_num_of_iterations=30,
                                    use_orthogonal_arrays=True, use_explore_nodes=True, randomize_remaining_data=True,
                                    random_batch_size=int(round(len(orig_data)*0.1)),
                                    pre_defined_random_size_per_iteration=[],
                                    min_oa_strength=2)

    print("Working on: " + ALCS_configuration.file_name)

    oa_by_strength_map = init_orthogonal_arrays(ALCS_configuration.use_orthogonal_arrays)
    git_version = get_current_git_version()
    metrics_by_iteration, induced, experiment_fk = run_ALCS(ALCS_configuration, orig_data, oa_by_strength_map, write_iterations_batch_size, enable_write_experiments_to_DB, git_version)
    if len(metrics_by_iteration) > 0:
        write_batch(enable_write_experiments_to_DB, ALCS_configuration, induced, experiment_fk, metrics_by_iteration, write_iterations_batch_size, git_version)