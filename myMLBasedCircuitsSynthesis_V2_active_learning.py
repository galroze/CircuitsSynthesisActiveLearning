from os import listdir
from os.path import isfile, join
import itertools
import time
import re
import datetime

from IteraionContext import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from LogicUtils import *
from LogicTypes import *
from ActiveFeaturesServiceImpl import *
from Experiments.ExperimentServiceImpl import write_experiment
from Experiments.ExperimentServiceImpl import write_iterations
from dataSystemsUtil import *


orig_cached_data = []

class ActiveLearningCircuitSynthesisConfiguration:

    def __init__(self, file_name, total_num_of_instances, possible_gates, subset_min, subset_max, max_num_of_iterations,
                 use_orthogonal_arrays, oa_hold_iterations, use_explore_nodes, randomize_remaining_data, random_batch_size,
                 pre_defined_random_size_per_iteration, min_oa_strength, active_features_thresh,
                 min_prev_iteration_participation):

        self.file_name = file_name
        self.total_num_of_instances = total_num_of_instances
        self.possible_gates = possible_gates
        self.subset_min = subset_min
        self.subset_max = subset_max
        self.max_num_of_iterations = max_num_of_iterations
        self.use_orthogonal_arrays = use_orthogonal_arrays
        self.oa_hold_iterations = oa_hold_iterations
        self.use_explore_nodes = use_explore_nodes
        # when using OA and there are no more arrays to use, True for random, False for taking all remaining data
        self.randomize_remaining_data = randomize_remaining_data
        self.random_batch_size = random_batch_size
        self.pre_defined_random_size_per_iteration = pre_defined_random_size_per_iteration
        self.min_oa_strength = min_oa_strength
        self.active_features_thresh = active_features_thresh
        self.min_prev_iteration_participation = min_prev_iteration_participation


    def __str__(self):
        return '\n'.join('%s=%s' % item for item in vars(self).items())


def insert_gate_as_att(data, gate_feature, use_cache):
    global orig_cached_data
    new_attribute_name = gate_feature.to_string()
    if use_cache:
        new_column = get_transformed_att_value_cache_enabled(None, orig_cached_data, gate_feature, data, True)
    else:
        new_column = get_transformed_att_value(data, gate_feature.inputs, gate_feature.gate)
    data.insert(len(data.columns) - len(get_output_names(data)), new_attribute_name, new_column)


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


def get_tree_features(active_features, tree):
    tree_ = tree.tree_
    features = [
        active_features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
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
        if comma_index != -1:  # comma leftover from previous iteration
            start_index = comma_index + 1
        gate = attribute_name[start_index:parenthesis_index]
        if not gates_map.__contains__(gate):
            gates_map[gate] = 0
        gates_map[gate] += 1
        attribute_name = attribute_name[parenthesis_index + 1:]
        parenthesis_index = attribute_name.find("(")
    return gates_map


def get_batch_using_explorable_nodes(data, values_to_explore_by_tree):
    if len(values_to_explore_by_tree) > 0:
        batch_indices = []
        for tree_values in values_to_explore_by_tree.values():
            # dropping former tree batch that is already included in the returned batch
            potential_batch = data.drop(batch_indices)
            for gate_feature, value in tree_values:
                if len(potential_batch) > 0:
                    value = True if value == 1 else False
                    potential_batch = potential_batch[potential_batch[gate_feature.to_string()] == value]
                else:
                    break
            batch_indices.extend(potential_batch.index)
        return list(set(batch_indices))
    else:
        return list(data.index)


def get_value_to_explore(value, values_to_explore):
    for value_to_explore in values_to_explore:
        if value == (value_to_explore[0]):
            return value_to_explore
    return None


def get_oa_by_values_to_explore(values_to_explore, active_features, data_list):
    # going over each input and placing an explorable value if exist or any 0 or 1(don't care) value if it doesn't exist in values to explore
    # for every value look for its position in the feature gates collection and apply oa filter at the same position
    explorable_regex = '^'
    max_att = min(len(active_features), len(data_list[0]))
    for feature_input_index in range(max_att):
        value_to_explore = get_value_to_explore(active_features[feature_input_index], values_to_explore)
        if value_to_explore is not None:
            explorable_regex += str(value_to_explore[1])  # appends the node's value
        else:
            explorable_regex += '[0-1]'  # appends don't care
    explorable_regex += '.*$'
    compiled_regex = re.compile(explorable_regex)
    return list(filter(compiled_regex.match, data_list))


def get_instances_indices_by_values_to_explore(values_to_explore_by_tree, data_frame, active_features):

    at_least_one_tree_has_val_to_explore = False
    instances_indices = []
    for tree_values in values_to_explore_by_tree.values():
        if len(data_frame) == 0:
            break
        else:
            # filtering only values to explore which are active
            active_features_names = [active_feature.to_string() for active_feature in active_features]
            tree_values = [tree_value for tree_value in tree_values if tree_value[0] in active_features_names]
            if len(tree_values) > 0:
                at_least_one_tree_has_val_to_explore = True
                for k, v in tree_values:
                    curr_indices = data_frame.index[data_frame[k] == (False if v == 0 else True)].tolist()
                    instances_indices.extend(curr_indices)
                    data_frame = data_frame.drop(curr_indices)
    if not at_least_one_tree_has_val_to_explore:
        return data_frame.index.tolist()
    else:
        return instances_indices


def get_batch_using_oa(data, potential_data_indices, active_features_names_list, oa):
    # converting oa list to dataframe for fast joining oa with data
    oa_df = pd.DataFrame([list(oa_item) for oa_item in oa], columns=active_features_names_list)
    oa_df = oa_df.replace('1', True).replace('0', False)
    merged_df = data.loc[potential_data_indices].reset_index().merge(oa_df, how="inner", on=active_features_names_list).set_index('index')
    return merged_df


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
        max_impurity = max(nodes_impurity[1:])  # get max impurity from inner nodes and not the root
        if max_impurity != 0:
            for node_index in range(1, len(nodes_impurity)):
                # in case there are several nodes with same entropy value
                if max_impurity == nodes_impurity[node_index]:
                    break  # don't take all equal entropies, only the first one for consistency

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


def get_batch_indices(ALCS_conf, iteration_context, orig_data, potential_data_indices):
    batch_indices = []
    if len(potential_data_indices) > 0:
        if ALCS_conf.use_explore_nodes and len(iteration_context.values_to_explore_by_tree) > 0:
            # filter out indices that don't comply with active values to explore
            potential_data_indices = get_instances_indices_by_values_to_explore(iteration_context.values_to_explore_by_tree,
                                                        orig_data.loc[potential_data_indices].copy(deep=True),
                                                                         iteration_context.active_features)
        if ALCS_conf.randomize_remaining_data:  # Random batch
            if len(ALCS_conf.pre_defined_random_size_per_iteration) > 0:
                if iteration_context.iteration_num == 1:
                    batch_size = ALCS_conf.pre_defined_random_size_per_iteration[iteration_context.iteration_num - 1]
                else:
                    batch_size = ALCS_conf.pre_defined_random_size_per_iteration[iteration_context.iteration_num - 1] - \
                                 ALCS_conf.pre_defined_random_size_per_iteration[iteration_context.iteration_num - 2]
            else:
                batch_size = min(ALCS_conf.random_batch_size, len(potential_data_indices))
        else:  # All
            batch_size = len(potential_data_indices)

        if batch_size > 0:
            random.seed(2018)
            sample_size = min(len(potential_data_indices), batch_size)
            batch_indices = random.sample(potential_data_indices, sample_size)
    return batch_indices


def get_batch(ALCS_conf, orig_data, iteration_context, oa_strength):
    nearest_oa_name = '-'
    # remove used indices from total orig data indices
    potential_data_indices = np.delete(orig_data.index.copy(deep=True), iteration_context.curr_instances_indices).tolist()

    if ALCS_conf.use_orthogonal_arrays:
        if iteration_context.oa_by_strength_map.__contains__(oa_strength):

            active_features_for_oa = get_input_names(orig_data)
            nearest_oa = get_nearest_oa(iteration_context.oa_by_strength_map[oa_strength], len(active_features_for_oa))
            if nearest_oa is not None:
                nearest_oa_name = nearest_oa.name
                next_oa = trim_oa_to_fit_inputs(nearest_oa, len(active_features_for_oa))
                data_batch = get_batch_using_oa(orig_data, potential_data_indices, active_features_for_oa, next_oa.array)
                data_batch_indices = get_batch_using_explorable_nodes(data_batch, iteration_context.values_to_explore_by_tree)
            else:  # no OA found(probably not enough attributes in current strength)
                data_batch_indices = get_batch_indices(ALCS_conf, iteration_context, orig_data, potential_data_indices)
        else:  # no more OA strengths
            data_batch_indices = get_batch_indices(ALCS_conf, iteration_context, orig_data, potential_data_indices)
    else:  # Random Batch
        data_batch_indices = get_batch_indices(ALCS_conf, iteration_context, orig_data, potential_data_indices)

    # joining last iterations instances with current new batch instances
    iteration_context.curr_instances_indices = np.array(list(iteration_context.curr_instances_indices) + data_batch_indices)
    iteration_data = orig_data.loc[iteration_context.curr_instances_indices].copy(deep=True)

    return orig_data, iteration_data, nearest_oa_name


def get_model_error(best_trees_dump, data, active_features):
    error = 0
    for output, tree in best_trees_dump.items():
        pred_y = tree.predict(data[[feature.to_string() for feature in active_features]])
        row_index = 0
        for index, instance in data.iterrows():
            if instance[output] != pred_y[row_index]:
                error += 1
            row_index += 1
    return error/len(data)


def write_batch(enable_write_experiments_to_DB, ALCS_configuration, induced, experiment_fk, metrics_by_iteration,
                original_metrics, write_iterations_batch_size, git_version):

    def first_batch(induced, write_iterations_batch_size):
        return induced == write_iterations_batch_size

    if (write_iterations_batch_size == 1) or ((induced > 1) & ((induced % write_iterations_batch_size) == 0)):
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
                lines_to_write.append("\niteration_number\titeration_time\tnum_of_instances\ttest_set_error\toa_name"
                                      "\tnumber_of_selected_gates\tedges"
                                      "\tedges_dist\tvertices\tvertices_dist\tcomponent_distribution_and"
                                      "\tcomponent_distribution_and_dist"
                                      "\tcomponent_distribution_or\tcomponent_distribution_or_dist"
                                      "\tcomponent_distribution_not\tcomponent_distribution_not_dist"
                                      "\tcomponent_distribution_xor\tcomponent_distribution_xor_dist"
                                      "\tavg_vertex_degree\tavg_vertex_degree_dist"
                                      "\tdegree_distribution\tdegree_distribution1_dist\n")

            for iteration, metrics in metrics_by_iteration.items():
                lines_to_write.append(str(iteration))
                lines_to_write.append("\t" + str(metrics["iteration_time"]))
                lines_to_write.append("\t" + str(metrics["num_of_instances"]))
                lines_to_write.append("\t" + str(metrics["test_set_error"]))

                lines_to_write.append("\t" + str(metrics["oa_name"]))
                lines_to_write.append("\t" + str(metrics["number_of_selected_gates"]))

                lines_to_write.append("\t" + str(metrics["sys_description"]["edges"]))
                edges_dist = calculate_dist_metric(original_metrics["edges"], metrics["sys_description"]["edges"])
                lines_to_write.append("\t" + str(edges_dist))
                
                lines_to_write.append("\t" + str(metrics["sys_description"]["vertices"]))
                vertices_dist = calculate_dist_metric(original_metrics["vertices"], metrics["sys_description"]["vertices"])
                lines_to_write.append("\t" + str(vertices_dist))

                component_distribution_and = get_metric_to_persist(metrics["sys_description"]["comp_distribution_map"], TwoAnd.name)
                lines_to_write.append("\t" + str(component_distribution_and))
                component_distribution_and_dist = calculate_dist_metric(original_metrics["component_distribution_and"],
                                                                        component_distribution_and)
                lines_to_write.append("\t" + str(component_distribution_and_dist))

                component_distribution_or = get_metric_to_persist(metrics["sys_description"]["comp_distribution_map"], TwoOr.name)
                lines_to_write.append("\t" + str(component_distribution_or))
                component_distribution_or_dist = calculate_dist_metric(original_metrics["component_distribution_or"],
                                                                       component_distribution_or)
                lines_to_write.append("\t" + str(component_distribution_or_dist))

                component_distribution_not = get_metric_to_persist(metrics["sys_description"]["comp_distribution_map"], OneNot.name)
                lines_to_write.append("\t" + str(component_distribution_not))
                component_distribution_not_dist = calculate_dist_metric(original_metrics["component_distribution_not"],
                                                                        component_distribution_not)
                lines_to_write.append("\t" + str(component_distribution_not_dist))

                component_distribution_xor = get_metric_to_persist(metrics["sys_description"]["comp_distribution_map"], TwoXor.name)
                lines_to_write.append("\t" + str(component_distribution_xor))
                component_distribution_xor_dist = calculate_dist_metric(original_metrics["component_distribution_xor"],
                                                                        component_distribution_xor)
                lines_to_write.append("\t" + str(component_distribution_xor_dist))

                lines_to_write.append("\t" + str(metrics["sys_description"]["avg_vertex_degree"]))
                avg_vertex_degree_dist = calculate_dist_metric(original_metrics["avg_vertex_degree"],
                                                      metrics["sys_description"]["avg_vertex_degree"])
                lines_to_write.append("\t" + str(avg_vertex_degree_dist))

                # We've definitely converged
                if metrics["sys_description"]['degree_distribution'][1] != -1:
                    original_metric_degrees = [int(key[len("degree_distribution"):]) for key in original_metrics.keys()
                                               if key.startswith("degree_distribution")]
                    current_metric_degrees = list(metrics["sys_description"]['degree_distribution'].keys())
                    original_metric_degrees.extend(current_metric_degrees)
                    max_degree = max(original_metric_degrees)
                    for degree in range(1, max_degree + 1):
                        if original_metrics.__contains__("degree_distribution" + str(degree)):
                            if metrics["sys_description"]['degree_distribution'].__contains__(degree):
                                curr_degree_dist = original_metrics["degree_distribution" + str(degree)]
                                degree_count = metrics["sys_description"]['degree_distribution'][degree]
                                lines_to_write.append("\t" + str(degree_count))
                                # original and current have values
                                lines_to_write.append("\t" + str(calculate_dist_metric(curr_degree_dist, degree_count)))
                            else:
                                # original has value, current doesn't
                                lines_to_write.append("\t\t0")
                        else:
                            if metrics["sys_description"]['degree_distribution'].__contains__(degree):
                                lines_to_write.append("\t" + str(metrics["sys_description"]['degree_distribution'][degree]))
                                # original doesn't have value, current does
                                lines_to_write.append("\t0")
                            else:
                                # original and current don't have values
                                lines_to_write.append("\t\t1")

                lines_to_write.append("\n")

            with open(circuit_name + '_experiments_log.txt', 'a') as experiments_log_file:
                experiments_log_file.writelines(lines_to_write)
        metrics_by_iteration = {}

    return experiment_fk, metrics_by_iteration


def calculate_dist_metric(numerator, denominator):
    # metric value doesn't exist
    if denominator == -1:
        return ""
    # original didn't have value
    if numerator == -1:
        # metric had value
        if denominator != 0:
            return 0
        # both didn't have metric
        else:
            return 1
    else:
        # both have values
        if denominator != 0:
            return round(Decimal(numerator) / Decimal(denominator), 6)
        # original has value but current doesn't
        else:
            return 0


def add_not_gate(orig_data, iteration_context, features_info_map):
    # ADD NOT GATE !AT THE END! FOR NEW ADDED ARG FROM LAST ITERATION
    for new_gate_feature in iteration_context.new_gate_features:
        gate_feature = GateFeature(OneNot, [new_gate_feature])
        insert_gate_as_att(orig_data, gate_feature, True)
        features_info_map[gate_feature.to_string()] = init_feature_info(iteration_context.iteration_num - 1,
                                                                        gate_feature, get_output_names(orig_data))


def run_ALCS(ALCS_configuration, orig_data, oa_by_strength_map, write_iterations_batch_size,
             enable_write_experiments_to_DB, git_version, original_metrics):
    outputs = get_output_names(orig_data)
    number_of_outputs = len(outputs)

    orig_max_input_index = len(orig_data.columns.values) - number_of_outputs
    inputs = orig_data.columns[:orig_max_input_index]
    print(str(orig_max_input_index) + " inputs: " + str(inputs) + ", \n" + str(number_of_outputs) + " outputs: " + str(outputs)
          + "\nInitial data size: " + str(len(orig_data)) + " instances\n##################")

    # init the starting inputs as feature gates for later process in the tree nodes
    gate_features_inputs = generate_gate_features(inputs)
    values_to_explore_by_tree = {}
    metrics_by_iteration = {}
    best_features_list = []
    experiment_fk = -1  # dummy experiment

    # FIRST ADD THE NOT GATE FOR ALL CURRENT INPUTS
    not_gate_arguments = itertools.combinations(gate_features_inputs, 1)
    for not_gate_argument_tuple in not_gate_arguments:
        gate_feature = GateFeature(OneNot, [not_gate_argument_tuple[0]])
        insert_gate_as_att(orig_data, gate_feature, False)
        gate_features_inputs.append(gate_feature)

    global orig_cached_data
    orig_cached_data = orig_data.copy()

    # init the starting inputs' info for later process in the tree nodes
    features_info_map = {curr_input.to_string(): init_feature_info(0, curr_input, outputs) for curr_input in gate_features_inputs}

    iteration_context = IterationContext(1, np.array([]), get_active_features(features_info_map), oa_by_strength_map,
                                         [], values_to_explore_by_tree)

    oa_strength = ALCS_configuration.min_oa_strength
    feature_names_black_list = set()
    #############################################  -- STARTING ITERATIONS -- #########################################
    while iteration_context.iteration_num < ALCS_configuration.max_num_of_iterations:
        start_time = int(round(time.time() * 1000))
        best_quality = 10000
        best_attribute_name = ""
        best_attribute_gates_map = {}

        add_not_gate(orig_data, iteration_context, features_info_map)
        iteration_context.new_gate_features = []
        # monte carlo
        update_features_activeness_for_iteration(features_info_map, ALCS_configuration.active_features_thresh)
        iteration_context.active_features = get_active_features(features_info_map)
        if (iteration_context.iteration_num > 1) and \
                ((iteration_context.iteration_num - 1) % ALCS_configuration.oa_hold_iterations == 0):
            oa_strength += 1
        orig_data, iteration_data, nearest_oa_name = get_batch(ALCS_configuration, orig_data, iteration_context, str(oa_strength))

        num_of_instances = len(iteration_data)
        iteration_data_input_len = len(iteration_data.columns) - number_of_outputs
        active_feature_names = [active_feature.to_string() for active_feature in iteration_context.active_features]
        print("\n**** iteration #: " + str(iteration_context.iteration_num) + ", " + "# of instances: " + str(num_of_instances)
              + ",\n(" + str(len(active_feature_names)) + ") Active features: " + str(active_feature_names)
              + ",\n(" + str(len(features_info_map) - len(active_feature_names)) +
              ") Non active features: " + str([feat_name for feat_name in get_input_names(orig_data) if feat_name not in active_feature_names])
              + "****\n")

        # Now working on all other gates
        possible_arguments_combinations = []
        # appends all combinations of different sizes of subsets
        for i in range(max(ALCS_configuration.subset_min, 2), ALCS_configuration.subset_max + 1):
            possible_arguments_combinations += itertools.combinations(iteration_context.active_features, i)

        data_input_col_names = [active_feature.to_string() for active_feature in iteration_context.active_features]
        # for each combination
        for curr_arg_combination in possible_arguments_combinations:
            # for each possible gate
            for possible_gate in ALCS_configuration.possible_gates:
                if possible_gate.numInputs == len(curr_arg_combination):
                    new_gate_feature = GateFeature(possible_gate, list(curr_arg_combination))
                    new_attribute_name = new_gate_feature.to_string()
                    if (not iteration_data.__contains__(new_attribute_name)) and \
                            (not feature_names_black_list.__contains__(new_attribute_name)):

                        new_column = get_transformed_att_value_cache_enabled(
                            orig_data[[c for c in orig_data.columns if c not in outputs]],
                            orig_cached_data, new_gate_feature, iteration_data, True)
                        # column's truth table is identical to some other column already existing
                        if new_column is None:
                            feature_names_black_list.add(new_attribute_name)
                            continue
                        iteration_data.insert(iteration_data_input_len, new_attribute_name, new_column)
                        data_input_col_names.append(new_attribute_name)

                        tree_quality = 0
                        fitted_trees = {}
                        for output_index in range(len(outputs)):
                            tree_data = DecisionTreeClassifier(random_state=0, criterion="entropy")
                            tree_data.fit(iteration_data[data_input_col_names], iteration_data[outputs[output_index]])
                            tree_quality += tree_data.tree_.node_count
                            fitted_trees[outputs[output_index]] = tree_data
                        if tree_quality < best_quality:
                            best_quality = tree_quality
                            best_gate_feature = new_gate_feature
                            best_attribute_name = new_attribute_name
                            best_attribute_gates_map = get_gates_map(best_attribute_name)
                            best_trees_dump = fitted_trees
                            best_features_list = [(best_gate_feature, best_trees_dump)]
                        elif tree_quality == best_quality:
                            best_features_list.append((new_gate_feature, fitted_trees))
                            if is_better_combination(new_attribute_name, best_attribute_gates_map):
                                best_gate_feature = new_gate_feature
                                best_attribute_name = new_attribute_name
                                best_attribute_gates_map = get_gates_map(best_attribute_name)
                                best_trees_dump = fitted_trees
                        del iteration_data[new_attribute_name]
                        data_input_col_names.remove(new_attribute_name)

        selected_gates = [sel_gate[0].to_string() for sel_gate in best_features_list]
        print("============ " + str(len(selected_gates)) + " Selected Gates are: %s" % selected_gates)
        for best_gate, best_tree_dump in best_features_list:  # insert all best features having same tree_quality
            features_info_map[best_gate.to_string()] = init_feature_info(iteration_context.iteration_num, best_gate, outputs)
            insert_gate_as_att(orig_data, best_gate, True)
            iteration_context.new_gate_features.append(best_gate)

            iteration_context.active_features.append(best_gate)  # for printing trees and exploration purposes, includes the best of all best list only
            for output, tree in best_tree_dump.items():
                update_att_score(ALCS_configuration, iteration_context.active_features, features_info_map, output,
                                 tree, iteration_context.iteration_num)

                # only the chosen gate updates the values to explore
                if ALCS_configuration.use_explore_nodes and best_gate.to_string() == best_attribute_name:
                    print("\ntree for " + str(output))
                    print_tree_to_code(tree, iteration_context.active_features)
                    iteration_context.values_to_explore_by_tree[output] = get_curr_values_to_explore(tree, iteration_context.active_features)
            iteration_context.active_features.remove(best_gate)  # for printing trees and exploration purposes
        print("=============")

        iteration_context.active_features.append(best_gate_feature)  #  for evaluating  metrics over one best feature only
        evaluate_metrics(iteration_context, metrics_by_iteration, orig_data, iteration_context.iteration_num,
                         num_of_instances, nearest_oa_name, best_quality, best_trees_dump, number_of_outputs, len(selected_gates))
        iteration_context.active_features.remove(best_gate_feature)  # for evaluating  metrics oven one best feature only

        end_time = int(round(time.time() * 1000))
        metrics_by_iteration[iteration_context.iteration_num]['iteration_time'] = (end_time - start_time) / 1000
        experiment_fk, metrics_by_iteration = write_batch(enable_write_experiments_to_DB, ALCS_configuration,
                                                          iteration_context.iteration_num, experiment_fk,
                                                          metrics_by_iteration, original_metrics,
                                                          write_iterations_batch_size, git_version)
        iteration_context.iteration_num += 1
        print('black list: ' + str(len(feature_names_black_list)))
    return metrics_by_iteration, iteration_context.iteration_num, experiment_fk


def evaluate_metrics(iteration_context, metrics_by_iteration, orig_data, induced, num_of_insances, nearest_oa_name,
                     best_quality, best_trees_dump, number_of_outputs, number_of_selected_gates):
    sys_description = {'edges': -1,
                       'vertices': -1,
                       'comp_distribution_map': {
                           TwoAnd.name: -1,
                           TwoOr.name: -1,
                           TwoXor.name: -1,
                           OneNot.name: -1
                       },
                       # degree 1 is the degree for all outputs which is always 1,
                       # this is an indicator whether the metric is converged or not
                       # if it didn't, it stays -1, if it did it is necessarily not -1.
                       'degree_distribution': {1: -1},
                       'avg_vertex_degree': -1
                       }
    test_set_error = -1

    if best_quality <= 3 * number_of_outputs:
        curr_gates_map = generate_gates_map(best_trees_dump, iteration_context.active_features)
        sys_description = create_system_description(curr_gates_map, number_of_outputs)

        test_set_indices = np.delete(orig_data.index.copy(deep=True),
                                           iteration_context.curr_instances_indices).tolist()
        if len(test_set_indices) > 0:
            test_set_data = orig_data.loc[test_set_indices].copy(deep=True)
            test_set_error = get_model_error(best_trees_dump, test_set_data, iteration_context.active_features)
            print('test_set_error: ' + str(test_set_error) + " test size: " + str(len(test_set_data)))


    metrics_by_iteration[induced] = {'num_of_instances': num_of_insances,
                                     'sys_description': sys_description,
                                     'test_set_error': test_set_error,
                                     'oa_name': nearest_oa_name,
                                     'number_of_selected_gates': number_of_selected_gates
                                     }


def generate_gates_map(trees_dump, active_features):
    curr_gates_map = {}
    # all trees are decision stumps, merge them all to distinct gates with inputs
    for output, tree in trees_dump.items():
        root_feature = get_tree_features(active_features, tree)[0]  # there is only one feature in the root
        add_gates(curr_gates_map, root_feature, output)
    return curr_gates_map


def get_gate_from_set(gate, set):
    for item in set:
        if item == gate:
            return item


def add_gates(gates_map, gate, output):

    if gate != "undefined!":  # all instances are of the same class - no tree at all
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
    write_iterations_batch_size = 1
    circuit_name = "c17"
    # pre_def_list = [12, 20, 37, 57, 58, 106, 234, 362, 466, 466, 502, 508, 508, 508, 508, 508, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
    # pre_def_list = [12, 22, 46, 60, 64, 116, 167, 218, 269, 320, 371, 422, 743, 512, 512, 512, 512, 512, 512, 512, 512,
    #                 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
    #                 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
    #                 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
    #                 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
    pre_def_list_74283 = [12, 21, 37, 48, 50, 94, 145, 196, 247, 298, 349, 400, 451, 502, 512, 512, 512, 512, 512, 512, 512,
                        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512,
                        512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
    # pre_def_list = [8,12,18,25,25,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]
    file_name = TRUTH_TABLE_PATH + circuit_name + ".tab"
    possible_gates = [TwoXor, TwoAnd, TwoOr]

    orig_data = pandas.read_csv(file_name, delimiter='\t', header=0)
    ALCS_configuration = ActiveLearningCircuitSynthesisConfiguration(file_name=file_name, total_num_of_instances=len(orig_data),
                                    possible_gates=possible_gates, subset_min=1, subset_max=2, max_num_of_iterations=400,
                                    use_orthogonal_arrays=True,
                                    oa_hold_iterations=2,
                                    use_explore_nodes=True, randomize_remaining_data=True,
                                    random_batch_size=int(round(len(orig_data)*0.1)),
                                    pre_defined_random_size_per_iteration=[],
                                    min_oa_strength=2,
                                    active_features_thresh=60, # len(get_input_names(orig_data)) * 2,
                                    min_prev_iteration_participation=5)

    print("Working on: " + ALCS_configuration.file_name)

    oa_by_strength_map = init_orthogonal_arrays(ALCS_configuration.use_orthogonal_arrays)
    git_version = get_current_git_version()
    original_metrics = read_original_metrics(circuit_name)
    metrics_by_iteration, induced, experiment_fk = run_ALCS(ALCS_configuration, orig_data, oa_by_strength_map,
                                            write_iterations_batch_size, enable_write_experiments_to_DB, git_version,
                                            original_metrics)
    if len(metrics_by_iteration) > 0:
        write_batch(enable_write_experiments_to_DB, ALCS_configuration, induced, experiment_fk, metrics_by_iteration,
                    original_metrics,write_iterations_batch_size, git_version)