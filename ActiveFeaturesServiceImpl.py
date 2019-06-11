from LogicTypes import OneNot
import numpy as np
import pandas as pd
import random


# CONSTANTS
FEATURE_GATE = 'feature_gate'
ITER_UPDATE = 'iter_update'
ACTIVE = 'active'
SCORE = 'score'
IMPURITY_BY_OUTPUT_ITERATION = 'impurity_by_iteration'


def init_feature_info(curr_iteration, feature_gate, output_names):
    return {FEATURE_GATE: feature_gate,
            ACTIVE: True,
            SCORE: 1,
            IMPURITY_BY_OUTPUT_ITERATION: {output_name: list() for output_name in output_names},
            ITER_UPDATE: curr_iteration}


# getting the relevant active features for starting a new iteration
def get_active_features(features_info_map):
    active_features = []
    for feature_name, feature_info_map in features_info_map.items():
        active_features.append(feature_info_map[FEATURE_GATE]) if feature_info_map[ACTIVE] else None
    return active_features


def calc_score_apply_binary_participation_approach(feature_info_map, output_name, curr_iteration):
    if feature_info_map[IMPURITY_BY_OUTPUT_ITERATION][output_name][curr_iteration] > 0:
        feature_info_map[SCORE] = 1
    # zero score only if first tree in current iteration so not to override previous trees
    elif curr_iteration > feature_info_map[ITER_UPDATE]:
        feature_info_map[SCORE] = 0
    feature_info_map[ITER_UPDATE] = curr_iteration


def update_score_apply_accumulated_binary_participation_approach(feature_info_map, output_name, curr_iteration,
                                                                 min_prev_iteration_participation):
    # sum all last 'min_prev_iteration_participation' iterations to see if feature participated in any of them.
    if np.sum(feature_info_map[IMPURITY_BY_OUTPUT_ITERATION][output_name]
              [max(curr_iteration - min_prev_iteration_participation, 0): curr_iteration]) > 0:
        feature_info_map[SCORE] = 1
    # zero score only if first tree in current iteration so not to override previous trees
    elif curr_iteration > feature_info_map[ITER_UPDATE]:
        feature_info_map[SCORE] = 0
    feature_info_map[ITER_UPDATE] = curr_iteration


# Updating all features' scores after an iteration is done and a new attribute was chosen
def update_att_score(ALCS_configuration, active_features, features_info_map, output_name, tree, induced):
    tree_ = tree.tree_
    nodes_impurity = tree_.impurity
    node_index = 0
    # iterating feature_inputs to obtain same order as the decision tree's input data for getting the impurity
    for feature_input in active_features:
        feature_info_map = features_info_map[feature_input.to_string()]
        if feature_info_map[ACTIVE]:
            tree_feature_index = np.where(tree_.feature == node_index)[0]  # Get the index of a feature
            # feature took part in current tree
            node_impurity = nodes_impurity[tree_feature_index[0]] if len(tree_feature_index) > 0 else 0
        else:
            node_impurity = 0  # feature didn't take part in current tree, it's impurity should be 0
        # updating impurity using all trees of best chosen features, therefore updating and not overriding
        curr_impurity_list = feature_info_map[IMPURITY_BY_OUTPUT_ITERATION][output_name]
        if len(curr_impurity_list) == induced:
            curr_impurity_list[induced - 1] = curr_impurity_list[induced - 1] + node_impurity
        else:
            curr_impurity_list.insert(induced - 1, node_impurity)
        update_score_apply_accumulated_binary_participation_approach(feature_info_map, output_name, induced,
                                                            ALCS_configuration.min_prev_iteration_participation)
        node_index += 1


# monte carlo style for choosing active features by their score
def update_features_activeness_for_iteration(features_info_map, active_features_thresh):
    random.seed(2018)
    features_info_df = pd.DataFrame(features_info_map.values())
    # adjusting zeros so that aggregated weights would be distinct but still remaining low score
    features_info_df.loc[features_info_df['score'] == 0, 'score'] = 0.0001

    active_features = []
    features_info_df['agg_weight'] = 0
    # trim number of rolls by either threshold or number of possible features
    active_features_thresh = min(active_features_thresh, len(features_info_df))
    for i in range(active_features_thresh):
        scores_sum = sum(features_info_df[SCORE])
        features_info_df['weight'] = features_info_df[SCORE] / scores_sum
        prev_agg = 0
        for index, row in features_info_df.iterrows():
            features_info_df.loc[index, 'agg_weight'] = row['weight'] + prev_agg
            prev_agg += row['weight']

        roll = random.random()
        # get closest point to roll
        possible_values = np.asarray(features_info_df['agg_weight'])
        nearest_value = min(possible_values[possible_values >= roll])

        roll_feature = features_info_df.loc[features_info_df['agg_weight'] == nearest_value][FEATURE_GATE]
        active_features.append(roll_feature.item())
        features_info_df.drop([roll_feature.index.item()], inplace=True)

    for feature_info_map in features_info_map.values():
        feature_info_map[ACTIVE] = True if feature_info_map[FEATURE_GATE] in active_features else False
    return