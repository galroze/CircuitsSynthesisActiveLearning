
class IterationContext:
    def __init__(self, iteration_num, curr_instances_indices, active_features, oa_by_strength_map,
                 new_gate_feature, values_to_explore_by_tree):
        self.iteration_num = iteration_num
        self.curr_instances_indices = curr_instances_indices
        self.active_features = active_features
        self.oa_by_strength_map = oa_by_strength_map
        self.new_gate_feature = new_gate_feature
        self.values_to_explore_by_tree = values_to_explore_by_tree
