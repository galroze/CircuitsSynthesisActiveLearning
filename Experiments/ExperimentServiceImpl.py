from Experiments.ConfigurationsDAO import *
from Experiments.IterationsDAO import *
from Experiments.ExperimentsDAO import *
from LogicTypes import *


def write_experiment(git_version, run_time, ALCS_configuration, metrics_by_iteration):

    possible_gates = ",".join([gate.name for gate in ALCS_configuration.possible_gates])
    conf = ConfigurationsDAO.get_configurations(escape_str(ALCS_configuration.file_name), ALCS_configuration.total_num_of_instances,
                        possible_gates, ALCS_configuration.subset_min, ALCS_configuration.subset_max,
                        ALCS_configuration.max_num_of_iterations, convert_boolean(ALCS_configuration.use_orthogonal_arrays),
                        convert_boolean(ALCS_configuration.use_explore_nodes),
                        convert_boolean(ALCS_configuration.randomize_remaining_data), ALCS_configuration.random_batch_size,
                        ALCS_configuration.min_oa_strength)
    if len(conf) == 1:
        configuration_fk = conf[0][0]
    elif len(conf) > 1:
        raise Exception("There can't be more than one configuration")
    else:
        configuration_fk = ConfigurationsDAO.insert_configuration(escape_str(ALCS_configuration.file_name),
                        ALCS_configuration.total_num_of_instances, possible_gates,
                        ALCS_configuration.subset_min, ALCS_configuration.subset_max, ALCS_configuration.max_num_of_iterations,
                        convert_boolean(ALCS_configuration.use_orthogonal_arrays), convert_boolean(ALCS_configuration.use_explore_nodes),
                        convert_boolean(ALCS_configuration.randomize_remaining_data), ALCS_configuration.random_batch_size,
                        ALCS_configuration.min_oa_strength)

    experiment_fk = ExperimentsDAO.insert_experiment(git_version, configuration_fk, run_time)

    for induced, metric in metrics_by_iteration.items():
        IterationsDAO.insert_iteration(experiment_fk, induced,
                                       get_metric_to_persist(metric['component_distribution'], TwoAnd.name),
                                       get_metric_to_persist(metric['component_distribution'], TwoOr.name),
                                       get_metric_to_persist(metric['component_distribution'], OneNot.name),
                                       get_metric_to_persist(metric['component_distribution'], TwoXor.name),
                                       convert_boolean(metric["oa_is_optimal"]), metric["num_of_instances"])


def convert_boolean(value):
    return 1 if value else 0


def get_metric_to_persist(metric, entry):
    return metric[entry] if metric.get(entry) is not None else -1


def escape_str(value):
    escaped = value.translate(str.maketrans({"-": r"\-",
                                                "]": r"\]",
                                                "\\": r"\\",
                                                "^": r"\^",
                                                "$": r"\$",
                                                "*": r"\*",
                                                ".": r"\."}))
    return escaped