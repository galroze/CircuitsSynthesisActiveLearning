from Experiments.ConfigurationsDAO import *
from Experiments.IterationsDAO import *
from Experiments.ExperimentsDAO import *
from LogicTypes import *
from LogicUtils import *


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
        sys_desc = metric['sys_description']
        IterationsDAO.insert_iteration(experiment_fk, induced, metric["num_of_instances"],
                                       sys_desc['edges'],
                                       sys_desc['vertices'],
                                       get_metric_to_persist(sys_desc['comp_distribution_map'], TwoAnd.name),
                                       get_metric_to_persist(sys_desc['comp_distribution_map'], TwoOr.name),
                                       get_metric_to_persist(sys_desc['comp_distribution_map'], OneNot.name),
                                       get_metric_to_persist(sys_desc['comp_distribution_map'], TwoXor.name),
                                       ",".join([(str(degree) + ':' + str(degree_count)) for degree, degree_count in sys_desc['degree_distribution'].items()]),
                                       sys_desc['avg_vertex_degree'],
                                       convert_boolean(metric["oa_is_optimal"]))


def convert_boolean(value):
    return 1 if value else 0


def escape_str(value):
    escaped = value.translate(str.maketrans({"-": r"\-",
                                                "]": r"\]",
                                                "\\": r"\\",
                                                "^": r"\^",
                                                "$": r"\$",
                                                "*": r"\*",
                                                ".": r"\."}))
    return escaped