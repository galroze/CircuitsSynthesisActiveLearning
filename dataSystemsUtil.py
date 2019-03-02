import csv
import pandas
import math
from LogicTypes import *
from LogicUtils import *
import collections

logic_types = {OneNot.name: OneNot, TwoXor.name: TwoXor, TwoAnd.name: TwoAnd, ThreeAnd.name: ThreeAnd,
               FourAnd.name: FourAnd, FiveAnd.name: FiveAnd, EightAnd.name: EightAnd, NineAnd.name: NineAnd,
               TwoOr.name: TwoOr, ThreeOr.name: ThreeOr,
               FourOr.name: FourOr, TwoNor.name: TwoNor, ThreeNor.name: ThreeNor, FourNor.name: FourNor,
               FiveNor.name: FiveNor, NotAnd.name: NotAnd, ThreeNand.name: ThreeNand, FourNand.name: FourNand,
               FiveNand.name: FiveNand, Buffer.name: Buffer}
basic_gates = {OneNot.name, TwoXor.name, TwoAnd.name, TwoOr.name}


def is_inner_output(output_str):
    return not output_str.startswith('o')


def handle_row_with_transform_to_basic(expected_gates_map, auxilary_gates_map, gate_name, row_inputs, row_output):
    if gate_name == Buffer.name:
       handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, row_inputs, "temp")
       handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, ["temp"], row_output)
    if gate_name == NotAnd.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs, "temp")
        handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, ["temp"], row_output)
    if gate_name == ThreeNand.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[1:], "temp")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, [row_inputs[0], "temp"], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, ["temp2"], row_output)
    if gate_name == FourNand.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[2:], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[:2], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp1", "temp2"], "temp3")
        handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, ["temp3"], row_output)
    if gate_name == FiveNand.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[:2], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[2:4], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp1", "temp2"], "temp3")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, [row_inputs[4], "temp3"], "temp4")
        handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, ["temp4"], row_output)
    elif gate_name == ThreeAnd.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[1:], "temp")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, [row_inputs[0], "temp"], row_output)
    elif gate_name == FourAnd.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[2:], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[:2], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp1", "temp2"], row_output)
    elif gate_name == FiveAnd.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[3:], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[:2], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp1", "temp2"], "temp3")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, [row_inputs[2], "temp3"], row_output)
    elif gate_name == EightAnd.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[:2], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[2:4], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[4:6], "temp3")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[6:], "temp4")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp1", "temp2"], "temp5")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp3", "temp4"], "temp6")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp5", "temp6"], row_output)
    elif gate_name == NineAnd.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[:2], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[2:4], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[4:6], "temp3")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, row_inputs[6:8], "temp4")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp1", "temp2"], "temp5")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp3", "temp4"], "temp6")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, ["temp5", "temp6"], "temp7")
        handle_row(expected_gates_map, auxilary_gates_map, TwoAnd.name, [row_inputs[8], "temp7"], row_output)
    elif gate_name == ThreeOr.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, row_inputs[1:], "temp")
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, [row_inputs[0], "temp"], row_output)
    elif gate_name == FourOr.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, row_inputs[2:], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, row_inputs[:2], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, ["temp1", "temp2"], row_output)
    elif gate_name == TwoNor.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, row_inputs, "temp")
        handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, ["temp"], row_output)
    elif gate_name == ThreeNor.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, row_inputs[1:], "temp")
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, [row_inputs[0], "temp"], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, ["temp1"], row_output)
    elif gate_name == FourNor.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, row_inputs[2:], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, row_inputs[:2], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, ["temp1", "temp2"], "temp3")
        handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, ["temp3"], row_output)
    elif gate_name == FiveNor.name:
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, row_inputs[3:], "temp1")
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, row_inputs[:2], "temp2")
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, ["temp1", "temp2"], "temp3")
        handle_row(expected_gates_map, auxilary_gates_map, TwoOr.name, [row_inputs[2], "temp3"], "temp4")
        handle_row(expected_gates_map, auxilary_gates_map, OneNot.name, ["temp4"], row_output)


def handle_row(expected_gates_map, auxilary_gates_map, gate_name, row_inputs, row_output):
    if gate_name not in basic_gates:
        handle_row_with_transform_to_basic(expected_gates_map, auxilary_gates_map, gate_name, row_inputs, row_output)
    else:
        if not expected_gates_map.__contains__(gate_name):
            expected_gates_map[gate_name] = set()

        for row_input_index in range(len(row_inputs)):
            feature_gate = auxilary_gates_map[row_inputs[row_input_index]]
            if feature_gate is None:
                raise Exception('inner gate not found')
            if row_inputs[row_input_index].startswith("temp"):
                del auxilary_gates_map[row_inputs[row_input_index]]
            row_inputs[row_input_index] = feature_gate

        gate_feature = GateFeature(logic_types.get(gate_name), [])
        gate_feature.inputs = row_inputs #constructor deep copies inputs and I want to avoid it

        if is_inner_output(row_output):
            auxilary_gates_map[row_output] = gate_feature
        else:
            gate_feature.outputs.add(row_output)
        expected_gates_map[gate_name].add(gate_feature)


def generate_expected_gates_map(circuit_name):
    fileName = CIRCUIT_SYSTEM_PATH + circuit_name + ".sys"
    lines = [line.rstrip('\n') for line in open(fileName)]

    inputs_list = get_inputs_list(lines)
    outputs_list = get_outputs_list(lines)

    auxilary_gates_map = {}
    for curr_input in inputs_list:
        auxilary_gates_map[curr_input] = GateFeature(None, [curr_input])

    expected_gates_map = {}
    for row_index in range(3, len(lines)):
        gate, row_inputs, row_output = parse_data_system_row(lines, row_index, logic_types)
        handle_row(expected_gates_map, auxilary_gates_map, gate.name, row_inputs, row_output)

    expected_gates_map['basic_inputs'] = [(basic_gate) for gate_name, basic_gate in auxilary_gates_map.items() if gate_name.startswith('i')]

    for gate_type_set in expected_gates_map.values():
        for gate in gate_type_set:
            if gate.gate is not None:
                for input in gate.inputs:
                    input.outputs.add(gate)

    return expected_gates_map, outputs_list


def parse_data_system_row(lines, row_index, logic_types):
    split_row = lines[row_index].split(sep=',')

    if row_index < len(lines) - 1:
        split_row.remove(split_row[len(split_row) - 1])
    gate_name = split_row[0].replace('[', '')
    row_output = split_row[2]
    row_inputs = split_row[3:]
    row_inputs[len(row_inputs) - 1] = row_inputs[len(row_inputs) - 1].replace(']', '').replace(',', '').replace('.', '')

    gate = logic_types.get(gate_name)
    return gate, row_inputs, row_output


def get_inputs_list(lines):
    inputs = lines[1].split(sep=',')
    inputs[0] = inputs[0].replace('[', '')
    inputs[len(inputs) - 1] = inputs[len(inputs) - 1].replace('].', '')
    return inputs


def get_outputs_list(lines):
    outputs = lines[2].split(sep=',')
    outputs[0] = outputs[0].replace('[', '')
    outputs[len(outputs) - 1] = outputs[len(outputs) - 1].replace('].', '')
    return outputs


def generate_truth_table(circuit_name):
    fileName = CIRCUIT_SYSTEM_PATH + circuit_name + ".sys"
    lines = [line.rstrip('\n') for line in open(fileName)]

    inputs = get_inputs_list(lines)

    truth_table_columns = inputs.copy()
    truth_table = pandas.DataFrame(columns=truth_table_columns, dtype=bool)

    rows = math.pow(2, len(inputs))

    for i in range(int(rows)):
        new_row = []
        for j in range(len(inputs) - 1, -1, -1):
            new_row.append(False if (int(i / math.pow(2, j)) % 2) == 0 else True)
        truth_table = truth_table.append(pandas.Series(new_row, index=truth_table.columns), ignore_index=True)

    for row_index in range(3, len(lines)):
        gate, row_inputs, row_output = parse_data_system_row(lines, row_index, logic_types)
        output_column = get_transformed_att_value(truth_table, row_inputs, gate)
        truth_table.insert(len(truth_table.columns), row_output, output_column)

    truth_table = truth_table[truth_table.columns[pandas.Series(truth_table.columns).str.startswith('z') == False]]
    truth_table.to_csv(TRUTH_TABLE_PATH + circuit_name + ".tab", sep='\t', index=False)
    print("Done generating: " + str(TRUTH_TABLE_PATH + circuit_name + ".tab") + " \ninputs: " + str(get_input_names(truth_table)) + " outputs: " + str(get_output_names(truth_table)))


def write_metrics_to_csv(circuit_name, metric_dict):
    with open('system_description_metrics\\' + circuit_name + '.csv', 'w', newline='') as csvfile:
        fieldNames = ['edges', 'vertices', 'component_distribution_and', 'component_distribution_or', 'component_distribution_not',
                      'component_distribution_xor', 'degree_distribution', 'avg_vertex_degree']
        writer = csv.DictWriter(csvfile, fieldnames=fieldNames)

        writer.writeheader()

        writer.writerow({'edges': metric_dict['edges'], 'vertices': metric_dict['vertices'],
                         'component_distribution_and': get_metric_to_persist(metric_dict['comp_distribution_map'], TwoAnd.name),
                         'component_distribution_or': get_metric_to_persist(metric_dict['comp_distribution_map'], TwoOr.name),
                        'component_distribution_not': get_metric_to_persist(metric_dict['comp_distribution_map'], OneNot.name),
                        'component_distribution_xor': get_metric_to_persist(metric_dict['comp_distribution_map'], TwoXor.name),
                        'degree_distribution': ",".join([(str(degree) + ':' + str(degree_count)) for degree, degree_count in  metric_dict['degree_distribution'].items()]),
                        'avg_vertex_degree': metric_dict['avg_vertex_degree']})


def create_system_description(curr_gates_map, number_of_outputs):

    vertices = 0
    edges = 0
    degree_distribution = {}
    comp_distribution_map = {}
    avg_vertex_degree = 0
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
    degree_distribution = collections.OrderedDict(sorted(degree_distribution.items()))

    # for comp in comp_distribution_map.keys():
    #     input_size = len(curr_gates_map['basic_inputs']) if curr_gates_map.__contains__('basic_inputs') else 0
        # comp_distribution_map[comp] = comp_distribution_map[comp] / (vertices - input_size - number_of_outputs)
    for degree, degree_count in degree_distribution.items():
        avg_vertex_degree += int(degree) * degree_count
    avg_vertex_degree /= sum(degree_distribution.values())

    return {'edges': edges, 'vertices': vertices, 'comp_distribution_map': comp_distribution_map,
            'degree_distribution': degree_distribution, 'avg_vertex_degree': avg_vertex_degree}


if __name__ == '__main__':
    circuit_name = '74181'
    # generate_truth_table(circuit_name)

    expected_gates_map, outputs_list = generate_expected_gates_map(circuit_name)
    write_metrics_to_csv(circuit_name, create_system_description(expected_gates_map, len(outputs_list)))


