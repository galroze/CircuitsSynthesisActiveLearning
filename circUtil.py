from LogicTypes import *

CIRC_FILE_PATH = "E:\\ise_masters\\gal_thesis\\data_sets\\circuits\\circ_for_orig_gate_map_read\\"


def get_gate_class(gate_name):
    if gate_name.__eq__("inv"):
        return OneNot
    elif gate_name.__eq__("and"):
        return TwoAnd
    elif gate_name.__eq__("or"):
        return TwoOr
    elif gate_name.__eq__("xor"):
        return TwoXor


def generate_gates_recursively(gate_class, args, gates_map):
    gate_feature = GateFeature(gate_class, args[:gate_class.numInputs])

    new_args = [gate_feature]
    new_args.extend(args[gate_class.numInputs:])

    if len(new_args) > 1:
        if not gates_map.__contains__(gate_class.name):
            gates_map[gate_class.name] = []
        gates_map[gate_class.name].append(gate_feature)
        return generate_gates_recursively(gate_class, new_args, gates_map)
    else:
        return gate_feature


def generate_gate_feature(gate_class, args, gates_map, aux_map):
    converted_args = []
    for arg in args:
        if arg in aux_map:
            arg = aux_map[arg]
        converted_args.append(arg)

    if len(converted_args) > gate_class.numInputs:
        gate_feature = generate_gates_recursively(gate_class, converted_args, gates_map)
    else:
        gate_feature = GateFeature(gate_class, converted_args)

    if not gates_map.__contains__(gate_class.name):
        gates_map[gate_class.name] = []
    gates_map[gate_class.name].append(gate_feature)

    return gate_feature


def parse_row(row, gates_map, aux_map):
    split_row = row.split(sep=' = ')
    id = split_row[0]
    gate_str = split_row[1]

    gate_name = gate_str.split(sep='(')[0]
    gate_class = get_gate_class(gate_name)

    args = gate_str.split(sep='(')[1].split(sep=')')[0].split(sep=', ')
    gate_feature = generate_gate_feature(gate_class, args, gates_map, aux_map)
    return id, gate_feature


def generate_gates_map_from_circ_file(circ_name, inputs, outputs):
    file_name = CIRC_FILE_PATH + circ_name + ".txt"
    lines = [line.rstrip('\n') for line in open(file_name)]

    gates_map = {}
    aux_map = {}
    for inp in inputs:
        aux_map[inp] = GateFeature(None, [inp])
    for row in lines:
        id, gate_feature = parse_row(row, gates_map, aux_map)
        if id not in outputs:
            aux_map[id] = gate_feature
    return gates_map


# if __name__ == '__main__':
    # outputs = 5
    # generate_gates_map_from_circ_file('demux5', ['o' + str(n) for n in range(1, 1 + outputs)])