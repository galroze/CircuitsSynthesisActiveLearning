import pandas
import math
from LogicTypes import OneNot
from LogicTypes import TwoXor
from LogicTypes import TwoAnd
from LogicTypes import ThreeAnd
from LogicTypes import FourAnd
from LogicTypes import TwoOr
from LogicTypes import ThreeOr
from LogicTypes import FourOr
from LogicTypes import TwoNor
from LogicTypes import ThreeNor
from LogicTypes import FourNor
from LogicUtils import *


if __name__ == '__main__':

    fileName = "E:\\ise_masters\\gal_thesis\\data_sets\\circuits\\Data_Systems\\74182.sys"
    lines = [line.rstrip('\n') for line in open(fileName)]

    logic_types = {OneNot.name: OneNot, TwoXor.name: TwoXor, TwoAnd.name: TwoAnd, ThreeAnd.name: ThreeAnd,
                       FourAnd.name: FourAnd, TwoOr.name: TwoOr, ThreeOr.name: ThreeOr, FourOr.name: FourOr, TwoNor.name: TwoNor, ThreeNor.name: ThreeNor, FourNor.name: FourNor}

    inputs = lines[1].split(sep=',')
    inputs[0] = inputs[0].replace('[', '')
    inputs[len(inputs) - 1] = inputs[len(inputs) - 1].replace('].', '')

    outputs = lines[2].split(sep=',')
    outputs[0] = outputs[0].replace('[', '')
    outputs[len(outputs) - 1] = outputs[len(outputs) - 1].replace('].', '')

    truth_table_columns = inputs.copy()
    truth_table = pandas.DataFrame(columns=truth_table_columns, dtype=bool)

    rows = math.pow(2, len(inputs))

    for i in range(int(rows)):
        new_row = []
        for j in range(len(inputs) - 1, -1, -1):
            new_row.append(False if (int(i / math.pow(2, j)) % 2) == 0 else True)
        truth_table = truth_table.append(pandas.Series(new_row, index=truth_table.columns), ignore_index=True)

    for row_index in range(3, len(lines)):
        split_row = lines[row_index].split(sep=',')
        if row_index < len(lines) - 1:
            split_row.remove(split_row[len(split_row) - 1])
            gate_name = split_row[0].replace('[', '')
        output = split_row[2]
        inputs = split_row[3:]
        inputs[len(inputs) - 1] = inputs[len(inputs) - 1].replace(']', '').replace(',', '').replace('.', '')
        # print("gate " + str(gate_name) + " output " + str(output) + " input " + str(inputs))

        gate = logic_types.get(gate_name)
        output_column = get_transformed_att_value(truth_table, inputs, gate)
        truth_table.insert(len(truth_table.columns), output, output_column)

    truth_table = truth_table[truth_table.columns[pandas.Series(truth_table.columns).str.startswith('z') == False]]
    truth_table.to_csv(TRUTH_TABLE_PATH_74182, sep='\t', index=False)
    print("Done generating: " + str(TRUTH_TABLE_PATH_74182) + " \ninputs: " + str(get_input_names(truth_table)) + " outputs: " + str(get_output_names(truth_table)))
