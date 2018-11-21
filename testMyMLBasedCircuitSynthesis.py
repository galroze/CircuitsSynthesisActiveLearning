from myMLBasedCircuitsSynthesis_V2_active_learning import process_oa
from myMLBasedCircuitsSynthesis_V2_active_learning import GateFeature
from myMLBasedCircuitsSynthesis_V2_active_learning import OA
from myMLBasedCircuitsSynthesis_V2_active_learning import merge_values



if __name__ == '__main__':

    oa_path = "E:\\ise_masters\\gal_thesis\\data_sets\\oa\\"
    oa_by_strength_map = {}
    for strength_index in range(2, 3 + 1):
        with open(oa_path + str(strength_index) + ".txt") as oa_file:
            oa_array = oa_file.read().split('\n').copy()
            oa_num_of_att = len(oa_array[0])
            oa_by_strength_map[strength_index] = OA(oa_array, oa_num_of_att)

    x = GateFeature(None, 'x')
    y = GateFeature(None, 'y')
    z = GateFeature(None, 'z')
    p = GateFeature(None, 'p')

    # process_oa(None, [x, y], oa_by_strength_map[2], [])
    list = (merge_values([], []))