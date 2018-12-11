from LogicUtils import *
import itertools
import random
import sys


def  write_oa_to_file(new_oa_array, target_input_size, target_strength, oa_path):
    new_file_name = oa_path + 'GEN_' + str(len(new_oa_array)) + '_' + str(target_input_size) + '_2_' + str(
        target_strength) + '.txt'
    with open(new_file_name, "w+") as the_file:
        for line_index in range(len(new_oa_array)):
            write_str = new_oa_array[line_index]
            if line_index < len(new_oa_array) - 1:
                write_str += '\n'
            the_file.write(write_str)

    print('file ' + new_file_name + ' generated')


def get_relevant_lines(target_strength, target_input_size, src_oa, all_possible_order_list):

    num_of_instances_per_tuple = pow(target_strength, 2)
    tuple_dict = {}
    all_combinations = list(itertools.combinations(list(range(0, target_input_size)), target_strength))
    for comb_tuple in all_combinations:
        tuple_dict[comb_tuple] = set()

    min_length_oa = sys.maxsize
    best_oa_array = []
    for possible_order in all_possible_order_list:
        curr_tuple_dict = tuple_dict.copy()
        new_oa_array = []
        for oa_line_index in range(len(possible_order)):
            curr_oa_line = src_oa.array[possible_order[oa_line_index]]
            used_line = False
            for comb_tuple in all_combinations:
                if len(curr_tuple_dict[comb_tuple]) < num_of_instances_per_tuple:
                    curr_str = ""
                    for curr_index in list(comb_tuple):
                        curr_str += curr_oa_line[curr_index]
                    if not curr_tuple_dict[comb_tuple].__contains__(curr_str):
                        curr_tuple_dict[comb_tuple].add(curr_str)
                        used_line = True
            if used_line:
                new_oa_array.append(curr_oa_line)

        if (len(new_oa_array) > 0) & (len(new_oa_array) < min_length_oa):
            min_length_oa = len(new_oa_array)
            best_oa_array = new_oa_array

    return best_oa_array


def get_possible_order(oa_list):
    oa = set(oa_list)
    order_list = []
    for iteration in range(0, 100):
        order_list.append(random.sample(range(len(oa)), len(oa)))
    return order_list
    # return list(itertools.product(list(range(0, len(oa))), repeat=len(oa)))


if __name__ == '__main__':

    oa_path = "E:\\ise_masters\\gal_thesis\\data_sets\\oa\\"
    oa_source_file_name = "2048_16_2_7.txt"
    target_input_size = 11
    target_strength = 4
    source_oa = read_oa(oa_path + oa_source_file_name)

    trimmed_src_oa = trim_oa_to_fit_inputs(source_oa, target_input_size)

    all_possible_order_list = get_possible_order(trimmed_src_oa.array)

    new_oa_array = get_relevant_lines(target_strength, target_input_size, trimmed_src_oa, all_possible_order_list)
    write_oa_to_file(new_oa_array, target_input_size, target_strength, oa_path)






