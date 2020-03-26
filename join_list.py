
import os
import numpy as np


def _join_list_of_list(lst):
    """Joining list of list with space separator

    :param lst:
    :return:
    """

    int_list = [list(map(int, each)) for each in lst]
    # print(*int_list, sep="\n")
    str_list = [",".join(map(str, each)) for each in int_list]
    # print(*int_list, sep="\n")
    # print(str_list)
    final_str = " ".join(str_list)
    # print(final_str)
    return final_str


def _str2int(str_lst):
    """Convert string into int

    :param str_lst:
    :return:
    """
    final_list = []
    for each in line_split:
        print(each)
        each_split = each.split(",")
        temp_list = list(map(int, map(float, each_split[:-1])))
        temp_list.append(each_split[-1])
        final_list.append(temp_list)

    print(*final_list, sep="\n")
    return final_list


if __name__ == '__main__':

    # lst1 = [[120.0, 68.00001749999998, 877.0, 478.99982249999994, 0.0], [222.0, 21.0, 974.0, 486.0, 0.0],
    #         [664.0, 77.0, 1068.0, 336.0, 0.0], [1045.0, 327.0, 1183.0, 396.0, 1.0]]
    # _join_list_of_list(lst1)
    line_split = ['53.0,68.00001749999998,405.0,478.99982249999994,label1', '202,21,496,486,label1',
                  '589,77,737,336,label1', '723,327,793,396,label2']

    _str2int(line_split)