# import necessary libraries
import os
import cv2
from imutils import paths
import random
import pprint
import time
import numpy as np

# import Custom libraries
from Utility.annot_format import _rename_images_annot2, get_labels
from Utility.helper import *

random.seed(100)


def rectContains(rect, pt):
    logic = rect[0] < pt[0] < rect[0] + rect[2] and rect[1] < pt[1] < rect[1] + rect[3]
    return logic


def main_bkup():
    """
    
    """
    annot_path = r"C:\Data\AI\Door\Training\Check_1\AllAnnotations_check.csv"
    with open(annot_path, 'r') as readObj:
        inlines = readObj.readlines()

    for line in inlines:
        line_split = line.strip().split(",")
        image_path = line_split[0]
        label = line_split[1]
        x1, x2, y1, y2 = list(map(int, line_split[2:]))
        print("label: {}, x1: {}, x2: {}, y1: {}, y2: {}".format(label, x1, x2, y1, y2))
        image_obj = cv2.imread(image_path)
        cv2.imshow("Original", image_obj)
        temp_feature = image_obj[y1:y2, x1:x2]
        cv2.imshow("temp_feature", temp_feature)
        cv2.waitKey(0)


def generate_new_image():
    """

    Returns
    -------

    """


def feature_grab(image_obj, all_coord, label_dict_map, scale_x=1.0, scale_y=1.0, showFlag=False, resizeFlag=True):
    """

    Returns
    -------

    """
    id_label_dict_map = {val: key for key, val in label_dict_map.items()}
    label_check_list = ["Heatstake_7_Top"]
    base_size = image_obj.shape
    image_obj_n = image_obj.copy()
    rect_list = []
    for coord in all_coord:
        coordinates_temp = list(map(int, coord))
        x1, y1, x2, y2, label = coordinates_temp
        rect_list.append((x1, y1, x2 - x1, y2 - y1))

    rescaled_boxes = []
    for coord in all_coord:
        coordinates = list(map(int, coord))
        x1, y1, x2, y2, label = coordinates
        if not id_label_dict_map[label] in label_check_list:
            continue
        img_obj_vis = image_obj.copy()
        # print("coordinates", coordinates)
        cv2.rectangle(img_obj_vis, (x1, y1), (x2, y2), (0, 0, 255), 5)
        temp_feature = image_obj[y1:y2, x1:x2]
        # open_image("Original", img_obj_vis)
        rand_xloc = random.randint(1, base_size[1] // 2)
        rand_yloc = random.randint(1, base_size[0] // 2)
        new_x1, new_y1, new_x2, new_y2 = x1 + rand_xloc, y1 + rand_yloc, x2 + rand_xloc, y2 + rand_yloc

        # break
        # if rectContains()
        if new_x1 > base_size[1] or new_x2 > base_size[1]:
            new_x1 = x1 - rand_xloc
            new_x2 = x2 - rand_xloc

        if new_y1 > base_size[0] or new_y2 > base_size[0]:
            new_y1 = y1 - rand_yloc
            new_y2 = y2 - rand_yloc
        rect_flag = True
        count = 0
        for rect_pts in rect_list:
            if rectContains(rect_pts, [new_x1, new_y1]) or rectContains(rect_pts, [new_x2, new_y2]):
                pass
            else:
                count += 1
                rect_flag = False

        if count == len(rect_list) and not rect_flag:
            print(new_x2 - new_x1, new_y2 - new_y1)
            print(image_obj_n.shape)
            print(temp_feature.shape)
            print(new_x1, new_x2, new_y1, new_y2)
            image_obj_n[new_y1:new_y2, new_x1:new_x2] = temp_feature

    # open_image("image_obj_n", image_obj_n)

    return image_obj_n, all_coord


def main():
    parent_path = r"C:\Data\AI\Door\Training\Check_1"
    annotationPath = os.path.join(parent_path, "AllAnnotations_check.csv")
    labels_dicts, final_str = get_labels(annotationPath)
    annotationLabelsPath = annotationPath.replace(".csv", "_labels.csv")
    with open(annotationLabelsPath, 'w') as writeObj:
        writeObj.writelines(final_str)

    pprint.pprint(labels_dicts)
    counter_, outlines_ = _rename_images_annot2(annotationPath, labels_dicts)

    annotationVisPath = annotationPath.replace(".csv", "_vis.csv")
    with open(annotationVisPath, 'w') as outF:
        outF.writelines(outlines_)

    data_aug_path = os.path.join(parent_path, "DataAug2")
    if not os.path.exists(data_aug_path):
        os.makedirs(data_aug_path)

    with open(annotationVisPath, "r") as readObj:
        outlines = readObj.readlines()

    startTime = time.time()
    final_lines = []
    for idx, line in enumerate(outlines[:]):
        final_lines.append(line)
        print("{}/{}".format(idx + 1, len(outlines)))
        line_split = line.strip().split(" ")
        image_path = line_split[0]
        image_name = image_path[image_path.rfind(os.path.sep) + 1:]
        img_obj = cv2.imread(image_path)
        if not line_split[1:]:
            continue
        bboxes_orig = str2float(line_split[1:])
        bboxesVal = np.array(bboxes_orig)

        x_y_Scale = [(1, 1)]

        for i in range(len(x_y_Scale)):
            # scaleX = np.random.uniform(0.8, 1.2)
            # scaleY = np.random.uniform(0.6, 0.9)
            scaleX, scaleY = x_y_Scale[i]
            resizedObj, resizedBox = feature_grab(img_obj, bboxes_orig, labels_dicts, scale_x=scaleX, scale_y=scaleY,
                                                  resizeFlag=False)
            output_path = os.path.join(data_aug_path, image_name)
            output_path = output_path.replace(".png", "_{:02d}.png".format(i))
            cv2.imwrite(output_path, resizedObj)
    #         exit()
    #         l2 = [list(map(str, aa)) for aa in resizedBox]
    #         l3 = [",".join(a) for a in l2]
    #         box_string = " ".join(l3)
    #         resizedBox = np.array(resizedBox)
    #         # open_image(image_name, draw_rect(resizedObj, resizedBox))
    #         output_path = os.path.join(data_aug_path, image_name)
    #         output_path = output_path.replace(".png", "_{:02d}.png".format(i))
    #         # print("{} {}".format(output_path, box_string))
    #         new_line = "{} {}\n".format(output_path, box_string)
    #         final_lines.append(new_line)
    #         cv2.imwrite(output_path, resizedObj)
    # print("Total Length: ", len(final_lines))
    # dataAug_annotPath = os.path.join(parent_path, "dataAug_Annot2.csv")
    # with open(dataAug_annotPath, 'w') as writeObj:
    #     writeObj.writelines(final_lines)


if __name__ == '__main__':
    print("Happy Coding !!!\n")

    rect = (434, 509, 372, 457)
    pt = [75, 85]
    print(rectContains(rect, pt))
    main()
