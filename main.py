
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from func_def import *
import random


def data_aug_one(annot_lines, aug_map):
    """

    Returns
    -------

    """
    outlines = annot_lines.copy()
    for line in annot_lines:
        # print(line)
        line_split = line.strip().split(" ")
        image_path = line_split[0]
        # print(image_path)
        image_name = image_path[image_path.rfind(os.path.sep) + 1:]
        tempImagePath = os.path.join(outPath, image_name)
        bboxes_orig = list(map(eval, line_split[1:]))
        # bboxes_orig = str2int(line_split[1:])
        # print(bboxes_orig)
        bboxesVal = np.array(bboxes_orig)
        imgObj = cv2.imread(image_path)
        fileName, fileExt = os.path.splitext(tempImagePath)
        for funName in list(aug_map.keys())[:1]:
            print(funName)
            augmentation = aug_map[funName]
            image_box_list = augmentation(imgObj, bboxesVal)
            for idx, (imgObj_out, bboxesVal_out) in enumerate(image_box_list):
                outImagePath = tempImagePath.replace(fileExt, "{}_{}{}".format(funName, idx+1, fileExt))
                # print(outImagePath)
                # print(bboxesVal_out.tolist())
                final_bboxes_str = join_list_of_list(bboxesVal_out.tolist())
                final_line = " ".join([outImagePath, final_bboxes_str]) + "\n"
                outlines.append(final_line)
                # print(final_line)
                # cv2.imwrite(outImagePath, imgObj_out)
                # print(bboxesVal_out)
                # cv2.imshow("Out", draw_rect(imgObj_out, bboxesVal_out))
                # cv2.waitKey(0)
    annot_out_path = annotPath.replace(".csv", "_out.csv")
    with open(annot_out_path, 'w') as outF:
        outF.writelines(outlines)


def data_aug_two(annot_lines, aug_map):
    """

    Returns
    -------

    """
    outlines = annot_lines.copy()
    for funName in list(aug_map.keys()):
        print(funName)
        augmentation = aug_map[funName]
        annot_lines = outlines.copy()
        print(len(annot_lines))
        for line in annot_lines:
            # print(line)
            line_split = line.strip().split(" ")
            image_path = line_split[0]
            # print(image_path)
            image_name = image_path[image_path.rfind(os.path.sep) + 1:]
            tempImagePath = os.path.join(outPath, image_name)
            # bboxes_orig = list(map(eval, line_split[1:]))
            bboxes_orig = str2float(line_split[1:])
            # print(bboxes_orig)
            bboxesVal = np.array(bboxes_orig)
            imgObj = cv2.imread(image_path)
            fileName, fileExt = os.path.splitext(tempImagePath)

            image_box_list = augmentation(imgObj, bboxesVal)
            for idx, (imgObj_out, bboxesVal_out) in enumerate(image_box_list):
                outImagePath = tempImagePath.replace(fileExt, "{}_{}{}".format(funName, idx+1, fileExt))
                # print(outImagePath)
                # print(bboxesVal_out.tolist())
                final_bboxes_str = join_list_of_list(bboxesVal_out.tolist())
                final_line = " ".join([outImagePath, final_bboxes_str]) + "\n"
                outlines.append(final_line)
                # print(final_line)
                cv2.imwrite(outImagePath, imgObj_out)
                # print(bboxesVal_out)
                # cv2.imshow("Out", draw_rect(imgObj_out, bboxesVal_out))
                # cv2.waitKey(0)
    print(len(outlines))
    annot_out_path = annotPath.replace(".csv", "_out2.csv")
    with open(annot_out_path, 'w') as outF:
        outF.writelines(outlines)


def main(annot_path, dataAug_Dict1):
    """Main function to call data augmentation

    Returns
    -------

    """
    with open(annot_path) as inFilObj:
        annot_lines = inFilObj.readlines()

    # data_aug_one(annot_lines, dataAug_Dict1)
    data_aug_two(annot_lines, dataAug_Dict1)



if __name__ == '__main__':

    parentPath = r"/home/umesh/Umesh/DataScience/Data_Augmentation/Git_Repo/data_augmentation"
    outPath = os.path.join(parentPath, "Output")
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    annotPath = r"temp_data.csv"

    dataAugDict = {"HorizontalFlip": HorizontalFlip, "Scale": Scale(num_count=1), "Rotate": Rotate(num_count=10),
                    "Translate": Translate,
                    "Resize": Resize, "Shear": Shear(num_count=10), "RandomHSV": RandomHSV(100, 100, 100, 10)}
    """ Step 1"""
    dataAugDict1 = {"HorizontalFlip": HorizontalFlip(num_count=1), "Rotate": Rotate(num_count=2), "Shear": Shear(num_count=3),
                     "RandomHSV": RandomHSV(100, 100, 100, 2)}
    main(annotPath, dataAugDict1)


