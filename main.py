
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from func_def import *
import random


# def draw_rect(im, cords, color=None):
#     """Draw the rectangle on the image
#
#     Parameters
#     ----------
#
#     im : numpy.ndarray
#         numpy image
#
#     cords: numpy.ndarray
#         Numpy array containing bounding boxes of shape `N X 4` where N is the
#         number of bounding boxes and the bounding boxes are represented in the
#         format `x1 y1 x2 y2`
#
#     Returns
#     -------
#
#     numpy.ndarray
#         numpy image with bounding boxes drawn on it
#
#     """
#
#     im = im.copy()
#
#     cords = cords[:, :4]
#     cords = cords.reshape(-1, 4)
#     if not color:
#         color = [255, 255, 255]
#     for cord in cords:
#         pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
#
#         pt1 = int(pt1[0]), int(pt1[1])
#         pt2 = int(pt2[0]), int(pt2[1])
#
#         im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2]) / 200))
#     return im
#
#
# def horizontal_flip(img, bboxes):
#     img_center = np.array(img.shape[:2])[::-1 ] /2
#     img_center = np.hstack((img_center, img_center))
#
#     img = img[:, ::-1, :]
#     bboxes[:, [0, 2]] += 2* (img_center[[0, 2]] - bboxes[:, [0, 2]])
#
#     box_w = abs(bboxes[:, 0] - bboxes[:, 2])
#
#     bboxes[:, 0] -= box_w
#     bboxes[:, 2] += box_w
#
#     return img, bboxes


if __name__ == '__main__':

    parentPath = r"/home/umesh/Umesh/DataScience/Data_Augmentation"
    outPath = os.path.join(parentPath, "Output")
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    annot_path = r"/home/umesh/Umesh/DataScience/Data_Augmentation/temp_data.csv"

    dataAug_Dict = {"HorizontalFlip": HorizontalFlip, "Scale": Scale(num_count=1), "Rotate": Rotate(num_count=10),
                    "Translate": Translate,
                    "Resize": Resize, "Shear": Shear(num_count=10), "RandomHSV": RandomHSV(100, 100, 100, 10)}
    """ Step 1"""
    dataAug_Dict1 = {"HorizontalFlip": HorizontalFlip, "Rotate": Rotate(num_count=10), "Shear": Shear(num_count=10),
                     "RandomHSV": RandomHSV(100, 100, 100, 10)}

    with open(annot_path) as inFilObj:
        dataset = inFilObj.readlines()

    for line in dataset:
        # print(line)
        line_split = line.strip().split(" ")
        image_path = line_split[0]
        # print(image_path)
        image_name = image_path[image_path.rfind(os.path.sep) + 1:]
        tempImagePath = os.path.join(outPath, image_name)
        bboxes_orig = list(map(eval, line_split[1:]))
        # bboxes_orig = str2int(line_split[1:])
        print(bboxes_orig)
        bboxesVal = np.array(bboxes_orig)
        # print(bboxesVal)
        # imgObj = cv2.imread(image_path)[:, :, ::-1]  # OpenCV uses BGR channels
        imgObj = cv2.imread(image_path)
        # data_aug_list = ["HorizontalFlip", "Scale", "Rotate", "Translate", "Resize", "Shear", "RandomHSV"]
        data_aug_list = ["HorizontalFlip", "Rotate", "Shear", "RandomHSV"]
        fileName, fileExt = os.path.splitext(tempImagePath)
        for funName in data_aug_list[2:3]:
            print(funName)
            augmentation = dataAug_Dict1[funName]
            # imgObj_out, bboxesVal_out = augmentation(imgObj, bboxesVal)
            image_box_list = augmentation(imgObj, bboxesVal)
            for idx, (imgObj_out, bboxesVal_out) in enumerate(image_box_list):
                outImagePath = tempImagePath.replace(fileExt, "{}_{}{}".format(funName, idx+1, fileExt))
                print(outImagePath)
                print(bboxesVal_out.tolist())
                final_bboxes_str = join_list_of_list(bboxesVal_out.tolist())
                final_line = " ".join([outImagePath, final_bboxes_str]) + "\n"
                print(final_line)
                # cv2.imwrite(outImagePath, imgObj_out)
                # print(bboxesVal_out)
                # cv2.imshow("Out", draw_rect(imgObj_out, bboxesVal_out))
                # cv2.waitKey(0)



        # horFlipObj = HorizontalFlip()

        # outImagePath = tempImagePath.replace(fileExt, "_horFlip"+fileExt)
        # print(outImagePath)
        # imgObj_out, bboxesVal_out = horFlipObj(imgObj, bboxesVal)
        # cv2.imwrite(outImagePath, imgObj_out)
        # # cv2.imshow("Out", draw_rect(imgObj_out, bboxesVal_out))
        # # cv2.waitKey(0)
        # scaleObj = Scale()
        # outImagePath = tempImagePath.replace(fileExt, "_scale"+fileExt)
        # print(outImagePath)
        # imgObj_out, bboxesVal_out = scaleObj(imgObj, bboxesVal)
        # cv2.imwrite(outImagePath, imgObj_out)
