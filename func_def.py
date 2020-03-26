
import cv2
import numpy as np
from Utility.helper import *
import random


class Scale(object):
    """Scales the image

    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.


    Parameters
    ----------
    scale_x: float
        The factor by which the image is scaled horizontally

    scale_y: float
        The factor by which the image is scaled vertically

    Returns
    -------

    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, scale_x = 0.2, scale_y = 0.2, num_count=1):
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.num_count = num_count


    def __call__(self, img, bboxes):
        final_image_box = []
        for i in range(self.num_count):
            bboxes_scale = bboxes.copy()

            # Chose a random digit to scale by

            img_shape = img.shape


            resize_scale_x = 1 + random.uniform(0, 0.2)
            resize_scale_y = 1 + random.uniform(0, 0.2)

            img_scale = cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)

            bboxes_scale[: ,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]

            canvas = np.zeros(img_shape, dtype = np.uint8)

            y_lim = int(min(resize_scale_y ,1 ) *img_shape[0])
            x_lim = int(min(resize_scale_x ,1 ) *img_shape[1])

            canvas[:y_lim ,:x_lim ,:] =  img_scale[:y_lim ,:x_lim ,:]

            img_scale = canvas
            bboxes_scale = clip_box(bboxes_scale, [0 ,0 ,1 + img_shape[1], img_shape[0]], 0.25)
            final_image_box.append([img_scale, bboxes_scale])

        return final_image_box


class HorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*

    Parameters
    ----------
    p: float
        The probability with which the image is flipped


    Returns
    -------

    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, num_count=1):
        self.num_count = num_count
        pass

    def __call__(self, img, bboxes):
        final_image_box = []
        for i in range(self.num_count):
            img_center = np.array(img.shape[:2])[::-1]/2
            img_center = np.hstack((img_center, img_center))

            img = img[:, ::-1, :]
            # print(bboxes)
            # print(img_center)
            bboxes[:, [0, 2]] += 2*(img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w
            final_image_box.append([img, bboxes])
        return final_image_box


class Translate(object):
    """Randomly Translates the image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, translate_x=0.2, translate_y=0.2, diff=False, num_count=1):
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.num_count = num_count

        assert self.translate_x > 0 and self.translate_x < 1
        assert self.translate_y > 0 and self.translate_y < 1

    def __call__(self, img, bboxes):
        final_image_box = []
        for i in range(self.num_count):
            # Chose a random digit to scale by
            img_shape = img.shape

            # translate the image

            # percentage of the dimension of the image to translate
            translate_factor_x = self.translate_x
            translate_factor_y = self.translate_y

            canvas = np.zeros(img_shape).astype(np.uint8)

            # get the top-left corner co-ordinates of the shifted box
            corner_x = int(translate_factor_x * img.shape[1])
            corner_y = int(translate_factor_y * img.shape[0])

            # change the origin to the top-left corner of the translated box
            orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]),
                              min(img_shape[1], corner_x + img.shape[1])]

            mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]),
                   max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]
            canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
            img = canvas

            bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

            bboxes = clip_box(bboxes, [0, 0, img_shape[1], img_shape[0]], 0.25)

            final_image_box.append([img, bboxes])
        return final_image_box


class Rotate(object):
    """Rotates an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    angle: float
        The angle by which the image is to be rotated


    Returns
    -------

    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, angle=10, num_count=1):
        self.angle = angle
        self.num_count = num_count


    def __call__(self, img, bboxes):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.


        """
        final_image_box = []
        for i in range(self.num_count):
            # angle = self.angle
            angle = int(random.random()*100)
            # print(angle)

            w, h = img.shape[1], img.shape[0]
            cx, cy = w // 2, h // 2

            corners = get_corners(bboxes)

            corners = np.hstack((corners, bboxes[:, 4:]))

            img_rotate = rotate_im(img, angle)

            corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, h, w)

            new_bbox = get_enclosing_box(corners)

            scale_factor_x = img_rotate.shape[1] / w

            scale_factor_y = img_rotate.shape[0] / h

            img_rotate = cv2.resize(img_rotate, (w, h))

            new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

            bboxes_rotate = new_bbox

            bboxes_rotate = clip_box(bboxes_rotate, [0, 0, w, h], 0.25)
            final_image_box.append([img_rotate, bboxes_rotate])

        return final_image_box


class Shear(object):
    """Shears an image in horizontal direction


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    shear_factor: float
        Factor by which the image is sheared in the x-direction

    Returns
    -------

    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, shear_factor=0.2, num_count=1):
        self.shear_factor = shear_factor
        self.num_count = num_count


    def __call__(self, img, bboxes):
        final_image_box = []
        # int_list = random.sample(range(1, 100), self.num_count)
        # float_list = [x / 100 for x in int_list]

        for i in range(self.num_count):
            bboxes_shear = bboxes.copy()
            # shear_factor = self.shear_factor
            # shear_factor = float_list[i]
            shear_factor = random.random()
            # print(shear_factor)
            if shear_factor < 0:
                img, bboxes_shear = HorizontalFlip()(img, bboxes_shear)

            M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])
            # print(M)

            nW = img.shape[1] + abs(shear_factor * img.shape[0])
            # print(img.shape[1], abs(shear_factor * img.shape[0]), nW)

            bboxes_shear[:, [0, 2]] += ((bboxes_shear[:, [1, 3]]) * abs(shear_factor)).astype(int)
            # print(bboxes_shear)

            img_shear = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

            if shear_factor < 0:
                img_shear, bboxes_shear = HorizontalFlip()(img_shear, bboxes_shear)
            final_image_box.append([img_shear, bboxes_shear])
            # print(final_image_box)
        return final_image_box


class Resize(object):
    """Resize the image in accordance to `image_letter_box` function in darknet

    The aspect ratio is maintained. The longer side is resized to the input
    size of the network, while the remaining space on the shorter side is filled
    with black color. **This should be the last transform**


    Parameters
    ----------
    inp_dim : tuple(int)
        tuple containing the size to which the image will be resized.

    Returns
    -------

    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, inp_dim=300, num_count=1):
        self.inp_dim = inp_dim
        self.num_count = num_count


    def __call__(self, img, bboxes):
        final_image_box = []
        for i in range(self.num_count):
            w, h = img.shape[1], img.shape[0]
            img = letterbox_image(img, self.inp_dim)

            scale = min(self.inp_dim / h, self.inp_dim / w)
            bboxes[:, :4] *= (scale)

            new_w = scale * w
            new_h = scale * h
            inp_dim = self.inp_dim

            del_h = (inp_dim - new_h) / 2
            del_w = (inp_dim - new_w) / 2

            add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)

            bboxes[:, :4] += add_matrix

            img = img.astype(np.uint8)
            final_image_box.append([img, bboxes])
        return final_image_box


class RandomHSV(object):
    """HSV Transform to vary hue saturation and brightness

    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255.
    Chose the amount you want to change thhe above quantities accordingly.




    Parameters
    ----------
    hue : None or int or tuple (int)
        If None, the hue of the image is left unchanged. If int,
        a random int is uniformly sampled from (-hue, hue) and added to the
        hue of the image. If tuple, the int is sampled from the range
        specified by the tuple.

    saturation : None or int or tuple(int)
        If None, the saturation of the image is left unchanged. If int,
        a random int is uniformly sampled from (-saturation, saturation)
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.

    brightness : None or int or tuple(int)
        If None, the brightness of the image is left unchanged. If int,
        a random int is uniformly sampled from (-brightness, brightness)
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.

    Returns
    -------

    numpy.ndaaray
        Transformed image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, hue=None, saturation=None, brightness=None, num_count=1):
        if hue:
            self.hue = hue
        else:
            self.hue = 0

        if saturation:
            self.saturation = saturation
        else:
            self.saturation = 0

        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0

        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)

        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)

        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)

        self.num_count = num_count

    def __call__(self, img, bboxes):
        final_image_box = []
        for i in range(self.num_count):
            hue = random.randint(*self.hue)
            saturation = random.randint(*self.saturation)
            brightness = random.randint(*self.brightness)

            img = img.astype(int)

            a = np.array([hue, saturation, brightness]).astype(int)
            img += np.reshape(a, (1, 1, 3))

            img = np.clip(img, 0, 255)
            img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

            img = img.astype(np.uint8)
            final_image_box.append([img, bboxes])
        return final_image_box

