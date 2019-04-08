"""
Utils for the yolo_evaluaation notebook
"""
import numpy as np
import re
import seaborn as sns
from keras import backend as K
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

from yolo import YOLO
from yolo3.utils import letterbox_image, rgb_2_gray


class YOLOPlus(YOLO):
    def __init__(self, gray_scale=False, **kwargs):
        self.gray_scale = gray_scale
        super(YOLOPlus, self).__init__(**kwargs)

    def detect_all_signs(self, image, score_threshold):
        """
        Will draw all the bounding boxes and return the list of classes, confidence scores
        in the boxes and the boxes themselves.
        """
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0] % 32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1] % 32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')
        if self.gray_scale:
            image_data = rgb_2_gray(image_data)

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        list_c, list_score, list_box = [], [], []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            if score < score_threshold:
                continue

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            # print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

            list_c.append(c)
            list_score.append(score)
            list_box.append((left, top, right, bottom))  # x_min, y_min, x_max, y_max

        return image, list_c, list_score, list_box  # (returns this if no sign is found)


def detect_img(img, yolo, score_threshold):
    """
    all_bool: whether or not to detect all signs in the picture
    """
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        return
    else:
        r_images, labels, scores, boxes = yolo.detect_all_signs(image, score_threshold)
        return r_images, labels, scores, boxes


def IoU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def prediction_not_ok(score, true_label, true_box, label, box):
    """
    For 1 prediction (1 score, 1 label, 1 box),
    returns 0 if the prediction is considered True
    returns 1 if the prediction has the wrong label
    returns 2 if the prediction has the wrong Bounding Box
    returns 3 if the prediction has the wrong label and the wrong Bounding Box
    returns 4 if the prediction is not confident enough
    returns 5 if the prediction has a None inside (i.e. if an unexpected error happened)
    """
    if score is None or label is None or box is None:
        return 5

    if score < 0.3:
        return 4

    if label == true_label:
        label_ok = True
    else:
        label_ok = False

    if IoU(true_box, box) >= 0.5:
        iou_ok = True
    else:
        iou_ok = False

    if label_ok is iou_ok is False:
        return 3

    if not label_ok:
        return 1

    if not iou_ok:
        return 2

    return 0


def load_data(file_name):
    """
    Load data from train_lisa.txt or a similar structure file.
    :return: (dict{file_name: [bounding box 1 + label, bounding box 2 + label, ...]},
              [file_name_1, file_name_2, ...])
    """
    train_dict = dict()
    train_imgs = []  # useful to keep an ordering of the imgs
    with open(file_name, "r") as file:
        for line in file:
            info = re.split(",| ", line[:-1])  # the last character is '\n'
            img_path = info[0]
            train_imgs.append(img_path)
            bounding_boxes = []  # with the label
            for i in range(1, len(info), 5):
                bounding_boxes.append([int(value) for value in info[i:i + 5]])

            train_dict[img_path] = bounding_boxes

    return train_dict, train_imgs


def load_classes(class_file):
    """
    load the class file and put the classes in a list.
    """
    classes = []
    with open(class_file, "r") as file:
        for line in file:
            classes.append(line[:-1])
    return classes


def plot_bootstrap_curve(accuracy, n_data, boot_size=100000):
    n_good = int(n_data * accuracy)
    data = np.array([1] * n_good + (n_data - n_good) * [0])

    boot_samples = np.random.choice(data, (boot_size, n_data), replace=True)
    accs = np.mean(boot_samples, axis=1)

    sns.distplot(accs, hist=False)

