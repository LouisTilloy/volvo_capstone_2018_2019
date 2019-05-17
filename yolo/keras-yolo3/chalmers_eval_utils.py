"""
Utils for the yolo_evaluaation notebook
"""
import os
import numpy as np
import re
from keras import backend as K
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
from datetime import datetime
import tensorflow as tf
import colorsys

from yolo import YOLO
from yolo3.utils import letterbox_image

class YOLOPlus(YOLO):
    def __init__(self, **kwargs):
        super(YOLOPlus, self).__init__(**kwargs)

    def detect_all_signs(self, image):
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

        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
            
        return out_classes, out_scores, out_boxes

def detect_img(img, yolo):
    """
    all_bool: whether or not to detect all signs in the picture
    """
    try:
        image = Image.open(img)
    except:
        print('Open Error! Try again!')
        return
    else:
        labels, scores, boxes = yolo.detect_all_signs(image)
        return labels, scores, boxes

def save_image(img, save_dir, class_names, colors, predictions):
    out_boxes = [p[0:4] for p in predictions]
    out_classes = [p[4] for p in predictions]
    out_scores = [p[5] for p in predictions]

    image = Image.open(img)

    font = ImageFont.truetype(font="font/FiraMono-Medium.otf",
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype("int32"))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = "{} {:.2f}".format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        left, top, right, bottom = box
        top = max(0, np.floor(top + 0.5).astype("int32"))
        left = max(0, np.floor(left + 0.5).astype("int32"))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype("int32"))
        right = min(image.size[0], np.floor(right + 0.5).astype("int32"))
        # print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    image.save(save_dir + "/" + os.path.basename(img))
    
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

    # Compute the intersection over union by taking the intersection
    return interArea / float(boxAArea + boxBArea - interArea)
  
def get_mode(x):
    n_pass = 0
    n_fail = 0
    for v in x:
        if v == 0:
            n_pass += 1
        else:
            n_fail += 1
    return 0 if n_pass > n_fail else 1

def get_timestamp():
    return datetime.now().strftime("%H%M%S")

def get_latex(x):
    return x.replace("_", r"\_")

def get_colors(class_names):
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / len(class_names), 1., 1.)
                    for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.
    return colors

# Load the testing data from the .txt file.
# This method will also add the directory name of the .txt-file at the start of the path strings.
# This allows for the .txt to be placed anywhere in your file system as opposed to the same folder as this .py file.
# Make sure that the .txt is placed relative to the paths the entries are pointing to however. 
def load_data(file_name):
    train_dict = {}
    directory = os.path.dirname(file_name)
    with open(file_name, "r") as file:
        for line in file:
            info = re.split(",| ", line[:-1])  # the last character is '\n'
            bounding_boxes = []  # with the label
            for i in range(1, len(info), 5):
                bounding_boxes.append([int(value) for value in info[i:i + 5]])

            train_dict[directory + "/" + info[0]] = bounding_boxes

    return train_dict

def load_classes(class_file):
    classes = []
    with open(class_file, "r") as file:
        for line in file:
            classes.append(line[:-1])
    return classes
    
def load_log_dir(log_dir):
    log = []
    for file in os.listdir(log_dir):
        log = log + load_log_data(os.path.join(log_dir, file))
    
    return sorted(log, key = lambda i: i["step"])

def load_log_data(log_path):
    log_dict = {}

    for e in tf.train.summary_iterator(log_path):
        if not e.step in log_dict:
            log_dict[e.step] = {"step" : e.step, "loss" : 0.0, "val_loss" : 0.0, "lr" : 0.0}

        for v in e.summary.value:
            log_dict[e.step][v.tag] = v.simple_value
    
    return list(log_dict.values())

def merge(images, shape):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * shape[0]), int(w * shape[1]), 3))
    
    for idx, image in enumerate(images):
        i = idx % shape[1]
        j = idx // shape[1]
        img[j*h:(j+1)*h, i*w:(i+1)*w, :] = image

    return img