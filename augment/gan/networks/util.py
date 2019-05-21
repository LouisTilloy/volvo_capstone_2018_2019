import numpy as np
import scipy.misc
import os
import re
import glob

TRANSFORM_CROP = "crop"
TRANSFORM_RESIZE = "resize"
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

# Return a dictionary with the form { image_path : [[x0 y0 x1 y1 c]] }
def load_data(file_name):
    train_dict = {}
    with open(file_name, "r") as file:
        for line in file:
            info = re.split(",| ", line[:-1])  # the last character is "\n"
            img_path = os.path.dirname(file_name) + "/" + info[0]
            bounding_boxes = []  # with the label
            for i in range(1, len(info), 5):
                bounding_boxes.append([int(value) for value in info[i:i + 5]])
            train_dict[img_path] = bounding_boxes

    return train_dict

def is_pow2(x):
    return x & (x - 1) == 0

def resize_bounding_boxes(bounding_boxes, new_size):
    res = {}
    for path, val in bounding_boxes.items():
        image = imread(path)
        xr = new_size / image.shape[0]
        yr = new_size / image.shape[1]
        
        # Clamp values to [0, new_size) for the bounding box loss function to work.
        x0 = max(0, min(xr*val[0], new_size - 1))
        y0 = max(0, min(yr*val[1], new_size - 1)) 
        x1 = max(0, min(xr*val[2], new_size - 1))
        y1 = max(0, min(yr*val[3], new_size - 1))
        c = val[4]
        res[path] = [x0, y0, x1, y1, c]

    return res

# Produces a list of supported filetypes from the specified directory.
def get_paths(directory):
    paths = []
    for dirpath, _, files in os.walk(directory):
        for f in files:
            if any(f.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                paths.append(os.path.join(dirpath, f))
    return paths

# Loads the image and transforms it to image_size
def get_image(image_path, image_size, input_transform=TRANSFORM_RESIZE):
    return transform(imread(image_path), image_size, input_transform)

# Reads in the image (part of get_image function)
def imread(path):
    return scipy.misc.imread(path, mode="RGB").astype(np.float)

# Transforms the image by cropping and resizing and normalises intensity values between 0 and 1
def transform(image, size=64, input_transform=TRANSFORM_RESIZE):
    if input_transform == TRANSFORM_CROP:
        output = center_crop(image, size)
    elif input_transform == TRANSFORM_RESIZE:
        output = scipy.misc.imresize(image, (size, size), interp="bicubic")
    else:
        output = image
    return byte2norm(np.array(output))

# Crops the input image at the centre pixel
def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.0))
    i = int(round((w - crop_w) / 2.0))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_w, resize_w])

# Takes a set of "images" and creates an array from them.
def merge(images, shape):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * shape[0]), int(w * shape[1]), 3))
    
    for idx, image in enumerate(images):
        i = idx % shape[1]
        j = idx // shape[1]
        img[j*h:(j+1)*h, i*w:(i+1)*w, :] = image

    return img
    
# Save image to disk
def save_image(image, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return scipy.misc.imsave(path, norm2byte(image))

# Save array of images
def save_images(images, paths):
    for img, pth in zip(images, paths):
        save_image(img, pth)

# Save images as a single mosaic
def save_mosaic(images, shape, path):
    return save_image(merge(images, shape), path)

# Redistribute intensity values from [0 1] to [0 255]
def norm2byte(x):
    return (255*x).astype(np.uint8)

# Redistribute intensity values from [0 255] to [0 1]
def byte2norm(x):
    return x/255.0