import os
import re
import cv2
import sys
import numpy as np
from tqdm import tqdm

def load_data(file_name):
    dct = dict()
    with open(file_name, "r") as file:
        for line in file:
            info = re.split(",| ", line.rstrip())
            pth_img = info[0]
            bbs = []  # with the label
            for i in range(1, len(info), 5):
                bbs.append([int(value) for value in info[i:i + 5]])
            dct[pth_img] = bbs
    return dct

def save_data(file_name, data):
    with open(file_name, "w") as f:
        for img, bbs in data.items():
            f.write(img)
            for bb in bbs:
                f.write(" %s,%s,%s,%s,%s\n" % (bb[0], bb[1], bb[2], bb[3], bb[4]))

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def augment_img(img, bboxes):
    img_dark = adjust_gamma(img, gamma=0.2)
    arr = img_dark * np.array([0.4, 0.4, 0.4])
    img_res = (255 * arr / arr.max()).astype(np.uint8)

    img_top = img[0:int(img.shape[0]*0.5), :]
    img_top = adjust_gamma(img_top, gamma=0.5)
    img_top_dark = img_top * np.array([0.10, 0.10, 0.10])
    img_res [0:int(img.shape[0]*0.5), :] = img_top_dark

    for k in bboxes:
        img_v4 = img[k[1]:k[3], k[0]:k[2]]
        img_v4_dark = adjust_gamma(img_v4, gamma=0.8)
        img_res[k[1]:k[3], k[0]:k[2]] = img_v4_dark

    return img_res

def augment(path_ann, dir_out, dir_name):
    dct_ann = load_data(path_ann)
    ann_file = os.path.join(dir_out, dir_name + ".txt")

    for path, bboxes in tqdm(dct_ann.items()):
        # Read and augment source image
        path_abs = os.path.join(os.path.dirname(path_ann), path)
        img_src = cv2.imread(path_abs)
        img_aug = augment_img(img_src, bboxes)
        
        # Create new path name
        path_list = os.path.normpath(path).split(os.sep)
        path_list[0] = dir_name
        new_path = os.path.join(*path_list)
        new_path_abs = os.path.join(dir_out, new_path)
        new_dir = os.path.dirname(new_path_abs)
        
        # Create new path directory
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        cv2.imwrite(new_path_abs, img_aug)
        
        with open(ann_file, "a") as f:
            f.write(new_path)
            for b in bboxes:
                f.write(" %s,%s,%s,%s,%s\n" % (b[0], b[1], b[2], b[3], b[4]))

def main():
    argv = sys.argv
    ann_path = argv[1]
    ann_name = os.path.splitext(os.path.basename(ann_path))[0]
    dir_out = argv[2]
    dir_name = ann_name + "_saug"
    augment(ann_path, dir_out, dir_name)

if __name__ == "__main__":
    main()