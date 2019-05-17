import re
import cv2
import numpy as np
import os

def adjust_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(img, table)

def augment(img, bboxes):
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

def load_data(file_name):
    dct = dict()
    with open(file_name, "r") as file:
        for line in file:
            info = re.split(",| ", line[:-1])  # the last character is '\n'
            pth_img = info[0]
            bbs = []  # with the label
            for i in range(1, len(info), 5):
                bbs.append([int(value) for value in info[i:i + 5]])
            dct[pth_img] = bbs
    return dct

def main():
    pth_ann = "lisa.txt"
    dct_ann = load_data(pth_ann)

    for path, bboxes in dct_ann.items():
        # Read and augment source image
        img_src = cv2.imread(path)
        img_aug = augment(img_src, bboxes)
        # Create new path name
        path_list = os.path.normpath(path).split(os.sep)
        path_list[0] += "_augmented"
        new_path = os.path.join(*path_list)
        new_dir = os.path.dirname(new_path)
        print(new_path)
        # Create new path directory
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        cv2.imwrite(new_path, img_aug)

if __name__ == "__main__":
    main()