import os
import re
import cv2
import shutil
import argparse
import subprocess
from PIL import Image

from yolo.keras_yolo3.yolo3.utils import auto_augment

AUG_BLEND = "BLEND"
AUG_SAUG = "SAUG"
AUG_CG = "CG"
AUG_CG_INS = "CG_INS"
AUG_BBG = "BBG"
AUG_AAUG = "AAUG"
AUG_AAUG_BBG = "AAUG_BBG"

parser = argparse.ArgumentParser(description="Augment data set.")
parser.add_argument("method", metavar="method", type=str, help="The augmentation method.")
parser.add_argument("path_ann", metavar="annotations.txt", type=str, help="The path to the txt-file containing the annotations relative to the data_in directory.")
args = parser.parse_args()

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
            for b in bbs:
                f.write(" %s,%s,%s,%s,%s\n" % (b[0], b[1], b[2], b[3], b[4]))
                    
# Produces a list of supported filetypes from the specified directory.
def get_paths(directory):
    SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]
    paths = []
    for dirpath, _, files in os.walk(directory):
        for f in files:
            if any(f.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                paths.append(os.path.join(dirpath, f))
    return paths
    
def augment(method, path, path_ann, dir_out):
    if (method == AUG_BLEND):
        aug_blend(path, path_ann, dir_out)
    elif (method == AUG_SAUG):
        aug_saug(path, path_ann, dir_out)
    elif (method == AUG_CG):
        aug_cg(path, path_ann, dir_out)
    elif (method == AUG_CG_INS):
        aug_cg(path, path_ann, dir_out, insert=True)
    elif (method == AUG_BBG):
        aug_bbg(path, path_ann, dir_out)
    elif (method == AUG_AAUG):
        aug_aaug(path, path_ann, dir_out)
    elif (method == AUG_AAUG_BBG):
        pass

def aug_blend(path, path_ann, dir_out):
    path_blend = os.path.join(path, "augment/blend/augment.blend")
    path_script = os.path.join(path, "augment/blend/scripts/main.py")
    subprocess.Popen(["blender", path_blend, "--background", "--python", path_script, "--", path_ann, dir_out], shell=True)

def aug_saug(path, path_ann, dir_out):
    path_script = os.path.join(path, "augment/saug/saug.py")
    subprocess.Popen(["python", path_script, path_ann, dir_out], shell=True)

def aug_cg(path, path_ann, dir_out, insert=False):
    path_script = os.path.join(path, "augment/cg/pytorch-CycleGAN-and-pix2pix/test.py")
    dir_script = os.path.dirname(path_script)
    
    dir_in = os.path.dirname(path_ann)
    dir_temp_old = os.path.join(dir_in, "temp_cg")
    os.makedirs(dir_temp_old, exist_ok=True)
    
    suffix = "_cg_ins" if insert else "_ins"

    d_old, d_new = copy_annotations(path_ann, dir_out, suffix)

    for k in d_old.keys():
        name  = os.path.basename(k)
        s = os.path.join(dir_in, k)
        t = os.path.join(dir_temp_old, k)
        os.makedirs(os.path.dirname(t), exist_ok=True)
        shutil.copy(s, t)

    paths = get_paths(dir_temp_old)
    n = str(len(paths))
    dir_temp_new = os.path.join(dir_out, "temp_cg")

    subprocess.call(["python", path_script, "--dataroot", dir_temp_old, "--name", "day2night", "--model", "test", "--no_dropout", "--num_test", n, "--results_dir", dir_temp_new], shell=True, cwd=dir_script)

    shutil.rmtree(dir_temp_old)
    
    dir_model = os.path.join(dir_temp_new, "day2night")
    img_dir = os.path.join(dir_model, "test_latest", "images")
    img_paths = get_paths(img_dir)

    lookup_new = {os.path.splitext(os.path.basename(x))[0] : x for x in d_new.keys()}

    for img in img_paths:
        name = os.path.splitext(os.path.basename(img))[0]
        if name.endswith("_fake_B"):
            name = name[:-7]
            path_new  = lookup_new[name]
            path_new = os.path.join(dir_out, path_new)
            os.makedirs(os.path.dirname(path_new), exist_ok=True)
            if os.path.exists(path_new):
                os.remove(path_new)
            shutil.move(img, path_new)
    
    shutil.rmtree(dir_temp_new, ignore_errors=True)

    if insert:
        for path_img, bbs in d_old.items():
            path_img_abs = os.path.join(dir_in, path_img)
            name_img = os.path.splitext(os.path.basename(path_img))[0]
            path_new = os.path.join(dir_out, lookup_new[name_img])
            img_src = cv2.imread(path_img_abs)
            img = cv2.imread(path_new)

            for b in bbs:
                x0 = int(b[0])
                y0 = int(b[1])
                x1 = int(b[2])
                y1 = int(b[3])
                img[y0:y1,x0:x1] = img_src[y0:y1,x0:x1]
            
            cv2.imwrite(path_new, img)

def aug_bbg(path, path_ann, dir_out):
    copy_annotations(path_ann, dir_out, "_bbg")
    path_script = os.path.join(path, "augment/bbg/infer.py")
    subprocess.Popen(["python", path_script, "FeedForwardGAN", path_ann, dir_out], shell=True)

def aug_aaug(path, path_ann, dir_out):
    dir_in = os.path.dirname(path_ann)
    d_old, d_new = copy_annotations(path_ann, dir_out, "_aaug")

    for (p_old, bbs), p_new in zip(d_old.items(), d_new.keys()):
        p_old_abs = os.path.join(dir_in, p_old)
        img_old = Image.open(p_old_abs)
        img_new = auto_augment(img_old, bbs, check=False)
        p_new_abs = os.path.join(dir_out, p_new)
        os.makedirs(os.path.dirname(p_new_abs), exist_ok=True)
        img_new.save(p_new_abs, img_old.format)

def copy_annotations(path_ann, dir_out, suffix):
    d_new = {}
    d_old = load_data(path_ann)

    for k, v in d_old.items():
        l = os.path.normpath(k).split(os.sep)
        l[0] += suffix
        d_new[os.path.join(*l)] = v
    
    parts = os.path.splitext(os.path.basename(path_ann))
    new_path_ann = os.path.join(dir_out, parts[0] + suffix + parts[1])
    save_data(new_path_ann, d_new)

    return d_old, d_new

def main():
    path = os.getcwd()
    method = args.method
    path_ann = os.path.join(path, "data_in", args.path_ann)
    dir_out = os.path.join(path, "data_out")
    augment(method, path, path_ann, dir_out)

if __name__ == "__main__":
    main()