"""Script for cropping images.
Crops random regions in images from a directory which the user specify, then outputs the cropped versions in an output folder.
"""
import os
import cv2 
import random
import argparse

parser = argparse.ArgumentParser()

# Argument list
parser.add_argument('--input', type=str, help='Path to input dataset.')
parser.add_argument('--output', type=str, help='Path to output directory.')
parser.add_argument('--file_type', type=str, default='.jpg', help='File type of the image dataset.')
parser.add_argument('--height', type=int, default=256, help='Output image height.')
parser.add_argument('--width', type=int, default=256, help='Output image width.')
args = parser.parse_args()

directory = os.fsencode(args.input)
for file in os.listdir(directory):

    filename = os.fsdecode(file)

    if filename.endswith(args.file_type):
        img = cv2.imread(args.input + filename, 1)
        x = random.randint(0, img.shape[1] - args.width)
        y = random.randint(0, img.shape[0] - args.height)
        img = img[y:y+args.height, x:x+args.width]
        cv2.imwrite(args.output + filename, img)
