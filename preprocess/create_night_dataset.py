"""Script for creating a night dataset given a directory with night and day images.
This script reads the images from a directory and for each of them, calculates the average RGB intensity to determine whether it is
a night image or not. If it is a night image, it is copied to an output directory.

The script does not take all pixels into account by default, instead it gather pixel samples in dotted grid
with a user-defined sample rate. Higher value on sample_rate requires less computation time but may yield less accurate result.
"""
import cv2
import os
import argparse

parser = argparse.ArgumentParser()

# Argument list
parser.add_argument('--input', type=str, help='Path to input dataset.')
parser.add_argument('--output', type=str, help='Path to output directory.')
parser.add_argument('--break_point', type=int, default=60, help='Upper bound of average RGB value defining night.')
parser.add_argument('--sample_rate', type=int, default=10, help='How many pixels to step for each sample pixel.')
parser.add_argument('--file_type', type=str, default='.jpg', help='File type of the image dataset.')
args = parser.parse_args()

input_path = args.input
output_path = args.output
file_type = '.jpg'
break_point = args.break_point
sample_rate = args.sample_rate

directory = os.fsencode(input_path)
for file in os.listdir(directory):
    
    filename = os.fsdecode(file)

    if filename.endswith(file_type): 

        img = cv2.imread(input_path + filename, 1)
        pixelvalue = 0
        
        for widthpixel in [x * sample_rate for x in range(int(len(img)/sample_rate)) if x * sample_rate < len(img)]:
            for heightpixel in [y * sample_rate for y in range(int(len(img[0])/sample_rate)) if y * sample_rate < len(img[0])]:
                for colour in range(len(img[0][0])):
                    pixelvalue = pixelvalue + img[widthpixel][heightpixel][colour]

        mean = pixelvalue / (len(img) * len(img[0]) * len(img[0][0]) / sample_rate ** 2)
        if mean < break_point:
            cv2.imwrite(output_path + filename, img)
        continue
    else:
        continue