"""Script for extracting generated cycleGAN data.
Takes cycleGAN's result directory as input and creates a directory with only the generated images.
The script also removes cycleGAN's name extension '_fake_B'.
"""

from shutil import copyfile
import os
import argparse

parser = argparse.ArgumentParser()

# Argument list
parser.add_argument('--input', type=str, help='Path to cycleGAN result directory.')
parser.add_argument('--output', type=str, help='Path to desired output directory.')
args = parser.parse_args()


filenames = os.listdir(args.input)
for filename in filenames:
    if str(filename[-5]) is 'B':
        new_filename = str(filename).replace('_fake_B', '')
        copyfile(args.input + filename, args.output + new_filename)
