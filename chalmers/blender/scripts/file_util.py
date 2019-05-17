import os
import csv

def mkdir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("Directory ", dir_path, " Created ")
    else:    
        print("Directory ", dir_path, " already exists")
        
def mkcsv(file_path, dict_list):
    # https://realpython.com/python-csv/
    # https://stackoverflow.com/questions/5824530/python-struct-arrays
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for d in dict_list:
            writer.writerow(d)