import os
import copy
import numpy as np
import pandas as pd

ROOT_LISA = "LISA_TS"
ROOT_LISA_EXTENSION = "LISA_TS_extension"


def create_classes_file(original_df):
    """
    Take the original annotation dataframes and create a text file containing
    all the classes from it.
    Also returns the list of classes in the same order as in the text file.
    """
    classes = original_df["Annotation tag"].unique()
    with open("model_data/lisa_classes.txt", "w") as classes_file:
        for index, class_ in enumerate(classes):
            classes_file.write(class_)
            if index < len(classes) - 1:
                classes_file.write("\n")

    return classes


def get_clean_df(original_df, classes, extension=False):
    """
    Given the default annotation dataframe and the classes in the
    same order of the classes text file, returns the dataframe
    with only relevant information for yolo.
    """
    df = copy.deepcopy(original_df)
    # Replace classes with integers
    dict_map = dict([(class_, index) for index, class_ in enumerate(classes)])
    df["Annotation tag"] = df["Annotation tag"].map(dict_map)

    # Replace column names with standard yolo features names
    df = df.rename(index=str, columns={'Upper left corner X': 'x_min',
                                       'Upper left corner Y': 'y_min',
                                       'Lower right corner X': 'x_max',
                                       'Lower right corner Y': 'y_max',
                                       'Annotation tag': 'class_id'})

    # Drop unused columns
    columns = ["Origin file",
               "Origin frame number", "Origin track",
               "Origin track frame number"]
    if not extension:
        columns.append("Occluded,On another road")
    df = df.drop(columns=columns)

    # To keep track from where the data comes from
    df["is_extension"] = extension

    return df


def create_training_file(clean_df, train_file_name, extension=False):
    """
    From the clean dataframe, create a text file that can be used as an input
    for training yolo.
    """
    with open(train_file_name, "w") as train_file:
        for index in range(len(clean_df)):
            example = clean_df.iloc[index]
            # Get image folder root
            if example["is_extension"]:
                root_folder = ROOT_LISA_EXTENSION
            else:
                root_folder = ROOT_LISA
                
            train_file.write(os.path.join(root_folder, example["Filename"]))
            train_file.write(" ")

            for name in ["x_min", "y_min", "x_max", "y_max"]:
                train_file.write(str(example[name]))
                train_file.write(",")

            train_file.write(str(example["class_id"]))
            train_file.write("\n")


def split_df(df, test_size=0.2):
    """
    Split a dataframe into a train dataframe and a test dataframe.
    """
    np.random.seed(10)
    indices = np.arange(0, len(df))
    np.random.shuffle(indices)

    train_df = df.iloc[indices[int(len(indices)*test_size):]]
    test_df = df.iloc[indices[:int(len(indices)*test_size)]]

    return train_df, test_df



def main():
    # Pre-process original LISA dataset
    print("Pre-processing original LISA dataset...")
    annotations = pd.read_csv(os.path.join(ROOT_LISA, "allAnnotations.csv"), delimiter=";")
    annotations_ext = pd.read_csv(os.path.join(ROOT_LISA_EXTENSION, "allTrainingAnnotations.csv"), delimiter=";")

    classes = create_classes_file(pd.concat([annotations, annotations_ext]))
    df_original = get_clean_df(annotations, classes)
    print("Done.")
    print()

    # Pre-process extension LISA dataset
    print("Pre-processing extension LISA dataset...")

    df_ext = get_clean_df(annotations_ext, classes, extension=True)
    print("Done.")
    print()

    # Creating txt files
    print("Creating train and test txt files...")
    df = pd.concat([df_original, df_ext])
    train_df, test_df = split_df(df, test_size=0.2)

    create_training_file(train_df, "train_lisa.txt")
    create_training_file(test_df, "test_lisa.txt")
    print("Done.")


if __name__ == "__main__":
    main()