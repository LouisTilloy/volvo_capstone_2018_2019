import pandas as pd
import os
import copy

ROOT_LISA = "LISA_TS"
ROOT_LISA_EXTENSTION = "LISA_TS_extension"


def create_classes_file(original_df):
    """
    Take the original annotation dataframes and create a text file containing
    all the classes from it.
    Also returns the list of classes in the same order as in the text file.
    """
    classes = original_df["Annotation tag"].unique()
    with open("lisa_classes.txt", "w") as classes_file:
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

    return df


def create_training_file(clean_df, train_file_name, extension=False):
    """
    From the clean dataframe, create a text file that can be used as an input
    for training yolo.
    """
    root_folder = ROOT_LISA if not extension else ROOT_LISA_EXTENSTION
    with open(train_file_name, "w") as train_file:
        for index in range(len(clean_df)):
            example = clean_df.iloc[index]
            train_file.write(os.path.join(root_folder, example["Filename"]))
            train_file.write(" ")

            for name in ["x_min", "y_min", "x_max", "y_max"]:
                train_file.write(str(example[name]))
                train_file.write(",")

            train_file.write(str(example["class_id"]))
            train_file.write("\n")


def main():
    # Pre-process original LISA dataset
    print("Pre-processing original LISA dataset...")
    annotations = pd.read_csv(os.path.join(ROOT_LISA, "allAnnotations.csv"), delimiter=";")

    classes = create_classes_file(annotations)
    df = get_clean_df(annotations, classes)
    create_training_file(df, "train_lisa.txt")
    print("Done.")
    print()

    # Pre-process extension LISA dataset
    print("Pre-processing extension LISA dataset...")
    annotations = pd.read_csv(os.path.join(ROOT_LISA_EXTENSTION, "allTrainingAnnotations.csv"), delimiter=";")

    df = get_clean_df(annotations, classes, extension=True)
    create_training_file(df, "test_lisa.txt", extension=True)
    print("Done.")


if __name__ == "__main__":
    main()