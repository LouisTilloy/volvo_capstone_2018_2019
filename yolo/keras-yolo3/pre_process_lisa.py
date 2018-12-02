import pandas as pd
import os
import copy

ROOT_LISA = "LISA_TS"


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


def get_clean_df(original_df, classes):
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
    df = df.drop(columns=["Occluded,On another road", "Origin file",
                          "Origin frame number", "Origin track",
                          "Origin track frame number"])

    return df


def create_training_file(clean_df):
    """
    From the clean dataframe, create a text file that can be used as an input
    for training yolo.
    """
    with open("train_lisa.txt", "w") as train_file:
        for index in range(len(clean_df)):
            example = clean_df.iloc[index]
            train_file.write(os.path.join(ROOT_LISA, example["Filename"]))
            train_file.write(" ")

            for name in ["x_min", "y_min", "x_max", "y_max"]:
                train_file.write(str(example[name]))
                train_file.write(",")

            train_file.write(str(example["class_id"]))
            train_file.write("\n")


def main():
    annotations = pd.read_csv(os.path.join(ROOT_LISA, "allAnnotations.csv"), delimiter=";")

    classes = create_classes_file(annotations)
    df = get_clean_df(annotations, classes)
    create_training_file(df)


if __name__ == "__main__":
    main()