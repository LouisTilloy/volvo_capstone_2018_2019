import unittest
from unittest import mock
import os
from copy import deepcopy
import pandas as pd
import numpy as np

from pre_process_lisa import get_clean_df, create_file,\
    ROOT_LISA_EXTENSION, ROOT_LISA


class TestPreProcess(unittest.TestCase):

    def setUp(self):
        self.temp_file = "temp_train_file.txt"
        self.classes = ["cat", "dog", "horse"]

        # original df
        self.original_df = pd.DataFrame()
        self.original_df["Filename"] = ["1.txt", "1.txt", "2.txt", "3.txt", "4.txt"]
        self.original_df["Annotation tag"] = ["cat", "cat", "dog", "cat", "horse"]
        self.original_df['Upper left corner X'] = [1, 2, 3, 4, 4]
        self.original_df['Upper left corner Y'] = [1, 2, 3, 4, 4]
        self.original_df['Lower right corner X'] = [2, 3, 4, 5, 6]
        self.original_df['Lower right corner Y'] = [10, 10, 10, 10, 10]

        unused_cols = ["Origin file",
                       "Origin frame number", "Origin track",
                       "Origin track frame number"]
        for col in unused_cols:
            self.original_df[col] = ["zfgr"] * 5

        # original df with extension
        self.original_df_ext = deepcopy(self.original_df)
        self.original_df_ext["is_extension"] = [True] * 5

        # original df has 1 more column
        self.original_df["Occluded,On another road"] = ["aaa"] * 5

        # expected clean dataframe
        self.expected_clean_df = pd.DataFrame()
        self.expected_clean_df["Filename"] = ["1.txt", "1.txt", "2.txt", "3.txt", "4.txt"]
        self.expected_clean_df["class_id"] = [0, 0, 1, 0, 2]
        self.expected_clean_df["x_min"] = [1, 2, 3, 4, 4]
        self.expected_clean_df["y_min"] = [1, 2, 3, 4, 4]
        self.expected_clean_df["x_max"] = [2, 3, 4, 5, 6]
        self.expected_clean_df["y_max"] = [10, 10, 10, 10, 10]
        self.expected_clean_df["is_extension"] = [False] * 5

    @mock.patch("pre_process_lisa.tqdm")
    def test_get_clean_df_noext(self, mock_tqdm):
        def mock_fn(x):
            return x
        mock_tqdm.side_effects = mock_fn

        clean_original_df = get_clean_df(self.original_df, self.classes, False)
        np.testing.assert_equal(clean_original_df.values, self.expected_clean_df.values)

    @mock.patch("pre_process_lisa.tqdm")
    def test_get_clean_df_ext(self, mock_tqdm):
        def mock_fn(x):
            return x
        mock_tqdm.side_effects = mock_fn

        clean_original_df = get_clean_df(self.original_df_ext, self.classes, True)
        expected_clean_df = deepcopy(self.expected_clean_df)
        expected_clean_df["is_extension"] = [True] * 5

        np.testing.assert_equal(clean_original_df.values, expected_clean_df.values)

    def test_create_file(self):
        filenames = ["1.txt", "2.txt", "3.txt", "4.txt"]
        create_file(self.expected_clean_df, self.temp_file, filenames)

        with open(self.temp_file, "r") as file:
            line = file.readline()
            self.assertEqual(
                "{} 1,1,2,10,0 2,2,3,10,0\n".format(os.path.join(ROOT_LISA, "1.txt")),
                line
            )

            line = file.readline()
            self.assertEqual(
                "{} 3,3,4,10,1\n".format(os.path.join(ROOT_LISA, "2.txt")),
                line
            )

            line = file.readline()
            self.assertEqual(
                "{} 4,4,5,10,0\n".format(os.path.join(ROOT_LISA, "3.txt")),
                line
            )

            line = file.readline()
            self.assertEqual(
                "{} 4,4,6,10,2\n".format(os.path.join(ROOT_LISA, "4.txt")),
                line
            )


    def tearDown(self):
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
