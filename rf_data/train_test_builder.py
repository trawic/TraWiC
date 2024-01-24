import os

import pandas as pd
import tqdm

# only get the folders
# all_directories=[os.path.join(os.getcwd(),'rf_data',path)for path in os.listdir(os.path.join(os.getcwd(),'rf_data')) if os.path.isdir(os.path.join(os.getcwd(),'rf_data',path))]
syn = 100
sem = 20
senstivie = False
sen_thresh = 0.1
if not senstivie:
    all_directories = [os.path.join(os.getcwd(), "rf_data", f"syn{syn}_sem{sem}")]
else:
    all_directories = [
        os.path.join(
            os.getcwd(), "rf_data", f"syn{syn}_sem{sem}_sensitive_{sen_thresh}"
        )
    ]

# Read all the csv files in the directory, and concatenate them into a single dataframe
for directory in tqdm.tqdm(all_directories):
    df = pd.concat(
        [
            pd.read_csv(os.path.join(directory, file))
            for file in os.listdir(directory)
            if file.endswith(".csv")
        ]
    )
    # shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)
    # drop useless columns
    df.drop(
        columns=[
            "class_nums_total",
            "function_nums_total",
            "variable_nums_total",
            "string_nums_total",
            "comment_nums_total",
            "docstring_nums_total",
        ],
        inplace=True,
    )
    # split into train and test
    train_df = df.iloc[: int(0.8 * len(df))]
    test_df = df.iloc[int(0.8 * len(df)) :]
    # save the concatenated dataframe as a csv file
    if not senstivie:
        train_df.to_csv(
            os.path.join(os.getcwd(), "rf_data", f"syn{syn}_sem{sem}", "train.csv"),
            index=False,
        )
        test_df.to_csv(
            os.path.join(os.getcwd(), "rf_data", f"syn{syn}_sem{sem}", "test.csv"),
            index=False,
        )
    else:
        train_df.to_csv(
            os.path.join(
                os.getcwd(),
                "rf_data",
                f"syn{syn}_sem{sem}_sensitive_{sen_thresh}",
                "train.csv",
            ),
            index=False,
        )
        test_df.to_csv(
            os.path.join(
                os.getcwd(),
                "rf_data",
                f"syn{syn}_sem{sem}_sensitive_{sen_thresh}",
                "test.csv",
            ),
            index=False,
        )
