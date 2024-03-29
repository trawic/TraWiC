import json
import os
import tokenize
from io import BytesIO
from typing import List, Tuple

import pandas as pd
import tqdm
from fuzzywuzzy import fuzz
from skip_data import SKIPS


def extract_comments_and_docstrings(script: str) -> Tuple[List, List]:
    """
    Extracts comments and docstrings from a given script.

    Args:
        script (str): The script to extract comments and docstrings from.

    Returns:
        Tuple(List, List): A tuple containing a list of comments and a list of docstrings.
    """
    comments = []
    docstrings = []
    tokens = tokenize.tokenize(BytesIO(script.encode("utf-8")).readline)

    for token in tokens:
        if token.type == tokenize.COMMENT:
            comments.append(token.string.strip())
        elif token.type == tokenize.STRING and token.string.startswith(('"""', "'''")):
            docs = token.string.strip().split("\n")
            # remove the """ or ''' from the first and last lines
            docs = docs[1:-1]
            # append every element of docs to docstrings
            for element in docs:
                docstrings.append(element.strip())

    return comments, docstrings


def comment_to_code_ratio(script_path: str) -> float:
    """
    Calculates the ratio of comments and docstrings to code in a given script.

    Args:
        script_path (str): Path to the script to calculate the ratio for.

    Returns:
        float: ratio of comments and docstrings to code in the script.
    """
    try:
        script = open(script_path, "r").read()

        comments, docstrings = extract_comments_and_docstrings(script)

        comment_lines = len(comments)
        number_of_comment_chars= sum([len(comment) for comment in comments])
        
        docstring_lines = len(docstrings)
        number_of_docstring_chars = sum([len(docstring) for docstring in docstrings])
        
        code_lines = len(script.split("\n"))
        number_of_code_chars = len(script) - number_of_comment_chars - number_of_docstring_chars
        

        return (number_of_comment_chars + number_of_docstring_chars) / number_of_code_chars
    except Exception as e:
        return 2


def filter_syn_errors(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Read the dataset, first filter out the rows that don't have an 'exact' similarity_metric, Then filter those that don't have a 'result' of 1

    Args:
        dataset (pd.DataFrame): The dataset to filter
    """
    syn_errors = dataset[dataset["similarity_metric"] == "exact"]
    syn_errors = syn_errors[syn_errors["result"] != 1]

    syn_errors_in_thestack = syn_errors[syn_errors["trained_on"] == 1]
    syn_errors_not_in_thestack = syn_errors[syn_errors["trained_on"] == 0]

    return syn_errors, syn_errors_in_thestack, syn_errors_not_in_thestack


def filter_sem_errors(dataset: pd.DataFrame, threshold: int = 60) -> pd.DataFrame:
    """
    Read the dataset, first filter out the rows that don't have an 'exact' similarity_metric, Then filter those that don't have a 'result' of 1

    Args:
        dataset (pd.DataFrame): The dataset to filter
    """
    sem_errors = dataset[dataset["similarity_metric"] != "exact"]
    sem_errors = sem_errors[sem_errors["result"] < threshold]

    sem_errors_in_thestack = sem_errors[sem_errors["trained_on"] == 1]
    sem_errors_not_in_thestack = sem_errors[sem_errors["trained_on"] == 0]

    return sem_errors, sem_errors_in_thestack, sem_errors_not_in_thestack


def calc_len_sync_mistakes(dataset: pd.DataFrame):
    "calcuate the length of what the model is supposed to predict and what it actually predicts"
    lens = []
    for row in dataset.iterrows():
        try:
            lens.append(len(str(row[1]["model_output"]).split()))
        except ZeroDivisionError:
            pass
    print(
        "Average length of tokens generated by the model for syntactic mistakes: ",
        sum(lens) / len(lens),
    )


def calc_if_token_in_prediction(dataset: pd.DataFrame, threshold: float = 0.1):
    "calcuate if the token exists in the prediction"
    res = []
    all_tokens = {}
    for row in dataset.iterrows():
        "parse each prediction and split it inoto tokens. if the token is in all_tokens, add 1 to it, else add it to all_tokens"
        for token in str(row[1]["similarity_objective"]).split():
            if token in all_tokens:
                all_tokens[token] += 1
            else:
                all_tokens[token] = 1
    # normalize the values in all_tokens
    max_value = max(all_tokens.values())
    all_tokens = {k: v / max_value for k, v in all_tokens.items()}

    # keep only the tokens that have a value of 0.3 or less
    all_tokens = {k: v for k, v in all_tokens.items() if v <= threshold}
    token_types = {"variable_name": 0, "function_name": 0, "class_name": 0}
    all_token_types = {"variable_name": 0, "function_name": 0, "class_name": 0}
    for row in dataset.iterrows():
        similarity_objective = str(row[1]["similarity_objective"])
        model_output = str(row[1]["model_output"]).split()
        if similarity_objective in model_output and similarity_objective in all_tokens:
            token_types["variable_name"] += 1 if row[0] == "variable_names" else 0
            token_types["function_name"] += 1 if row[0] == "function_names" else 0
            token_types["class_name"] += 1 if row[0] == "class_names" else 0
        elif (
            similarity_objective not in model_output
            and similarity_objective in all_tokens
        ):
            all_token_types["variable_name"] += 1 if row[0] == "variable_names" else 0
            all_token_types["function_name"] += 1 if row[0] == "function_names" else 0
            all_token_types["class_name"] += 1 if row[0] == "class_names" else 0
        else:
            continue
    print(
        "Average number of times the token actually exists in the prediction for syntactic mistakes - vairable names: ",
        token_types["variable_name"]
        / (token_types["variable_name"] + all_token_types["variable_name"]),
    )
    print(
        "Average number of times the token actually exists in the prediction for syntactic mistakes - function names: ",
        token_types["function_name"]
        / (token_types["function_name"] + all_token_types["function_name"]),
    )
    print(
        "Average number of times the token actually exists in the prediction for syntactic mistakes - class names: ",
        token_types["class_name"]
        / (token_types["class_name"] + all_token_types["class_name"]),
    )


if __name__ == "__main__":
    # get the full path of csvs in the run_results folder that is in a folder starting with TokensRun
    # ds_paths = [os.path.join(root, name)
    #             for root, dirs, files in os.walk("/store/travail/vamaj/TWMC/run_results")
    #             for name in files
    #             if name.endswith((".csv")) and root.split('/')[-1].startswith('TokensRun')]
    # syn_erros = []
    # sem_errors = []

    # syn_errors_in_thestack = []
    # sem_errors_in_thestack = []

    # syn_errors_not_in_thestack = []
    # sem_errors_not_in_thestack = []

    # for path in tqdm.tqdm(ds_paths):
    #     ds = pd.read_csv(path, index_col=0)
    #     for row in ds.iterrows():
    #         path_to_file = os.path.join(os.getcwd(), "data", row[0])
    #         comment_to_code_ratio_file = comment_to_code_ratio(path_to_file)
    #         if comment_to_code_ratio_file == 2:
    #             ds.loc[row[0], "trained_on"] = 2
    #         elif 0.01 < comment_to_code_ratio_file < 0.8:
    #             ds.loc[row[0], "trained_on"] = 1
    #         else:
    #             ds.loc[row[0], "trained_on"] = 0
    #     # drop the rows that their trained_on value is 2
    #     ds = ds[ds["trained_on"] != 2]

    #     syn, syn_in, syn_not_in = filter_syn_errors(ds)
    #     sem, sem_in, sem_not_in = filter_sem_errors(ds)

    #     syn_erros.append(syn)
    #     sem_errors.append(sem)

    #     syn_errors_in_thestack.append(syn_in)
    #     sem_errors_in_thestack.append(sem_in)

    #     syn_errors_not_in_thestack.append(syn_not_in)
    #     sem_errors_not_in_thestack.append(sem_not_in)

    # syn_errors = pd.concat(syn_erros)
    # sem_errors = pd.concat(sem_errors)

    # syn_errors_in_thestack = pd.concat(syn_errors_in_thestack)
    # sem_errors_in_thestack = pd.concat(sem_errors_in_thestack)

    # syn_errors_not_in_thestack = pd.concat(syn_errors_not_in_thestack)
    # sem_errors_not_in_thestack = pd.concat(sem_errors_not_in_thestack)

    # syn_errors.to_csv('syn_errors.csv', index=False)
    # sem_errors.to_csv('sem_errors.csv', index=False)

    # syn_errors_in_thestack.to_csv('syn_errors_in_thestack.csv', index=False)
    # sem_errors_in_thestack.to_csv('sem_errors_in_thestack.csv', index=False)
    # syn_errors_not_in_thestack.to_csv('syn_errors_not_in_thestack.csv', index=False)
    # sem_errors_not_in_thestack.to_csv('sem_errors_not_in_thestack.csv', index=False)
    calc_if_token_in_prediction(
        pd.read_csv(
            "/store/travail/vamaj/TWMC/run_results/syn_errors_in_thestack.csv",
            index_col=0,
        ),
        threshold=0.3,
    )
    calc_if_token_in_prediction(
        pd.read_csv(
            "/store/travail/vamaj/TWMC/run_results/syn_errors_not_in_thestack.csv",
            index_col=0,
        ),
        threshold=0.05,
    )
