import json
import pandas as pd
import argparse
from typing import Tuple
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert csv to json")
    parser.add_argument("--input", type=str, help="input csv file")
    parser.add_argument("--output", type=str, help="output json file")
    parser.add_argument(
        "--image-column",
        type=str,
        help="image column name in input",
        default="filepath",
    )
    parser.add_argument(
        "--text-column", type=str, help="text column name in input", default="title"
    )
    parser.add_argument(
        "--data-root", type=str, help="data root directory", default=None
    )
    args = parser.parse_args()
    return args


def csv_to_json(
    input_csv: str,
    output_json: str,
    image_column: str = "filepath",
    text_column: str = "title",
    data_root: str = None,
) -> None:
    df = pd.read_csv(input_csv, sep="\t").rename(
        {image_column: "image", text_column: "caption"}, axis=1
    )
    if data_root:
        df["image"] = df["image"].apply(lambda x: os.path.join(data_root, x))
    df.to_json(output_json, orient="records")
    return


if __name__ == "__main__":
    args = parse_args()
    csv_to_json(
        args.input, args.output, args.image_column, args.text_column, args.data_root
    )
