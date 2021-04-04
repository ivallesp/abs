import argparse
import os
from multiprocessing import Pool
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
from glob import glob
from multiprocessing import Pool

SCALAR_TAGS = [
    "train/crossentropy",
    "train/accuracy",
    "test/crossentropy",
    "test/accuracy",
]


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-t",
        "--tensorboard-logs-dir",
        required=True,
        type=str,
        help="",
    )
    argparser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=str,
        help="",
    )
    argparser.add_argument(
        "-j",
        "--nproc",
        required=True,
        type=int,
        help="",
    )
    return argparser.parse_args()


def export_summaries(input_dir, output_dir):
    ea = EventAccumulator(input_dir)
    ea.Reload()
    df = []
    for tag in SCALAR_TAGS:
        _, epoch, value = zip(*ea.Scalars(tag))
        df.append(pd.DataFrame({"epoch": epoch, "metric": tag, "value": value}))
    df = pd.concat(df)
    output_path = os.path.join(output_dir, f"{os.path.split(input_dir)[1]}.csv")
    df.to_csv(output_path, index=False)


def func(x):
    export_summaries(x[0], x[1])


def main():
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    input_dirs = glob(os.path.join(args.tensorboard_logs_dir, "*"))
    print(f"{len(input_dirs)} models found!")

    input_arguments = [(i, args.output_dir) for i in input_dirs]

    with Pool(processes=args.nproc) as pool:
        for x in tqdm(
            pool.imap_unordered(func, input_arguments), total=len(input_arguments)
        ):
            pass


if __name__ == "__main__":
    main()
