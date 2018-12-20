#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import subprocess, re

from glob import glob


def readcsv(filename):
    d = re.match(
        r"ground_truth/precision_(?P<net>mnist_relu_[0-9]+_[0-9]+)_eps_(?P<eps>[0-9]+\.[0-9]+)\.txt",
        filename
    ).groupdict()

    df = pd.read_csv(filename, sep="  ", engine="python", names=("img", "status", "label"))
    df[["img"]] = df[["img"]].applymap(lambda x: x.replace(" ", ""))
    df[["status"]] = df[["status"]].applymap(lambda x: x.replace(" label", ""))
    df["net"], df["eps"] = d['net'], d['eps']
    df = df[df.status != "not considered"]
    df = df[df.img != "analysisprecision"]
    return df


def process(row, executable):
    net, img, eps = f"../mnist_nets/{row['net']}.txt", f"../mnist_images/{row['img']}.txt", row['eps']
    res = str(subprocess.check_output(f"python3 {executable} {net} {img} {eps}", shell=True, stderr=subprocess.STDOUT))
    if not re.search(f"can not be verified", res) and re.search(f"verified", res):
        print(res)
        return 1 if row['status'] == "verified" else -1        
    elif re.search(f"can not be verified", res):
        return 1 if row['status'] == "failed" else 0
    else:
        raise Exception("Test failed")
        return 0

    
if __name__ == "__main__":
    filenames = glob('ground_truth/*')
    dataset = pd.concat([readcsv(filename) for filename in filenames])
    score = 0
    for i, row in enumerate(dataset.to_dict(orient='records')):
        incr = process(row, "analyzer_new.py")
        assert incr >= 0, f"{row} failed"
        score += incr
        print(f"{score}/{i+1}")

