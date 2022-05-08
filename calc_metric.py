import cv2
import os
from math import ceil
import time
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import importlib
import random
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import os
import sys
import pandas as pd
import multiprocessing as mp

MAX = 100000000000000


class VideoReader:
    def __init__(self, path):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.height = int(self.cap.get(4))
        self.width = int(self.cap.get(3))
        self.it = iter(self)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __iter__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret == False:
                break
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame

    def __call__(self):
        return next(self.it)

    def __del__(self):
        self.cap.release()

    def refresh(self):
        self.cap = cv2.VideoCapture(self.path)


class MetricArray:
    def __init__(self, vid_path, mask_path, n_height, n_width):
        self.vid = None if vid_path is None else VideoReader(vid_path)
        self.n_height = n_height
        self.n_width = n_width
        self.block_height = self.vid.height // n_height
        self.block_width = self.vid.width // n_width
        self.scores = np.zeros((self.vid.length, self.n_height, self.n_width))
        self.mask_raw = None if mask_path is None else VideoReader(mask_path)
        self.mask = np.zeros_like(self.scores)
        self.length = self.vid.length
        from koniq_api.lib import MyModel
        self.model = MyModel()

    def get_subframe(self, frame, i, j) -> np.array:
        return frame[
            i * self.block_height : min(self.vid.height, (i + 1) * self.block_height),
            j * self.block_width : min(self.vid.width, (j + 1) * self.block_width),
        ]

    def split_video(self):
        """
        devide video on n * m subvideos and save each of them in ./tmp/ folder
        :return:
        """

        TMP_DIR = "./tmp/"
        if not os.path.exists(TMP_DIR):
            os.mkdir(TMP_DIR)
        for f in os.listdir(TMP_DIR):
            os.remove(TMP_DIR + f)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        for i in range(self.n_height):
            for j in range(self.n_width):
                out = cv2.VideoWriter(
                    f"{TMP_DIR}{i}_{j}.mp4",
                    fourcc,
                    self.vid.fps,
                    (self.block_width, self.block_height),
                )
                for frame in self.vid:
                    subframe = self.get_subframe(frame, i, j)
                    out.write(subframe)
                out.release()
                self.vid.refresh()

    def calc_raw_metric(self, full_frame: bool):
        for frame_idx, frame in tqdm(enumerate(self.vid), total=self.length):
            if frame_idx > MAX:
                return
            for i in range(self.n_height):
                for j in range(self.n_width):
                    if full_frame:
                        self.scores[frame_idx][i][j] = self.model(frame)
                    else:
                        self.scores[frame_idx][i][j] = self.model(
                            self.get_subframe(frame, i, j)
                        )

    def resize_mask(self):
        self.mask_raw.refresh()
        for fi, frame in enumerate(self.mask_raw):
            for i in range(self.mask.shape[1]):
                for j in range(self.mask.shape[2]):
                    self.mask[fi][i][j] = (
                        frame[
                            i * self.block_height : (i + 1) * (self.block_height),
                            j * self.block_width : (j + 1) * (self.block_width),
                        ].mean()
                        / 255
                    )
        self.mask_raw.refresh()

    def apply_mask(self):
        return self.scores * self.mask

    def calc_metric(self):
        return self.apply_mask().mean()


def parse(name: str) -> dict:
    out = dict()
    name = name.split(".")[0]
    name = name.split("enc_res_")[1]

    str1 = "_mv_offline_2k_v1_"
    str2 = "_mv_offline_2k_"

    if name.find(str1) >= 0:
        method = "mv_offline_2k_v1"
        name = name.split(str1)
    else:
        method = "mv_offline_2k"
        name = name.split(str2)

    out["method"] = method
    out["codec"] = name[0]

    name = name[1]
    name = name.split("_")
    out["crf"] = int(name[-1])

    name = "_".join(name[0:-1])
    out["name"] = name
    return out


def get_1_score(df: pd.DataFrame, name, codec, crf, method):
    x = df[
        (df["comparison"] == "ugc")
        & (df["sequence"] == name)
        & (df["crf"] == int(crf))
        & (df["codec"] == codec)
        & (df["preset"] == method)
    ]
    x = x["subjective"]

    if len(x) == 1:
        x = float(x)
    else:
        return None

    return x


def get_subj_scores(videos, file="subjective_scores.csv"):
    subj = pd.read_csv(file, sep=";")
    return [get_1_score(subj, **parse(name)) for name in videos]


def folder2name(folder):
    name = os.path.basename(os.path.normpath(folder))
    name = "_".join(name.split("_")[:-1])
    return name


# ==================================================================================


def test_single_video(IDLE, video, sequence, div_deg, suffics, out_folder, _MAX=MAX):
    global MAX
    MAX = _MAX

    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    array_file = out_folder + "/" + sequence + "_" + suffics + str(div_deg) + ".npy"

    print(video, sequence, array_file)
    print()

    if not IDLE:
        m = MetricArray(video, None, div_deg, div_deg)
        m.vid = VideoReader(video)
        m.calc_raw_metric(full_frame=True if div_deg is 1 else False)

        with open(array_file, "ab") as f:
            np.save(f, m.scores)
    else:
        pass

