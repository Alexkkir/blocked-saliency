import os, sys, multiprocessing as mp
from calc_metric import test_single_video, folder2name

DATASET = 'ugc'
folders = sorted(os.listdir(DATASET))
folders = list(map(lambda folder: os.path.join(DATASET, folder), folders))
for folder in folders[3:4]:
    videos = sorted(os.listdir(folder))
    videos = list(map(lambda video: os.path.join(folder, video), videos))
    total_videos = len(videos)
    for i, video in enumerate(videos):
        foldername = folder2name(folder)
        print()
        print(f"Video #{i + 1}/{total_videos}")
        print(folder, '\t', video)
        # p = mp.Process(target=test_single_video, args=(video, foldername, 8, 'div', 'np_arrays'))
        # p.start()
        # p.join()