import cv2 
import numpy as np
from pathlib import Path
from image_registration import *
from tools import *
from read_data import ReadData
from synthetic import generate_synthetics
import matplotlib.pyplot as plt



def synthetics_main():
    path_base_frame = "dataset/synthetics/base.png"
    path_series = "dataset/synthetics/series frames/"

    generate_synthetics(path_base_frame,count_frames=15, path_series=path_series)



def registration_main(path_dataset = "dataset/synthetics", save_metric=False):
    data = ReadData(f"{path_dataset}/series frames")

    ref = data.images[0]
    cv2.imwrite(f"{path_dataset}/align frames/frame_0.png", ref)

    mse_before_list = []
    mse_after_list = []

    ncc_before_list = []
    ncc_after_list = []

    for i in range(1, len(data.images)):

        mov = data.images[i]

        # --- метрики ДО ---
        before = compute_similarity(ref, mov)

        # --- ECC ---
        aligned, _, _ = register_images(ref, mov, method="ecc")

        # --- метрики ПОСЛЕ ---
        after = compute_similarity(ref, aligned)

        # сохраняем значения
        mse_before_list.append(before['mse'])
        mse_after_list.append(after['mse'])

        ncc_before_list.append(before['ncc'])
        ncc_after_list.append(after['ncc'])

        # сохраняем выровненный кадр
        cv2.imwrite(f"{path_dataset}/align frames/frame_{i}.png", aligned)

    # сохранить графики
    if(save_metric):
        save_metric_plots(
            mse_before_list, mse_after_list,
            ncc_before_list, ncc_after_list,
            path_dataset
        )
    


if(__name__ =="__main__"):
    synthetics_main()
    registration_main()