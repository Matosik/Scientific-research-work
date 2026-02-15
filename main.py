import cv2 
import numpy as np
from pathlib import Path
from image_registration import register_images
from tools import *
from read_data import ReadData
from synthetic import generate_synthetics


def synthetics_main():
    path_base_frame = "dataset/synthetics/base.png"
    path_series = "dataset/synthetics/series frames/"

    generate_synthetics(path_base_frame,count_frames=30, path_series=path_series)


def registration_main(path_dataset = "dataset/synthetics"):
    data = ReadData(f"{path_dataset}/series frames")             # data.images -> list[np.ndarray]
    cv2.imwrite(f"{path_dataset}/align frames/frame_0.png", data.images[0])
    for i in range(1,len(data.images)):
        aliging,_,_ = register_images(data.images[0], data.images[i], method="ecc")
        cv2.imwrite(f"{path_dataset}/align frames/frame_{i}.png", aliging)
    



if(__name__ =="__main__"):
    #synthetics_main()
    registration_main()