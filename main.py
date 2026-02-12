import cv2 
import numpy as np
from pathlib import Path
from tools import *
from microcredit.geom.geom_model import RegistrationModule
from microcredit.read_data import ReadData

def downscale(img, scale=0.5, method=cv2.INTER_AREA):
    h, w = img.shape[:2]
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=method)

def warp_affine_keep_size(img, angle_deg, tx, ty, scale=1.0, border_value=255):
    """
    Поворот вокруг центра + масштаб + сдвиг.
    img: uint8 (H,W) или (H,W,3)
    tx, ty: сдвиги в пикселях
    border_value: фон (для серого 255 = белый)
    """
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    warped = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return warped

def generate_synthetics(path_base_frame, count_frames, path_series):
    img_base = cv2.imread(str(path_base_frame), cv2.IMREAD_GRAYSCALE)
    downscale_base = downscale(img_base,scale=0.3)
    cv2.imwrite(f"{path_series}frame_0.png", downscale_base)

    for i in range(count_frames):
        wrap_img = warp_affine_keep_size(downscale_base,
                                         angle_deg=np.random.uniform(-3, 4), 
                                         tx=np.random.uniform(-3, 4), 
                                         ty=np.random.uniform(-3, 4))
        cv2.imwrite(f"{path_series}frame_{i+1}.png", wrap_img)


def synthetics_main():
    path_base_frame = "dataset/synthetics/base.png"
    path_series = "dataset/synthetics/series frames/"

    generate_synthetics(path_base_frame,count_frames=30, path_series=path_series)
    #cv2.imwrite(f"{path_align}/align_frame_0.png", images[0])
    #for i in range(1, len(images)):
    #    M, warped, warped_err, dbg = align_images(images[0], images[i])
    #    cv2.imwrite(f"{path_align}/align_frame_{i}.png", warped)


def registration_main():
    path_series = "dataset/synthetics/series frames/"
    data = ReadData(path_series)             # ✅ data.images -> list[np.ndarray]

    geom_model = RegistrationModule(image=data, increase=30)
    geom_model.registration()
    
    



if(__name__ =="__main__"):
    #synthetics_main()
    registration_main()