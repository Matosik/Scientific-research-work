import cv2, numpy as np
from geom_align import *
import cv2, numpy as np
from tools import *

a = "test_img/picture/1.jpg"
b = "test_img/picture/2.jpg"

img_a = cv2.imread(a, cv2.IMREAD_COLOR)
img_b = cv2.imread(b, cv2.IMREAD_COLOR)


img_a = to_gray(img_a)
img_b = to_gray(img_b)

err_a = load_error_map_for_image(a)  
err_b = load_error_map_for_image(b)



M, warped, warped_err, info = align_images(img_a, img_b,
                                            err_src=err_a,
                                            err_dst=err_b,
                                            prefer_sift=True,
                                            model='homography',   #  'affine'
                                            debug_dir="debug_out",
                                            ratio_thresh=0.75,
                                            ransac_thresh=3.0)

# сохранить результаты
cv2.imwrite(f"{a.split('/')[1]}_{ a.split('/')[2].split('.')[0] }_warped_on_{  b.split('/')[2].split('.')[0]  }.jpg", warped)
if warped_err is not None:
    np.save(f"{a.split('/')[1]}_{  a.split('/')[2].split('.')[0]  }_warped_on_{  b.split('/')[2].split('.')[0]  }_err.npy", warped_err)
print(info)