from microcredit.geom.geom_methods import SkimageRegistration, Pystackreg, Shift , ImageRegistration
import numpy as np
import copy
import cv2


class RegistrationModule:
    def __init__(self, image: np.array, increase: int):
        self._image = image
        self._increase = increase

    def registration(self):
        images = copy.copy(self._image.images)
        skimage_registration = SkimageRegistration.registration(images)
        pystackreg_registration = Pystackreg.registration(images)
        shift_registration = Shift.registration(images)
        image_registration = ImageRegistration.registration(images)
        save_path = f'dataset/synthetics/align frames'
        print("Дли7нна: ",len(images))
        for next_image in range(len(images)):

            cv2.imwrite(f'{save_path}/skimage_registration_frame_{next_image}.png',
                        skimage_registration[next_image])

            cv2.imwrite(f'{save_path}/shift/RGI/{self._increase}/registration{next_image}.png',
                        shift_registration[next_image])
            cv2.imwrite(f'{save_path}/pystackreg/RGI/{self._increase}/registration{next_image}.png',
                         pystackreg_registration[next_image])
            cv2.imwrite(f'{save_path}/skimage/RGI/{self._increase}/registration{next_image}.png',
                        skimage_registration[next_image])
            cv2.imwrite(f'{save_path}/image_registration/RGI/{self._increase}/registration{next_image}.png',
                        image_registration[next_image])

