from skimage import registration
from pystackreg import StackReg
from image_registration import cross_correlation_shifts
import numpy as np
import cv2
import itertools



class RegistrationBase:
    @staticmethod
    def registration(images: np.ndarray):
        pass


class Shift(RegistrationBase):
    @staticmethod
    def registration(images: np.ndarray):
        shift = np.array([0, 131, -131])
        h, w = images[0].shape[:2]
        next_img = 0
        result = []
        forward_list = list(itertools.combinations_with_replacement(shift, 2))
        reversed_list = [tuple(reversed(item)) for item in forward_list]
        combinations = sorted(list(set(forward_list + reversed_list)))
        for i in reversed(combinations):
            translation_matrix = np.float32([[1, 0, i[0]], [0, 1, i[1]]])
            dst = cv2.warpAffine(images[next_img], translation_matrix, (w, h))
            next_img += 1
            result.append(dst)
        return result


class SkimageRegistration(RegistrationBase):
    @staticmethod
    def registration(images: np.ndarray):
        corrected_image = []
        for next_image in range(len(images)):
            im_shift, error, diffphase = registration.phase_cross_correlation(images[4], images[next_image])
            M = np.float32([[1, 0, im_shift[1]], [0, 1, im_shift[0]]])
            height, width = images[next_image].shape[:2]
            shifted_image = cv2.warpAffine(images[next_image], M, (width, height))
            corrected_image.append(shifted_image)
        return corrected_image


class Pystackreg(RegistrationBase):
    @staticmethod
    def registration(images: np.ndarray):
        image_array = np.stack(images, axis=0)
        sr = StackReg(StackReg.TRANSLATION)
        registered_images = sr.register_transform_stack(image_array, reference='previous')
        return registered_images


class ImageRegistration(RegistrationBase):
    @staticmethod
    def registration(images: np.ndarray):
        corrected_image = []
        for next_image in range(len(images)):
            yoff, xoff = cross_correlation_shifts(images[4], images[next_image])
            M = np.float32([[1, 0, yoff], [0, 1, xoff]])
            height, width = images[next_image].shape[:2]
            shifted_image = cv2.warpAffine(images[next_image], M, (width, height))
            corrected_image.append(shifted_image)
        return corrected_image