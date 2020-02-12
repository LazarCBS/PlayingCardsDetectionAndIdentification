import cv2
import os
import glob
import matplotlib.pyplot as plt
import numpy as np


def to_gray_scale(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray


def preprocess_image(image):
    image = cv2.bilateralFilter(image, d=7, sigmaSpace=75, sigmaColor=75)
    img_gray = to_gray_scale(image)
    a = img_gray.max()
    _, thresh = cv2.threshold(img_gray, a / 2 + 60, a, cv2.THRESH_BINARY)
    return thresh


def get_contours(image):
    binary_image = preprocess_image(image)
    _, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def crop_and_rotate_card(image):
    contours, hierarchy = get_contours(image)
    croped_cards = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (width, height))
        dimensions = warped.shape
        if dimensions[0] < dimensions[1]:
            warped = np.rot90(warped)
        if len(contours) > 1:
            croped_cards.append(warped)
        else:
            return warped
    return croped_cards