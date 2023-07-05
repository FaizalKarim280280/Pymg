import pymg as pymg
import matplotlib.pyplot as plt
import cv2
import numpy as np


def main():
    PATH = '../coldplay.jpg'
    img = pymg.load_img(PATH, retain_png=False)

    img = pymg.convert2gray(img)
    img = pymg.normalize_image(img, between=(0, 1))
    print(img)

    img = pymg.discretize_mask(img, threshold=0.5)

    # img = pymg.resize_image(img, size = (100, 100))

    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
