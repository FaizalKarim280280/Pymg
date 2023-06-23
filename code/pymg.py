import numpy as np
import matplotlib.pyplot as plt
import cv2 as op
from typing import List, Tuple, Union


def resize_image(
        img: List[float],
        size: Union[str, Tuple[int, int]]):

    """
    Resizes the image.
    
    Args:
        img (List[float]): The image matrix.
        size (str or (int, int)): Target size of the image.
        
    Returns:
        list: Resized image
        
    Example:
        >>> img = np.random.randn(100, 200, 3)
        >>> img.shape
        (100, 200, 3)
        >>> img = resize_image(img = img, size = 'half')
        >>> img.shape
        (50, 100, 3)
        >>> img = resize_image(img = img, size = (80, 150))
        >>> img.shape
        (80, 150, 3)
        
    """
    
    width, height = img.shape[0], img.shape[1]
    
    if isinstance(size, str):    
        if size == 'original':
            size = (height, width)
        elif size == 'half':
            size = (int(height/2), int(width/2))
        else:
            raise ValueError("Unknown value '{}' for size parameter".format(size))
        
    img = op.resize(img, size)

    return img


def normalize_image(
        img: List[float],
        between: Tuple[float, float]):
    
    """
    Normalizes an image between a given range.
    
    Args:
        img (List[float]): The image matrix.
        between (Tuple[float, float]): Target range of the image pixels.
    
    Returns:
        list: Normalized image.
        
    Example:
        >>> img = np.random.uniform(0, 255, (100, 200, 3))
        >>> img.min(), img.max()
        (0, 255)
        >>> img = normalize_image(img = img, between = (0,1))
        >>> img.min(), img.max()
        (0, 1)
        
    """

    x, y = between[0], between[1]
    min_p, max_p = np.min(img), np.max(img)
    img = (img - min_p)/(max_p - min_p)

    img = ((y - x) * img) + x
    return img


def load_img(
        PATH: str,
        size: Union[str, Tuple[int, int]] = 'original',
        between: Tuple[float, float] = (0, 255),
        retain_png: bool = False):
    
    """
    Load image with the given path.
    
    Args:
        PATH (str): Path of the image.
        size (str or Tuple[int, int]): Target size of the image. Defaults to 'original'.
        between (Tuple[float, float]): Target range of the image pixels. Defaults to (0, 255).
        retain_png (bool): Retain the fourth channel of the image. Defaults to False
        
    Returns:
        list: Loaded image with necessary resizing and normalizing. 
    
    Example:
        >>> img = load_image('./image.jpg', size = 'half', between = (0, 1)) #Original dim of image: (100, 300)
        >>> img.shape
        (50, 150)
        >>> img.min(), img.max()
        (0, 1)
         
    """

    img = plt.imread(PATH)

    channels = len(img.shape)

    if channels != 4 and retain_png:
        raise TypeError(
            "Cannot retain png for image shape={}. Image needs to have the fourth channel.".format(img.shape))

    if not retain_png:
        img = img[:, :, :3]

    img = resize_image(img=img, size=size)
    img = normalize_image(img=img, between=between)

    return img


def convert2gray(
        img: List[float]):
    
    """
    Converts RGB image to grayscale
    
    Args:
        img (List[float]): The image matrix.
        
    Returns:
        list: Grayscale image.
        
    Example:
        >>> # img = (100, 200, 3)
        >>> img = convert2gray(img)
        >>> img.shape
        (100, 200)
        
    """

    if len(img.shape) != 3:
        raise TypeError("Should be a RGB image")

    img = img.astype(np.uint8)
    img = op.cvtColor(img, op.COLOR_RGB2GRAY)
    return img


def discretize_mask(
        img: List[float],
        threshold: float = 0.5):
    
    """
    Converts each pixel of the mask to either 0 or 1 based on a threshold.
    
    Args:
        img (List[float]): The image matrix.
        threshold (str): The treshold value to determine the pixel values.
            
    Returns:
        list: Thresholded image matrix.
        
    Example:
        >>> img = load_image('mask.jpeg')
        >>> img.shape
        (100, 200)
        >>> img = discretize_mask(img, threshold = 0.5)
        >>> img.min(), img.max()
        (0, 1)
    
    """

    if len(img.shape) != 2:
        raise TypeError(
            'Image needs to have shape = (height, width). Input image shape is = {}'.format(img.shape))
        
    if not (0 <= threshold <= 1):
        raise ValueError(
            'Threshold value should in between [0, 1]. Input threshold value = {}'.format(threshold))
        
    img = np.where(img < threshold, 0, 1)
    return img


