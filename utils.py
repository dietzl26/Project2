"""Utility functions."""

import numpy as np
import pandas as pd

from PIL import Image

IMAGE_SIZE = (252, 378)


def load_mask(mask_path):
    """Loads the segmentation mask from the specified path.

    Inputs:
        mask_path (str): the path from which the segmentation mask will be read.
        It should have the format "/PATH/TO/LOAD/DIR/XXXX_mask.png".

    Outputs:
        mask (np.array): segmentation mask as a numpy array.
    """
    mask = np.asarray(Image.open(mask_path)).astype(int)
    if mask.max() > 1:
        mask = mask // 255
    return mask


def mask_to_rle(mask):
    """
    Convert a binary mask (2D numpy array) to RLE (column-major).
    Returns a string of space-separated values.
    """
    pixels = mask.flatten(order='F')  # Fortran order (column-major)
    pixels = np.concatenate([[0], pixels, [0]])  # pad with zeros to catch transitions
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[::2]  # calculate run lengths
    return ' '.join(str(x) for x in runs)


def compute_iou(pred_mask, gt_mask, eps=1e-6):
    """Computes the IoU between two numpy arrays: pred_mask and gt_mask.

    Inputs:
        pred_mask (np.array): dtype:int, shape:(image_height, image_width), values are 0 or 1.
        gt_mask (np.array): dtype:int, shape:(image_height, image_width), values are 0 or 1.
        eps (float): epsilon to smooth the division in order to avoid 0/0.

    Outputs:
        iou_score (float)
    """
    intersection = (
        (pred_mask & gt_mask).astype(float).sum()
    )  # will be zero if gt=0 or pred=0
    union = (pred_mask | gt_mask).astype(float).sum()  # will be zero if both are 0
    iou = (intersection + eps) / (
        union + eps
    )  # we smooth our division by epsilon to avoid 0/0
    iou_score = iou.mean()
    return iou_score


def save_predictions(image_ids, pred_masks, save_path='submission.csv'):
    '''
    image_ids: list of image_ids [0000, 0001, ...]
    pred_masks: binary 2D numpy array
    '''
    assert len(image_ids) == len(pred_masks)
    predictions = {'ImageId': [], 'EncodedPixels': []}
    for i in range(len(image_ids)):
        mask = pred_masks[i]
        mask_rle = mask_to_rle(mask)
        predictions['ImageId'].append(image_ids[i])
        predictions['EncodedPixels'].append(mask_rle if mask_rle else ' ')

    pd.DataFrame(predictions).to_csv(save_path, index=False)
