'''
metrics

Contact: adalca@csail.mit.edu
'''

#  imports
import numpy as np


def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    # import ipdb; ipdb.set_trace()
    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        top = 2 * np.sum(np.logical_and(vol1 == lab, vol2 == lab))
        bottom = np.sum(vol1 == lab) + np.sum(vol2 == lab)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)

def get_disk_kernel(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))

def compute_boundary_acc(gt, seg, mask):

    gt = gt.astype(np.uint8)
    seg = seg.astype(np.uint8)
    mask = mask.astype(np.uint8)

    h, w = gt.shape

    min_radius = 1
    max_radius = (w+h)/300
    num_steps = 5

    seg_acc = [None] * num_steps
    mask_acc = [None] * num_steps

    for i in range(num_steps):
        curr_radius = min_radius + int((max_radius-min_radius)/num_steps*i)

        kernel = get_disk_kernel(curr_radius)
        boundary_region = cv2.morphologyEx(gt, cv2.MORPH_GRADIENT, kernel) > 0

        gt_in_bound = gt[boundary_region]
        seg_in_bound = seg[boundary_region]
        mask_in_bound = mask[boundary_region]

        num_edge_pixels = (boundary_region).sum()
        num_seg_gd_pix = ((gt_in_bound) * (seg_in_bound) + (1-gt_in_bound) * (1-seg_in_bound)).sum()
        num_mask_gd_pix = ((gt_in_bound) * (mask_in_bound) + (1-gt_in_bound) * (1-mask_in_bound)).sum()

        seg_acc[i] = num_seg_gd_pix / num_edge_pixels
        mask_acc[i] = num_mask_gd_pix / num_edge_pixels

    return sum(seg_acc)/num_steps, sum(mask_acc)/num_steps


def OS(mask_a,mask_b):
    mask_a = mask_a.flatten()
    mask_b = mask_b.flatten()
    return 1-np.sum(np.logical_and(mask_a,mask_b))/np.sum(mask_b)
def US(mask_a,mask_b):
    mask_a = mask_a.flatten()
    mask_b = mask_b.flatten()
    return 1-np.sum(np.logical_and(mask_a,mask_b))/np.sum(mask_a)
def RMS(mask_a,mask_b):
    return np.sqrt((OS(mask_a,mask_b)**2+US(mask_a,mask_b)**2)/2)
def AFI(mask_a,mask_b):
    return 0