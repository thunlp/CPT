# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))


class RefCOCOCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        # ret_imgs, ret_targets, caption, ret_color_names, idx, scale
        transposed_batch = list(zip(*batch))

        images = [y for x in transposed_batch[0] for y in x]
        images = to_image_list(images, self.size_divisible)

        targets = transposed_batch[1]
        targets = [y for x in targets for y in x]

        captions = transposed_batch[2]

        color_names = transposed_batch[3]

        rects = transposed_batch[4]

        img_ids = transposed_batch[5]

        return images, targets, captions, color_names, rects, img_ids


class VGCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        colors = transposed_batch[2]
        pair_labels = transposed_batch[3]
        img_ids = transposed_batch[4]
        rel_names = transposed_batch[5]
        return images, targets, colors, pair_labels, img_ids, rel_names