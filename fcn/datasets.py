# -*- coding: utf-8 -*-
import os
import random
import cv2
import numpy as np
import PIL.Image

from chainer.dataset import dataset_mixin


class ImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, data_dir, data_list, crop_size=(500, 500)):
        self.data_dir = data_dir
        self.data_list = os.path.join(self.data_dir, data_list)
        self.crop_size = crop_size
        self.img_ids = [i_id.strip() for i_id in open(self.data_list)]

        self.files = []
        for name in self.img_ids:
            img_file = os.path.join(self.data_dir, "images/%s.jpg" % name)
            label_file = os.path.join(self.data_dir, "labels/%s.png" % name)
            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": name,
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale,
                           interpolation=cv2.INTER_NEAREST)

        return image, label

    def get_example(self, i):
        datafiles = self.files[i]
        image = cv2.imread(datafiles["image"], cv2.IMREAD_COLOR)
        label = np.asarray(PIL.Image.open(datafiles["label"]), dtype=np.int32)

        image, label = self.generate_scale_label(image, label)

        image = np.asarray(image, np.int32)
        image -= (128, 128, 128)

        img_h, img_w = label.shape
        pad_h = max(self.crop_size[0] - img_h, 0)
        pad_w = max(self.crop_size[1] - img_w, 0)

        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(0.0, 0.0, 0.0))
            lbl_pad = cv2.copyMakeBorder(label, 0, pad_h, 0,
                                         pad_w, cv2.BORDER_CONSTANT,
                                         value=(255,))
        else:
            img_pad, lbl_pad = image, label

        img_h, img_w = lbl_pad.shape

        h_off = random.randint(0, img_h - self.crop_size[0])
        w_off = random.randint(0, img_w - self.crop_size[1])
        image = np.asarray(img_pad[h_off:h_off+self.crop_size[0],
                                   w_off:w_off+self.crop_size[1]], np.float32)
        label = np.asarray(lbl_pad[h_off:h_off+self.crop_size[0],
                                   w_off:w_off+self.crop_size[1]], np.float32)

        image = image.transpose((2, 0, 1))
        flip = np.random.choice(2) * 2 - 1
        image = image[:, :, ::flip]
        label = label[:, ::flip]

        return image.copy(), label.copy()
