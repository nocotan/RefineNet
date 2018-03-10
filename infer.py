# -*- coding: utf-8 -*-
import argparse
import os
import cv2
import numpy as np
from PIL import Image

import chainer.functions as F
from chainer import cuda
from chainer import serializers

from refinenet.models import RefineResNet
from color_map import make_color_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--n_class", type=int, default=5)
    args = parser.parse_args()
    print(args)

    model = RefineResNet(n_class=args.n_class)
    if model is not None:
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    org = cv2.imread(args.image, cv2.IMREAD_COLOR)
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)

    org = cv2.resize(org, (160, 160))
    image = cv2.resize(image, (160, 160))

    image = np.asarray(image, dtype=np.float32)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, axis=0)
    pred = F.softmax(model(image)).data
    pred = pred[0].argmax(axis=0)
    print(pred)

    color_map = make_color_map()

    row, col = pred.shape
    dst = np.ones((row, col, 3))
    for i in range(args.n_class):
        dst[pred == i] = color_map[i]
    img = Image.fromarray(np.uint8(dst))

    trans = Image.new('RGBA', img.size, (0, 0, 0, 0))
    w, h = img.size
    for x in range(w):
        for y in range(h):
            pixel = img.getpixel((x, y))
            if (pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0)or \
               (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
                continue
            trans.putpixel((x, y), pixel)

    if not os.path.exists("out"):
        os.mkdir("out")

    cv2.imwrite("out/original.jpg", org)
    trans.save("out/pred.png")

    o = cv2.imread("out/original.jpg", 1)
    p = cv2.imread("out/pred.png", 1)

    pred = cv2.addWeighted(o, 0.6, p, 0.4, 0.0)

    cv2.imwrite("out/pred_alpha.png", pred)

    os.remove("out/original.jpg")
    #os.remove("out/pred.png")


if __name__ == "__main__":
    main()
