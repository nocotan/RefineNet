# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np

import chainer.functions as F
from chainer import cuda
from chainer import Variable
from chainer import serializers
from chainer.optimizer import WeightDecay
from chainer.optimizers import Adam, MomentumSGD
from chainer.iterators import MultiprocessIterator

from refinenet.models import RefineResNet
from refinenet.datasets import ImageDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./datasets")
    parser.add_argument("--data_list", type=str, default="train.txt")
    parser.add_argument("--n_class", type=int, default=5)
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--snapshot_dir", type=str, default="./snapshots")
    parser.add_argument("--save_steps", type=int, default=50)
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    model = RefineResNet(n_class=args.n_class)
    if args.model is not None:
        serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
        xp = cuda.cupy
    else:
        xp = np

    optimizer = Adam()
    #optimizer = MomentumSGD()
    optimizer.setup(model)
    optimizer.add_hook(WeightDecay(1e-5), "hook_wd")

    train_dataset = ImageDataset(args.data_dir,
                                 args.data_list,
                                 crop_size=(320, 320))
    train_iterator = MultiprocessIterator(train_dataset,
                                          batch_size=args.batch_size,
                                          repeat=True,
                                          shuffle=True)

    step = 0
    for zipped_batch in train_iterator:
        step += 1
        x = Variable(xp.array([zipped[0] for zipped in zipped_batch]))
        y = Variable(xp.array([zipped[1] for zipped in zipped_batch], dtype=xp.int32))
        pred = xp.array(model(x).data, dtype=xp.float32)
        loss = F.softmax_cross_entropy(pred, y)
        optimizer.update(F.softmax_cross_entropy, pred, y)

        print("Step: {}, Loss: {}".format(step, loss.data))
        if step % args.save_steps == 0:
            serializers.save_npz(
                os.path.join(args.snapshot_dir, "model_{}.npz".format(step)),
                model
            )

        if step >= args.n_steps:
            break


if __name__ == "__main__":
    main()
