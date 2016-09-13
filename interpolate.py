#!/usr/bin/env python3.4
#
# Irmak Sirer
# License: MIT
# 2016-09

import argparse
import os
import tensorflow as tf

from model import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpointDir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='interpolation')
parser.add_argument('--vector1', type=str, default='')
parser.add_argument('--vector2', type=str, default='')

args = parser.parse_args()

assert(os.path.exists(args.checkpointDir))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=args.imgSize,
                  checkpoint_dir=args.checkpointDir, lam=args.lam)
    dcgan.interpolate(config=args)
