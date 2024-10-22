import os
import argparse
from glob import glob
import numpy as np
from model import R2RNet,IAT,APBSN
# from ptflops import get_model_complexity_info
import torch,thop
parser = argparse.ArgumentParser(description='')

parser.add_argument('--gpu_id', dest='gpu_id', default="0",
                    help='GPU ID (-1 for CPU)')
parser.add_argument('--epochs', dest='epochs', type=int, default=20,
                    help='number of total epochs')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2,
                    help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=120,
                    help='patch size')
parser.add_argument('--lr', dest='lr', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--data_dir', dest='data_dir',

                    # default=r'D:/lunwen2/newR2RNet-main/Training data/niqe-256-128',
                    default=r'./Training data/SIDD_s256_o128_trainlol',
                    help='directory storing the training data')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='./ckpts/',
                    help='directory for checkpoints')

args = parser.parse_args()


def train(model):

    lr = args.lr * np.ones([args.epochs])
    lr[10:] = lr[0] / 10.0

    # train_low_data_names = glob(args.data_dir + '/Huawei/low/*.jpg') +   glob(args.data_dir + '/Nikon/low/*.jpg')
    train_low_data_names = glob(args.data_dir + '/low/*.png')+glob(args.data_dir + '/low/*.jpg')+glob(args.data_dir + '/low/*.bmp')
    train_low_data_names.sort()
    # train_high_data_names = glob(args.data_dir + '/Huawei/high/*.jpg') + \
    #                         glob(args.data_dir + '/Nikon/high/*.jpg')
    train_high_data_names = glob(args.data_dir + '/high/*.png')+glob(args.data_dir + '/high/*.jpg')+glob(args.data_dir + '/high/*.bmp')
    train_high_data_names.sort()
    eval_low_data_names = glob('./data/eval15/low/*.png')
    eval_low_data_names.sort()
    eval_high_data_names = glob('./data/eval15/high/*.png')
    eval_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('Number of training data: %d' % len(train_low_data_names))

    # M
    # summary = 'macs: %s -->' % (macs / 1e9) + '\n'
    model.train(train_low_data_names,
                train_high_data_names,
                eval_low_data_names,
                eval_high_data_names,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                epoch=args.epochs,
                lr=lr,
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=1,
                train_phase="Decom")

    model.train(train_low_data_names,
                train_high_data_names,
                eval_low_data_names,
                eval_high_data_names,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                epoch=args.epochs,
                lr=lr,
                vis_dir=args.vis_dir,
                ckpt_dir=args.ckpt_dir,
                eval_every_epoch=1,
                train_phase="Denoise")

    # model.train(train_low_data_names,
    #             train_high_data_names,
    #             eval_low_data_names,
    #             eval_high_data_names,
    #             batch_size=args.batch_size,
    #             patch_size=args.patch_size,
    #             epoch=args.epochs,
    #             lr=lr,
    #             vis_dir=args.vis_dir,
    #             ckpt_dir=args.ckpt_dir,
    #             eval_every_epoch=1,
    #             train_phase="Relight")


if __name__ == '__main__':
    if args.gpu_id != "-1":
        # Create directories for saving the checkpoints and visuals
        args.vis_dir = args.ckpt_dir + '/visuals/'
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
        if not os.path.exists(args.vis_dir):
            os.makedirs(args.vis_dir)
        # Setup the CUDA env
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        # Create the model
        model = R2RNet().cuda()
        # Train the model
        train(model)


    else:
        # CPU mode not supported at the moment!
        raise NotImplementedError
