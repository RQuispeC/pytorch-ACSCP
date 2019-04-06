import os
import os.path as osp
import torch
import numpy as np
import sys
from torch.nn.utils import clip_grad_norm_

from architecture.crowd_count import CrowdCounter
from architecture import network
from architecture.data_loader import ImageDataLoader
from architecture.timer import Timer
from architecture import utils
from architecture.evaluate_model import evaluate_model

import argparse

from manage_data import dataset_loader
from manage_data.utils import Logger, mkdir_if_missing

import time
EPSILON = 1e-10
MAXIMUM_CNT = 2.7675038540969217
NORMALIZE_ADD = np.log(EPSILON) / 2.0

parser = argparse.ArgumentParser(description='Train crowd counting network using data augmentation')
# Datasets
parser.add_argument('-d', '--dataset', type=str, default='ucf',
                    choices=dataset_loader.get_names())
#Data augmentation hyperpameters
parser.add_argument('--force-den-maps', action='store_true', help="force generation of dentisity maps for original dataset, by default it is generated only once")
parser.add_argument('--force-augment', action='store_true', help="force generation of augmented data, by default it is generated only once")
parser.add_argument('--displace', default=70, type=int,help="displacement for sliding window in data augmentation, default 70")
parser.add_argument('--size-x', default=256, type=int, help="width of sliding window in data augmentation, default 200")
parser.add_argument('--size-y', default=256, type=int, help="height of sliding window in data augmentation, default 300")
parser.add_argument('--people-thr', default=0, type=int, help="threshold of people sliding window in data augmentation, default 200")
parser.add_argument('--not-augment-noise', action='store_true', help="not use noise for data augmetnation, default True")
parser.add_argument('--not-augment-light', action='store_true', help="not use bright & contrast for data augmetnation, default True")
parser.add_argument('--bright', default=10, type=int, help="bright value for bright & contrast augmentation, defaul 10")
parser.add_argument('--contrast', default=10, type=int, help="contrast value for bright & contrast augmentation, defaul 10")
parser.add_argument('--gt-mode', type=str, default='same', help="mode for generation of ground thruth.")

# Optimization options
parser.add_argument('--max-epoch', default=500, type=int,
                    help="maximum epochs to run")
parser.add_argument('--start-epoch', default=0, type=int,
                    help="manual epoch number (useful on restarts)")
parser.add_argument('--lr', '--learning-rate', default=0.00005, type=float,
                    help="initial learning rate")
parser.add_argument('--beta1', default=0.5, type=float,
                    help="training b1 for adam optimizer")
parser.add_argument('--beta2', default=0.999, type=float,
                    help="training b2 for adam optimizer")
parser.add_argument('--train-batch', default=32, type=int,
                    help="train batch size (default 32)")
# Miscs
parser.add_argument('--den-factor', type=float, default=1e3, help="factor to multiply for density maps to avoid too small values")
parser.add_argument('--overlap-test', action='store_true', help="overlap the sliding windows for test")
parser.add_argument('--seed', type=int, default=64678, help="manual seed")
parser.add_argument('--resume', type=str, default='', metavar='PATH', help="root directory where part/fold of previous train are saved")
parser.add_argument('--save-dir', type=str, default='log', help="path where results for each part/fold are saved")
parser.add_argument('--units', type=str, default='', help="folds/parts units to be trained, be default all folds/parts are trained")
parser.add_argument('--augment-only', action='store_true', help="run only data augmentation, default False")
parser.add_argument('--evaluate-only', action='store_true', help="run only data validation, --resume arg is needed, default False")
parser.add_argument('--save-plots', action='store_true', help="save plots of density map estimation (done only in test step), default False")

args = parser.parse_args()

def train(train_test_unit, out_dir_root):
    output_dir = osp.join(out_dir_root, train_test_unit.metadata['name'])
    mkdir_if_missing(output_dir)
    sys.stdout = Logger(osp.join(output_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    dataset_name = train_test_unit.metadata['name']
    train_path = train_test_unit.train_dir_img
    train_gt_path = train_test_unit.train_dir_den
    val_path =train_test_unit.test_dir_img
    val_gt_path = train_test_unit.test_dir_den

    #training configuration
    start_step = args.start_epoch
    end_step = args.max_epoch
    lr = args.lr

    #log frequency
    disp_interval = args.train_batch*20

    # ------------
    rand_seed = args.seed
    if rand_seed is not None:
        np.random.seed(rand_seed)
        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)

    # load net
    net = CrowdCounter()
    if not args.resume :
        network.weights_normal_init(net, dev=0.01)
    else:
        #network.weights_normal_init(net, dev=0.01) #init all layers in case of partial net load
        if args.resume[-3:] == '.h5':
            pretrained_model = args.resume
        else:
            resume_dir = osp.join(args.resume, pu.metadata['name'])
            pretrained_model = osp.join(resume_dir, 'best_model.h5')
        network.load_net(pretrained_model, net)
        print('Will apply fine tunning over', pretrained_model)
    net.cuda()
    net.train()

    optimizer_d_large = torch.optim.Adam(filter(lambda p: p.requires_grad, net.d_large.parameters()), lr=lr, betas = (args.beta1, args.beta2))
    optimizer_d_small = torch.optim.Adam(filter(lambda p: p.requires_grad, net.d_small.parameters()), lr=lr, betas = (args.beta1, args.beta2))
    optimizer_g_large = torch.optim.Adam(filter(lambda p: p.requires_grad, net.g_large.parameters()), lr=lr, betas = (args.beta1, args.beta2))
    optimizer_g_small = torch.optim.Adam(filter(lambda p: p.requires_grad, net.g_small.parameters()), lr=lr, betas = (args.beta1, args.beta2))

    # training
    train_loss = 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()

    #preprocess flags
    overlap_test = True if args.overlap_test else False

    data_loader = ImageDataLoader(train_path, train_gt_path, shuffle=True, batch_size = args.train_batch, test_loader = False)
    data_loader_val = ImageDataLoader(val_path, val_gt_path, shuffle=False, batch_size = 1, test_loader = True, img_width = args.size_x, img_height = args.size_y, test_overlap = overlap_test)
    best_mae = sys.maxsize

    for epoch in range(start_step, end_step+1):
        step = 0
        train_loss_gen_small = 0
        train_loss_gen_large = 0
        train_loss_dis_small = 0
        train_loss_dis_large = 0

        for blob in data_loader:
            step = step + args.train_batch
            im_data = blob['data']
            gt_data = blob['gt_density']
            idx_data = blob['idx']
            im_data_norm = im_data / 127.5 - 1. #normalize between -1 and 1
            gt_data = gt_data * args.den_factor

            optimizer_d_large.zero_grad()
            optimizer_d_small.zero_grad()
            density_map = net(im_data_norm, gt_data, epoch = epoch, mode = "discriminator")
            loss_d_small = net.loss_dis_small
            loss_d_large = net.loss_dis_large
            loss_d_small.backward()
            loss_d_large.backward()
            optimizer_d_small.step()
            optimizer_d_large.step()

            optimizer_g_large.zero_grad()
            optimizer_g_small.zero_grad()
            density_map = net(im_data_norm, gt_data, epoch = epoch, mode = "generator")
            loss_g_small = net.loss_gen_small
            loss_g_large = net.loss_gen_large
            loss_g = net.loss_gen
            loss_g.backward() # loss_g_large + loss_g_small
            optimizer_g_small.step()
            optimizer_g_large.step()
 
            density_map /= args.den_factor
            gt_data /= args.den_factor
            
            train_loss_gen_small += loss_g_small.data.item()
            train_loss_gen_large += loss_g_large.data.item()
            train_loss_dis_small += loss_d_small.data.item()
            train_loss_dis_large += loss_d_large.data.item()

            step_cnt += 1
            if step % disp_interval == 0:
                duration = t.toc(average=False)
                fps = step_cnt / duration
                density_map = density_map.data.cpu().numpy()
                train_batch_size = gt_data.shape[0]
                gt_count = np.sum(gt_data.reshape(train_batch_size, -1), axis = 1)
                et_count = np.sum(density_map.reshape(train_batch_size, -1), axis = 1)
                
                if args.save_plots:
                    plot_save_dir = osp.join(output_dir, 'plot-results-train/')
                    mkdir_if_missing(plot_save_dir)
                    utils.save_results(im_data, gt_data, density_map, idx_data, plot_save_dir, loss = args.loss)

                print("epoch: {0}, step {1}/{5}, Time: {2:.4f}s, gt_cnt: {3:.4f}, et_cnt: {4:.4f}, mean_diff: {6:.4f}".format(epoch, step, 1./fps, gt_count[0],et_count[0], data_loader.num_samples, np.mean(np.abs(gt_count - et_count))))
                re_cnt = True
        
            if re_cnt:
                t.tic()
                re_cnt = False

        save_name = os.path.join(output_dir, '{}_{}_{}.h5'.format(train_test_unit.to_string(), dataset_name,epoch))
        network.save_net(save_name, net)

        #calculate error on the validation dataset 
        mae,mse = evaluate_model(save_name, data_loader_val, epoch = epoch, den_factor = args.den_factor)
        if mae < best_mae:
            best_mae = mae
            best_mse = mse
            best_model = '{}_{}_{}.h5'.format(train_test_unit.to_string(),dataset_name,epoch)
            network.save_net(os.path.join(output_dir, "best_model.h5"), net)

        print("Epoch: {0}, MAE: {1:.4f}, MSE: {2:.4f}, loss gen small: {3:.4f}, loss gen large: {4:.4f}, loss dis small: {5:.4f}, loss dis large: {6:.4f}, loss: {7:.4f}".format(epoch, mae, mse, train_loss_gen_small, train_loss_gen_large, train_loss_dis_small, train_loss_dis_large, train_loss_gen_small + train_loss_gen_large + train_loss_dis_small + train_loss_dis_large))
        print("Best MAE: {0:.4f}, Best MSE: {1:.4f}, Best model: {2}".format(best_mae, best_mse, best_model))

def test(train_test_unit, out_dir_root):
    output_dir = osp.join(out_dir_root, train_test_unit.metadata['name'])
    mkdir_if_missing(output_dir)
    sys.stdout = Logger(osp.join(output_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    dataset_name = train_test_unit.metadata['name']
    val_path =train_test_unit.test_dir_img
    val_gt_path = train_test_unit.test_dir_den

    if not args.resume :
        pretrained_model = osp.join(output_dir, 'best_model.h5')
    else:
        if args.resume[-3:] == '.h5':
            pretrained_model = args.resume
        else:
            resume_dir = osp.join(args.resume, train_test_unit.metadata['name'])
            pretrained_model = osp.join(resume_dir, 'best_model.h5')
    print("Using {} for testing.".format(pretrained_model))

    overlap_test = True if args.overlap_test else False

    data_loader = ImageDataLoader(val_path, val_gt_path, shuffle=False, batch_size = 1, test_loader = True, img_width = args.size_x, img_height = args.size_y, test_overlap = overlap_test)
    mae,mse = evaluate_model(pretrained_model, data_loader, save_test_results=args.save_plots, plot_save_dir=osp.join(output_dir, 'plot-results-test/'), den_factor = args.den_factor)

    print("MAE: {0:.4f}, MSE: {1:.4f}".format(mae, mse))

def main():
    #augment data

    force_create_den_maps = True if args.force_den_maps else False
    force_augmentation = True if args.force_augment else False
    augment_noise = False if args.not_augment_noise else True 
    augment_light = False if args.not_augment_light else True
    augment_only = True if args.augment_only else False


    dataset = dataset_loader.init_dataset(name=args.dataset
    , force_create_den_maps = force_create_den_maps
    , force_augmentation = force_augmentation
    #sliding windows params
    , gt_mode = args.gt_mode
    , displace = args.displace
    , size_x= args.size_x
    , size_y= args.size_y
    , people_thr = args.people_thr
    #noise_params 
    , augment_noise = augment_noise
    #light_params
    , augment_light = augment_light
    , bright = args.bright
    , contrast = args.contrast)

    if augment_only:
        set_units = [unit.metadata['name'] for unit in dataset.train_test_set]
        print("Dataset train-test units are: {}".format(", ".join(set_units)))
        print("Augment only - network will not be trained")
        return

    metadata = "_".join([args.dataset, dataset.signature()])
    out_dir_root = osp.join(args.save_dir, metadata)

    if args.units != '':
        units_to_train = [name.strip() for name in args.units.split(',')]
        set_units = [unit.metadata['name'] for unit in dataset.train_test_set]
        print("Dataset train-test units are: {}".format(", ".join(set_units)))
        set_units = set(set_units)
        for unit in units_to_train:
            if not unit in set_units:
                raise RuntimeError("Invalid '{}' train-test unit".format(unit))
    else:
        units_to_train = [unit.metadata['name'] for unit in dataset.train_test_set]
    units_to_train = set(units_to_train)
    for train_test in dataset.train_test_set:
        if train_test.metadata['name'] in units_to_train:
            if args.evaluate_only:
                print("Testing {}".format(train_test.metadata['name']))
                test(train_test, out_dir_root)
            else:
                print("Training {}".format(train_test.metadata['name']))
                train(train_test, out_dir_root)
                print("Testing {}".format(train_test.metadata['name']))
                test(train_test, out_dir_root)

if __name__ == '__main__':
    main()    

