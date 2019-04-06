from architecture.crowd_count import CrowdCounter
import architecture.network as network
import numpy as np
import torch

from manage_data.utils import Logger, mkdir_if_missing
from architecture import utils

def evaluate_model(trained_model, data_loader, epoch = 0, save_test_results = False, plot_save_dir = "/tmp/", den_factor = 1e3):
    net = CrowdCounter()
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0

    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        idx_data = blob['idx']
        new_shape = blob['new_shape']
        orig_shape = blob['orig_shape']
        im_data_norm = im_data / 127.5 - 1. #normalize between -1 and 1
        gt_data = gt_data * den_factor

        density_map = net(im_data_norm, epoch = epoch)
        density_map = density_map.data.cpu().numpy()
        density_map /= den_factor
        gt_data /= den_factor
        im_data, gt_data = data_loader.recontruct_test(im_data, gt_data, orig_shape, new_shape)
        _, density_map = data_loader.recontruct_test(im_data_norm, density_map, orig_shape, new_shape)
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        print("image {} gt {:.3f} es {:.3f}".format(idx_data[0], gt_count, et_count))
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))

        if save_test_results:
            print("Plotting results")
            mkdir_if_missing(plot_save_dir)
            utils.save_results(im_data, gt_data, density_map, idx_data, plot_save_dir)

    mae = mae/data_loader.get_num_samples()
    mse = np.sqrt(mse/data_loader.get_num_samples())
    return mae,mse