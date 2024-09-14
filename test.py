import torch
import torch.nn as nn
import torchvision
import torch.optim
import numpy as np

import os
import argparse
import utils
from tqdm import tqdm

from networks import network, d_network
from networks.Ablation_Net import network_nores, d_network_norsb

import time

def test(config):
    d_net = d_network.d_net().cuda()
    enhan_net = network.net().cuda()
    utils.load_checkpoint(d_net, os.path.join(config.checkpoint_path, config.d_net_name, 'model_best.pth'))
    utils.load_checkpoint(enhan_net, os.path.join(config.checkpoint_path, config.net_name, 'model_best.pth'))
    
    print("gpu_id:", config.cudaid)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
    device_ids = [i for i in range(torch.cuda.device_count())]

    if torch.cuda.device_count() > 1:
        d_net = nn.DataParallel(d_net, device_ids=device_ids)
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)
        
    print(os.path.join(config.ori_images_path, config.dataset_name))
    test_dataset = utils.test_loader(os.path.join(config.ori_images_path, config.dataset_name))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                               num_workers=config.num_workers, drop_last=False, pin_memory=True)
    result_dir = os.path.join(config.result_path, config.net_name, config.dataset_name)

    enhan_net.eval()
    d_net.eval()

    with torch.no_grad():
        for _, (img_ori, filenames) in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            img_ori = img_ori.cuda()
            depth_image = d_net(img_ori, False)
            enhan_image = enhan_net(img_ori,depth_image)
            for i in range(len(enhan_image)):                
                torchvision.utils.save_image(enhan_image[i], os.path.join(result_dir, filenames[i]))
            # torchvision.utils.save_image(enhan_image, os.path.join(result_dir, filenames[0]))
            # print(filenames[0], "is done!")

if __name__ == '__main__':
    """
        param orig_images_path:original underwater images
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--net_name', type=str, default="net_UC")
    parser.add_argument('--d_net_name', type=str, default="d_net")
    parser.add_argument('--dataset_name', type=str, default="UIEB")
    parser.add_argument('--ori_images_path', type=str, default="../../../share/UIE/datasets/test/")
    
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--checkpoint_path', type=str, default="./trained_model/")
    parser.add_argument('--result_path', type=str, default="./result/")
    parser.add_argument('--cudaid', type=str, default="0,1,2,3", help="choose cuda device id 0-7).")

    config = parser.parse_args()

    if not os.path.exists(os.path.join(config.result_path, config.net_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name))

    if not os.path.exists(os.path.join(config.result_path, config.net_name, config.dataset_name)):
        os.mkdir(os.path.join(config.result_path, config.net_name, config.dataset_name))
    
    start_time = time.time()
    test(config)
    print("final_time:"+str(time.time()-start_time))
