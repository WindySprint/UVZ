import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os
import argparse
import time
import utils
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from loss import Charbonnier_Loss, SSIM_Loss, VGG_Loss, torchPSNR

from networks import network, d_network
from networks.Ablation_Net import network_nores, d_network_norsb

def train(config):
    d_net = d_network.d_net().cuda()
    utils.load_checkpoint(d_net, os.path.join(config.checkpoint_path, config.d_net_name, 'model_best.pth'))
    enhan_net = network.net().cuda()

    print("gpu_id:", config.cudaid)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
    device_ids = [i for i in range(torch.cuda.device_count())]
    
    if torch.cuda.device_count() > 1:
        d_net = nn.DataParallel(d_net, device_ids=device_ids)
        enhan_net = nn.DataParallel(enhan_net, device_ids=device_ids)

    train_dataset = utils.train_val_loader(config.enhan_images_path,config.ori_images_path)
    val_dataset = utils.train_val_loader(config.enhan_images_path,config.ori_images_path, mode="val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)

    ######### Adam optimizer ###########
    # optimizer = optim.Adam(filter(lambda p:p.requires_grad, enhan_net.parameters()), lr=config.lr)
    optimizer = optim.Adam(enhan_net.parameters(), lr=config.lr)
    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs - warmup_epochs,
                                                            eta_min=config.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    criterion_char = Charbonnier_Loss()
    criterion_sl1 = nn.SmoothL1Loss()
    # criterion_vgg = VGG_Loss()
    criterion_ssim = SSIM_Loss()

    enhan_net.train()
    d_net.eval()

    # Record best index and corresponding epoch
    best_psnr = 0
    best_epoch = 0

    for epoch in range(1, config.num_epochs+1):
        epoch_start_time = time.time()
        # Record train loss and validation index
        train_loss = []
        val_psnr = []
        print("*" * 30 + "The %i epoch" % epoch + "*" * 30+'\n')
        
        for _, (img_clean, img_ori) in enumerate(tqdm(train_loader)):
            img_clean = img_clean.cuda()
            img_ori = img_ori.cuda()

            try:
                d = d_net(img_ori, False)
                enhanced_image = enhan_net(img_ori, d)
                char_loss = criterion_char(img_clean, enhanced_image)
                # sl1_loss = criterion_sl1(img_clean, enhanced_image)
                ssim_loss = criterion_ssim(img_clean, enhanced_image)
                ssim_loss = 1 - ssim_loss
                
                # sum_loss = char_loss + 10 * sl1_loss + 0.5 * ssim_loss
                sum_loss = char_loss + 0.5 * ssim_loss
                
                # print("char_loss:" + str(char_loss))
                # print("ssim_loss:" + str(0.5*ssim_loss))
                # print("vgg_loss:" + str(0.1*vgg_loss))
                # print("all_loss:" + str(char_loss + 10 * sl1_loss + 0.5*ssim_loss))
                train_loss.append(sum_loss.item())
                optimizer.zero_grad()
                sum_loss.backward()
                torch.nn.utils.clip_grad_norm_(enhan_net.parameters(), config.grad_clip_norm)
                optimizer.step()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

        with open(os.path.join(config.checkpoint_path, config.net_name, "loss.log"), "a+", encoding="utf-8") as f:
            s = "The %i Epoch mean_loss is :%f" % (epoch, np.mean(train_loss)) + "\n"
            f.write(s)

        # Validation Stage
        with torch.no_grad():
            for _, (img_clean, img_ori) in enumerate(val_loader):
                img_clean = img_clean.cuda()
                img_ori = img_ori.cuda()
                d = d_net(img_ori, False)
                enhanced_image = enhan_net(img_ori, d)

                psnr = torchPSNR(img_clean, enhanced_image)
                val_psnr.append(psnr.item())

        val_psnr = np.mean(np.array(val_psnr))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': enhan_net.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(config.checkpoint_path, config.net_name, "model_best.pth"))
        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format
              (epoch, time.time() - epoch_start_time, np.mean(train_loss), scheduler.get_lr()[0]))

        print("------------------------------------------------------------------")

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
              (epoch, val_psnr, best_epoch, best_psnr))

        with open(os.path.join(config.checkpoint_path, config.net_name, "val_PSNR.log"), "a+", encoding="utf-8") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                    (epoch, val_psnr, best_epoch, best_psnr) + "\n")

        torch.save({'epoch': epoch,
                    'state_dict': enhan_net.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(config.checkpoint_path, config.net_name, "model_latest.pth"))

if __name__ == "__main__":
    """
        input:enhanced underwater images and original underwater images
    	param enhan_images_path:enhanced underwater images
    	param orig_images_path:original underwater images
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--net_name', type=str, default="net_UC")
    parser.add_argument('--d_net_name', type=str, default="d_net")
    parser.add_argument('--enhan_images_path', type=str, default="../../../share/UIE/datasets/train/CYCLE/target/")
    parser.add_argument('--ori_images_path', type=str, default="../../../share/UIE/datasets/train/CYCLE/input/")

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--checkpoint_path', type=str, default="./trained_model/")
    parser.add_argument('--cudaid', type=str, default="0,1,2,3",help="choose cuda device id 0-7).")

    config = parser.parse_args()

    if not os.path.exists(os.path.join(config.checkpoint_path, config.net_name)):
        os.mkdir(os.path.join(config.checkpoint_path, config.net_name))

    torch.cuda.empty_cache()
    s = time.time()
    train(config)
    e = time.time()
    print(str(e-s))
