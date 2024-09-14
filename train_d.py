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
from loss import torchPSNR

from networks.Ablation_Net import d_network_norsb

def train(config):
    d_net = d_network_norsb.d_net().cuda()

    print("gpu_id:", config.cudaid)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.cudaid
    device_ids = [i for i in range(torch.cuda.device_count())]

    if torch.cuda.device_count() > 1:
        d_net = nn.DataParallel(d_net, device_ids=device_ids)

    train_dataset = utils.train_val_loader(config.depth_images_path,config.ori_images_path, mode="d_train")
    val_dataset = utils.train_val_loader(config.depth_images_path,config.ori_images_path, mode="d_val")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=False,num_workers=config.num_workers, pin_memory=True)

    ######### Adam optimizer ###########
    optimizer = optim.Adam(d_net.parameters(), lr=config.lr)
    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs - warmup_epochs,
                                                            eta_min=config.lr)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    criterion_l1 = nn.L1Loss()

    d_net.train()

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
                # enhanced_image, reconstruct_image = d_net(img_ori, True)
                enhanced_image, _ = d_net(img_ori, True)
                l1_loss1 = criterion_l1(img_clean, enhanced_image)
                # l1_loss2 = criterion_l1(img_ori, reconstruct_image)
                # print("l1_loss1:" + str(3*l1_loss1))
                # print("l1_loss2:" + str(l1_loss2))
                # print("all_loss:" + str(3*l1_loss1 + l1_loss2))                
                # sum_loss = 3*l1_loss1 + l1_loss2
                sum_loss = 3*l1_loss1

                train_loss.append(sum_loss.item())
                optimizer.zero_grad()
                sum_loss.backward()
                torch.nn.utils.clip_grad_norm_(d_net.parameters(), config.grad_clip_norm)
                optimizer.step()

            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(e)
                    torch.cuda.empty_cache()
                else:
                    raise e

        with open(config.checkpoint_path + "loss.log", "a+", encoding="utf-8") as f:
            s = "The %i Epoch mean_loss is :%f" % (epoch, np.mean(train_loss)) + "\n"
            f.write(s)

        # Validation Stage
        with torch.no_grad():
            for _, (img_clean, img_ori) in enumerate(val_loader):
                img_clean = img_clean.cuda()
                img_ori = img_ori.cuda()

                enhanced_image = d_net(img_ori, False)

                psnr = torchPSNR(img_clean, enhanced_image)
                val_psnr.append(psnr.item())

        val_psnr = np.mean(np.array(val_psnr))

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_epoch = epoch
            torch.save({'epoch': epoch,
                        'state_dict': d_net.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(config.checkpoint_path, "model_best.pth"))
        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format
              (epoch, time.time() - epoch_start_time, np.mean(train_loss), scheduler.get_lr()[0]))

        print("------------------------------------------------------------------")

        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
              (epoch, val_psnr, best_epoch, best_psnr))

        with open(config.checkpoint_path + "val_PSNR.log", "a+", encoding="utf-8") as f:
            f.write("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" %
                    (epoch, val_psnr, best_epoch, best_psnr) + "\n")

        torch.save({'epoch': epoch,
                    'state_dict': d_net.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(config.checkpoint_path, "model_latest.pth"))

if __name__ == "__main__":
    """
        input: underwater depth images and original underwater images
    	param depth_images_path:underwater depth images
    	param orig_images_path:original underwater images
    """

    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--depth_images_path', type=str, default="../../../share/UIE/datasets/train/UWCNN3/target/")
    parser.add_argument('--ori_images_path', type=str, default="../../../share/UIE/datasets/train/UWCNN3/input/")

    parser.add_argument('--resume', type=str, default='results')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--checkpoint_path', type=str, default="trained_model/d_net/")
    parser.add_argument('--cudaid', type=str, default="0,1,2,3",help="choose cuda device id 0-7).")

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_path):
        os.mkdir(config.checkpoint_path)

    torch.cuda.empty_cache()
    s = time.time()
    train(config)
    e = time.time()
    print(str(e-s))
