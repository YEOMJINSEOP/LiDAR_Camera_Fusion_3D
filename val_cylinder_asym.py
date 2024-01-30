# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import yaml

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")

def main(args):
    pytorch_device = torch.device('cuda:0')
    config_path = args.config_path
    configs = load_config_data(config_path)
    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    data_dir = args.demo_folder
    demo_label_dir = args.demo_label_folder
    save_dir = args.save_folder + "/"

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']
    wd = train_hypers['weight_decay']      # weight decay
    amp = train_hypers['mixed_fp16']
    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))[1:] - 1
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label + 1]

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):
        my_model = load_checkpoint(model_load_path, my_model)

    my_model.to(pytorch_device)

    # Mixed Precision
    amp_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # EPS is a fixed value to prevent MixedPrecision training errors.
    optimizer = optim.AdamW(my_model.parameters(), lr=train_hypers["learning_rate"], eps=1e-4, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, train_hypers['max_num_epochs'])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)

    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)
    with open(dataset_config["label_mapping"], 'r') as stream:
        semkittiyaml = yaml.safe_load(stream)
    inv_learning_map = semkittiyaml['learning_map_inv']
    
    my_model.eval()
    hist_list = []
    val_loss_list = []
    with torch.no_grad():
        for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, val_img_fea) in enumerate(
                            val_dataset_loader):
                        
            val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                          val_pt_fea]
            val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
            val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
            val_img_fea_ten = [i.type(torch.FloatTensor).to(pytorch_device) for i in val_img_fea]
                                              
            predict_labels = my_model(val_pt_fea_ten, val_img_fea_ten, val_grid_ten, val_batch_size)
            loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                    ignore=0) + loss_func(predict_labels.detach(), val_label_tensor)
            predict_labels = torch.argmax(predict_labels, dim=1)
            predict_labels = predict_labels.cpu().detach().numpy()
            for count, i_val_grid in enumerate(val_grid):
                hist_list.append(fast_hist_crop(predict_labels[
                                                    count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                    val_grid[count][:, 2]], val_pt_labs[count],
                                                unique_label))
                inv_labels = np.vectorize(inv_learning_map.__getitem__)(predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]) 
                inv_labels = inv_labels.astype('uint32')
                outputPath = save_dir + str(i_iter_val).zfill(6) + '.label'
                inv_labels.tofile(outputPath)
                print("save " + outputPath)
            val_loss_list.append(loss.detach().cpu().numpy())

    if demo_label_dir != '':                            
        my_model.train()
        iou = per_class_iu(sum(hist_list))
        print('Validation per class iou: ')
        for class_name, class_iou in zip(unique_label_str, iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))                    
        val_miou = np.nanmean(iou) * 100
        del val_vox_label, val_grid, val_pt_fea, val_grid_ten

        print('Current val miou is %.3f' %
              (val_miou))
        print('Current val loss is %.3f' %
              (np.mean(val_loss_list)))

           

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/semantickitti.yaml')
    parser.add_argument('--demo-folder', type=str, default='', help='path to the folder containing demo lidar scans', required=True)
    parser.add_argument('--save-folder', type=str, default='', help='path to save your result', required=True)
    parser.add_argument('--demo-label-folder', type=str, default='', help='path to the folder containing demo labels')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)

