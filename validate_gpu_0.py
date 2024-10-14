'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import os
GPUS_EN = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = GPUS_EN
import torch.backends.cudnn as cudnn
import time
import cv2
import open3d as o3d
import pickle

time_now = time.localtime()
time_log = '%04d-%02d-%02d-%02d-%02d-%02d' % (time_now.tm_year, time_now.tm_mon, time_now.tm_mday, time_now.tm_hour, time_now.tm_min, time_now.tm_sec)

from baseline.utils.config import Config
from baseline.engine.runner import Runner

from baseline.utils.vis_utils import *
import configs.config_vis as cnf

def main():
    ### Set here ###
    path_config = "./configs/Grid-DATrNet_using_Global_Attention.py"
    path_ckpt = "/media/ave/sub4/Ofel/First-3D-Road-Detection-for-Autonomous-Vehicles/logs/2024-04-19-20-35-27/ckpt/9.pth"#"/media/ave/sub4/Ofel/First-3D-Road-Detection-for-Autonomous-Vehicles/logs/2024-04-23-18-24-11/ckpt/4.pth" #'./configs/Proj28_GFC-T3_RowRef_82_73.pth'
    ### Set here ###

    ### Settings ###a
    cudnn.benchmark = True
    
    #cfg.gpus_ids= GPUS_EN
    cfg, runner = load_config_and_runner(path_config, GPUS_EN)
    cfg.gpus = len(GPUS_EN.split(','))
    cfg.work_dirs = cfg.log_dir + '/' + cfg.dataset.train.type

    cfg.time_log = time_log 
    cfg.experiment_name = "{}_{}".format( cfg.experiment_name , cfg.time_log ) 
    
    print(f'* Config: [{path_config}] is loaded')
    runner.load_ckpt(path_ckpt)
    print(f'* ckpt: [{path_ckpt}] is loaded')
    ### Settings ###
	
    runner.validate( is_visualized_result = cfg.is_visualized_result , is_save_visualization = cfg.is_save_visualization )
    runner.eval_conditional()

if __name__ == '__main__':
    main()
