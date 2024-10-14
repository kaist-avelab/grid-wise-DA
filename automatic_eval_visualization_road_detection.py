import os

import torch.backends.cudnn as cudnn
import time
#import cv2
#import open3d as o3d
#import pickle

import os
GPUS_EN = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPUS_EN

time_now = time.localtime()
time_log = '%04d-%02d-%02d-%02d-%02d-%02d' % (time_now.tm_year, time_now.tm_mon, time_now.tm_mday, time_now.tm_hour, time_now.tm_min, time_now.tm_sec)

from baseline.utils.config import Config
from baseline.engine.runner import Runner

from baseline.utils.vis_utils import *
import configs.config_vis as cnf

def main() :

    NAME_OF_FOLDER_CHECKPOINT = "/home/ofel04/K-Lane/logs/2024-02-28-14-51-39/ckpt/"

    path_config = "/home/ofel04/K-Lane/configs/Proj28_GFC-T3_RowRef_82_73_for_Argoverse_Dataset.py" #'./configs/Proj28_GFC-T3_RowRef_82_73.py'

    range_of_eval = 5

    NAME_OF_FOLDER_VISUALIZATION_RESULT = "./evaluation_visualization/"

    maximum_epoch_of_road_detection = max(int( i.replace(".pth", "")) for i in os.listdir( NAME_OF_FOLDER_CHECKPOINT ) if "best" not in str( i ) )

    os.makedirs( NAME_OF_FOLDER_VISUALIZATION_RESULT , exist_ok= True )

    for epoch_eval in range( 0 , maximum_epoch_of_road_detection , range_of_eval ) :

            ### Settings ###
            cudnn.benchmark = True
            cfg, runner = load_config_and_runner(path_config, GPUS_EN)
            cfg.work_dirs = cfg.log_dir + '/' + cfg.dataset.train.type
            cfg.gpus = len(GPUS_EN.split(','))
            print(f'* Config: [{path_config}] is loaded')

            #print( "Config file is : " + str( cfg ))

            path_ckpt = NAME_OF_FOLDER_CHECKPOINT + str( epoch_eval ) + ".pth"
            runner.load_ckpt(path_ckpt)
            print(f'* ckpt: [{path_ckpt}] is loaded')

            print( "Evaluating road detection epoch : " + str( epoch_eval ))

            validate_visualization = runner.validate( epoch = epoch_eval , is_small= True , valid_samples= 100, is_visualized_result= True , is_save_visualization= False )

            validate_visualization.savefig( NAME_OF_FOLDER_VISUALIZATION_RESULT + "evaluation_visualization_" + str( epoch_eval ) + ".png" )

            print( "Save image visualization to : " + NAME_OF_FOLDER_VISUALIZATION_RESULT + "evaluation_visualization_" + str( epoch_eval ) + ".png"  )

            print( "----------------------------------------------------------------")


if __name__ == '__main__':
    main()

