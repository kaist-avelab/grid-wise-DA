'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import os
GPUS_EN = "0" #"2" #'0'
#os.environ["CUDA_VISIBLE_DEVICES"] = GPUS_EN
import torch
import torch.backends.cudnn as cudnn
import time
import shutil

time_now = time.localtime()
time_log = '%04d-%02d-%02d-%02d-%02d-%02d' % (time_now.tm_year, time_now.tm_mon, time_now.tm_mday, time_now.tm_hour, time_now.tm_min, time_now.tm_sec)

from baseline.utils.config import Config
from baseline.engine.runner import Runner

LIST_OF_LOG_FOR_VISUALIZATION = [ "e17eed4f-3ffd-3532-ab89-41a3f24cf226" ,
                                 "b3def699-884b-3c9e-87e1-1ab76c618e0b" ,
                                 "af706af1-a226-3f6f-8d65-b1f4b9457c48",
                                 "11953248-1195-1195-1195-511954366464",
                                 "64c12551-adb9-36e3-a0c1-e43a0e9f3845"
                                 ]

def main():
    path_config = './configs/Grid-DATrNet_using_Global_Attention.py' #'./configs/Proj28_GFC-T3_RowRef_82_73_for_Argoverse_Dataset _MLP_Mixer.py' #'./configs/Proj28_GFC-T3_RowRef_82_73_for_Argoverse_Dataset.py'
    path_split = path_config.split('/')
    cfg = Config.fromfile(path_config)
    cfg.log_dir = cfg.log_dir + '/' + time_log
    cfg.time_log = time_log
    cfg.experiment_name = "{}_{}".format( cfg.experiment_name , cfg.time_log ) 
    

    os.makedirs(cfg.log_dir, exist_ok=True)
    shutil.copyfile(path_config, cfg.log_dir + '/' + path_split[-2]
                        + '_' + path_split[-1].split('.')[0] + '.txt')
    cfg.work_dirs = cfg.log_dir + '/' + cfg.dataset.train.type
    cfg.gpus = len(GPUS_EN.split(','))
    cfg.gpus_ids= GPUS_EN

    print( "Number of GPU used : " + str( cfg.gpus ))

    cfg.LIST_OF_LOG_FOR_VISUALIZATION = LIST_OF_LOG_FOR_VISUALIZATION

    cudnn.benchmark = True
    
    runner = Runner(cfg)

    runner.train()
    # runner.train_small(train_batch=2, valid_samples=40)

if __name__ == '__main__':
    main()
    
