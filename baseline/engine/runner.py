'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import shutil
from sys import api_version
import time
import torch
from tqdm import tqdm
import pytorch_warmup as warmup
import numpy as np
import random
import cv2
import os

from baseline.models.registry import build_net
from .optimizer import build_optimizer
from .scheduler import build_scheduler
from baseline.datasets import build_dataloader, build_dataset
from baseline.utils.metric_utils import calc_measures
from baseline.utils.net_utils import save_model, load_network

# Function for heuristic DA detection using Bayesian Gaussian Kernel
from baseline.heuristic_da_detection import drivable_area_prediction_using_non_dl_lidar

import seaborn as sns
import matplotlib.pyplot as plt

from PIL import Image 

import matplotlib

class Runner(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        self.cfg = cfg
        self.log_dir = cfg.log_dir
        self.epoch = 0

        ### Custom logs ###
        self.batch_bar = None#tqdm(total = 1, desc = 'batch', position = 1)
        self.val_bar = None#tqdm(total = 1, desc = 'val', position = 2)
        self.info_bar = tqdm(total = 0, position = 3, bar_format='{desc}')
        self.val_info_bar = tqdm(total = 0, position = 4, bar_format='{desc}')
        ### Custom logs ###
        
        self.net = build_net(self.cfg)
        
        
        print( "GPU used is GPUS ID : " + str( [ int(i) for i in str(cfg.gpus_ids).split(",")]))
        
        #if len((cfg.gpus_ids).split(",")) <= 1 :
            #self.net.to(torch.device('cuda:' + str(cfg.gpus_ids)))
        
        # self.net.to(torch.device('cuda'))
        #else :
            
        self.net = torch.nn.parallel.DataParallel(
                self.net, device_ids = [ int(i) for i in (cfg.gpus_ids).split(",")]).cuda( int( cfg.gpus_ids))
        if self.cfg.load_from is not None :
            self.resume()
                
        #self.                ()
        self.optimizer = build_optimizer(self.cfg, self.net)
        self.scheduler = build_scheduler(self.cfg, self.optimizer)
        
        self.warmup_scheduler = None
        if self.cfg.optimizer.type == 'SGD':
            self.warmup_scheduler = warmup.LinearWarmup(
                self.optimizer, warmup_period=5000)
                
        self.metric = 0.
        self.val_loader = None

    def resume(self):
        if not self.cfg.load_from and not self.cfg.finetune_from:
            return
        
        print( "Load model frin previous trained model : " + str( self.cfg.load_from))
        load_network(self.net, self.cfg.load_from,
                finetune_from=self.cfg.finetune_from)
        # Start epoch from the save model

        try :
            self.cfg.start_epochs = int( str( str( self.cfg.load_from ).split("/")[-1]).replace( ".pth", "")) + 1
        except :
            print( "Cannot start epoch from last epoch in load file..")
            

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            batch[k] = batch[k].cuda()
        return batch
    
    def write_to_log(self, log, log_file_name):
        f = open(log_file_name, 'a')
        f.write(log)
        f.close()
    
    def train_epoch(self, epoch, train_loader):
        self.net.train()
        end = time.time()
        max_iter = len(train_loader)
        self.batch_bar = tqdm(total = max_iter, desc='batch', position=1)

        max_data = len(train_loader)

        for i, data in enumerate(train_loader):
            if i == max_data - 1:
                continue

            # if self.recorder.step >= self.cfg.total_iter:
            #     break
            date_time = time.time() - end
            # self.recorder.step += 1

            # with torch.autograd.set_detect_anomaly(True):
            data = self.to_cuda(data)#.type( torch.cuda.FloatTensor )

            #print( "Data is : " + str( data ))
            try :
                output = self.net(data)
            except :
                data["pillars"] = data["pillars"].type( torch.cuda.FloatTensor )
                data[ "pillar_indices" ] = data[ "pillar_indices" ].type( torch.cuda.FloatTensor )
                output = self.net(data)
            
            
            if isinstance( output, tuple ) :
                
                #global_attention_visualization = output[1]
                output = output[0]
            
                      
            
            self.optimizer.zero_grad()
            loss = output['loss']

            #print( "Loss is : " + str( loss ))

            try :
                loss = loss.mean()
            except :
                loss = loss
                
            if torch.isfinite(loss):
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if self.warmup_scheduler:
                    self.warmup_scheduler.dampen()
                batch_time = time.time() - end
                end = time.time()

                ### Logging ###
                log_train = f'epoch={epoch}/{self.cfg.epochs}, loss={loss.detach().cpu()}'
                loss_stats = output['loss_stats']
                for k, v in loss_stats.items():
                    log_train += f', {k}={v.detach().cpu()}'

                self.info_bar.set_description_str(log_train)
                self.write_to_log(log_train + '\n', os.path.join(self.log_dir, 'train.txt'))
                ### Logging ###
            else:
                print(f'problem index = {i}')
                self.write_to_log(f'problem index = {i}' + '\n', os.path.join(self.log_dir, 'prob.txt'))

            self.batch_bar.update(1)

    def train(self):
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)

        for epoch in range( self.cfg.start_epochs , self.cfg.epochs):
            self.train_epoch(epoch, train_loader)
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt(epoch)
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt(epoch)
                self.validate(epoch , is_visualized_result= self.cfg.is_visualized_result)

    def validate(self, epoch=None, is_small=False, valid_samples=100 , is_visualized_result = False , is_save_visualization = True , num_visualized_result = 10 , radius_lidar_point = 2 ):
        self.cfg.is_eval_conditional = False
        if is_small:
            self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)
        else:
            #if not self.val_loader:

                #if is_visualized_result == True :

                #    assert LIST_OF_LOG_FOR_VISUALIZATION is not None 
            self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)
        self.net.eval()

        if is_small:
            self.val_bar = tqdm(total = valid_samples, desc='val', position=2)
        else:
            self.val_bar = tqdm(total = len(self.val_loader), desc='val', position=2)

        if self.cfg.is_making_video_result == True :

            height, width, layers = 1280, 3600 , 3 #1200 , 1920 , 3

            fourcc = cv2.VideoWriter_fourcc(*'MP4V')

            name_of_video_road_detection_result = "Road_Detection_Result_Video_" + self.cfg.time_log + "_epoch_+" + str( epoch ) +  ".mp4"

            self.video = cv2.VideoWriter( name_of_video_road_detection_result , fourcc , 2, (width,height))

        list_conf_f1 = []
        list_conf_f1_strict = []
        list_conf_by_cls_f1 = []
        list_conf_by_cls_f1_strict = []

        list_cls_f1 = []
        list_cls_f1_strict = []
        list_cls_w_conf_f1 = []
        list_cls_w_conf_f1_strict = []

        if is_visualized_result == True :
            number_of_total_frame = valid_samples #len( self.val_loader )
            
            list_of_visualized_result = [ frame_number for frame_number in range( number_of_total_frame )] #np.linspace( 0 , number_of_total_frame - 1 , num_visualized_result ).astype( int )

            print( "Number of Total validation dataset is : " + str(number_of_total_frame ) + "and number of visualized result is : " + str( len( list_of_visualized_result )))

            name_of_visualization_folder = "./" + str( self.cfg.experiment_name ) + "/" + "validation_visualization/"

            os.makedirs( name_of_visualization_folder , exist_ok= True )

            list_of_visualized_prediction_image = []

            #list_of_visualized_prediction_image_key = []

        sum_of_time_for_prediction = 0

        for i, data in enumerate(self.val_loader):

            #print( "Process data number : " + str( i ))

            if is_small:
                if i>valid_samples:
                    break

            data = self.to_cuda(data)

            if self.cfg.is_making_video_result == True :

                assert data[ "ground_point_projection_to_image" ] is not None 

            with torch.no_grad():

                start_time_for_3D_DA_detection = time.time()

                output = self.net(data)
                
                if self.cfg.is_visualize_global_attention == True :
                    global_attention_visualization = output[ "Global_Attention_Visualization" ]
                    #print( "Shape of global attention visualization is : " + str( global_attention_visualization.shape ) )
                    global_attention_visualization= torch.squeeze( global_attention_visualization ).permute( 1, 2 , 0 )
                    global_attention_visualization = torch.tensor( [[torch.mean( torch.square(i )) for i in j ] for j in global_attention_visualization] )
                    # Normalize global attention visualization
                    
                    global_attention_visualization_max_correlation = torch.max( global_attention_visualization )
                    global_attention_visualization_min_correlation = torch.min( global_attention_visualization )
                    
                    global_attention_visualization_normalized = ( global_attention_visualization - global_attention_visualization_min_correlation )/( ( global_attention_visualization_max_correlation - global_attention_visualization_min_correlation + 1e-12) )
                    
                    global_attention_visualization_with_color = np.array( [[[255 * i , 0 , 0 ] for i in j ] for j in global_attention_visualization_normalized ] ).astype( np.uint8 )
                    #output = output[0]

                finished_time_for_3D_DA_detection = time.time()

                long_time_for_prediction = finished_time_for_3D_DA_detection - start_time_for_3D_DA_detection

                sum_of_time_for_prediction = sum_of_time_for_prediction + long_time_for_prediction

                lane_maps = output['lane_maps']

                for batch_idx in range(len(output['conf'])):
                    conf_label = lane_maps['conf_label'][batch_idx]
                    cls_label = lane_maps['cls_label'][batch_idx]
                    conf_pred = lane_maps['conf_pred'][batch_idx]
                    conf_by_cls = lane_maps['conf_by_cls'][batch_idx]
                    cls_idx = lane_maps['cls_idx'][batch_idx]
                    conf_cls_idx = lane_maps['conf_cls_idx'][batch_idx]

                    #print( "Shape of Confidence Label is : " + str( conf_label.shape ))
                    #print( "Shape of Confidence prediction is : " + str( conf_pred.shape ))

                    _, _, _, f1 = calc_measures(conf_label, conf_pred, 'conf' , image_size= self.cfg.image_size)
                    _, _, _, f1_strict = calc_measures(conf_label, conf_pred, 'conf', is_wo_offset=True , image_size= self.cfg.image_size)
                    list_conf_f1.append(f1)
                    list_conf_f1_strict.append(f1_strict)

                    _, _, _, f1 = calc_measures(conf_label, conf_by_cls, 'conf' , image_size= self.cfg.image_size)
                    _, _, _, f1_strict = calc_measures(conf_label, conf_by_cls, 'conf', is_wo_offset=True , image_size= self.cfg.image_size)
                    list_conf_by_cls_f1.append(f1)
                    list_conf_by_cls_f1_strict.append(f1_strict)

                    _, _, _, f1 = calc_measures(cls_label, cls_idx, 'cls' , image_size= self.cfg.image_size)
                    _, _, _, f1_strict = calc_measures(cls_label, cls_idx, 'cls', is_wo_offset=True , image_size= self.cfg.image_size)
                    list_cls_f1.append(f1)
                    list_cls_f1_strict.append(f1_strict)

                    _, _, _, f1 = calc_measures(cls_label, conf_cls_idx, 'cls' , image_size= self.cfg.image_size)
                    _, _, _, f1_strict = calc_measures(cls_label, conf_cls_idx, 'cls', is_wo_offset=True , image_size= self.cfg.image_size)
                    list_cls_w_conf_f1.append(f1)
                    list_cls_w_conf_f1_strict.append(f1_strict)

                    if is_visualized_result == True :

                        if i in list_of_visualized_result :

                            #print( "Shape of confidence label is : " + str( conf_label ) + "with average drivable area : " + str( np.mean( conf_label )))

                            height_of_image , width_of_image = conf_label.shape

                            #print( "Heigh of the image is : {} and width : {}".format( height_of_image , width_of_image ))

                            #conf_label = conf_label.reshape( -1 , height_of_image , width_of_image )

                            road_detection_label = np.array( data[ "label" ][batch_idx].cpu() )

                            road_detection_prediction = np.array( lane_maps[ "cls_pred_raw" ][ batch_idx ][0] )

                            # Change image to RGB image with red color as drivable area

                            road_detection_label_with_color = np.array( [[[120, 120 , 120] if i == 1 else [ 255 , 255 , 255 ] for i in j ] for j in road_detection_label] ).astype(np.uint8)

                            #print( "Shape of Road Detection label with color for Autonomous Vehicle road detection is : " + str( road_detection_label_with_color.shape ) )

                            road_detection_prediction_with_color = np.array( [[[( i ) * ( 255 - 120 ) + 120, ( i )* ( 255 - 120 ) + 120 , ( i  )* (255 - 120) + 120] for i in j] for j in road_detection_prediction] ).astype(np.uint8)

                            lidar_points_visualization = data[ "LiDAR_points_visualization" ]

                            # Make lidar points segmentation for point-wise DA detection
                
                            lidar_pts_visualization_this_epoch = np.array( torch.squeeze( lidar_points_visualization ).cpu()) #lidar_pts_visualization( lidar_pts )
                            
                            #print( "Number of voxel with lidar point cloud is : " + str( np.sum()))
                        
                            
                            lidar_point_visualization_with_DA_label_color = lidar_pts_visualization_this_epoch.astype( np.uint8 ) # torch.squeeze( lidar_pts_visualization_this_epoch )#.copy()
                            
                            lidar_point_visualization_with_DA_label_color_BEV_image = Image.fromarray( lidar_point_visualization_with_DA_label_color )#.rotate( 90 )
                            
                            lidar_point_visualization_with_DA_label_color = np.array( lidar_point_visualization_with_DA_label_color_BEV_image  )
                            
                            assert lidar_point_visualization_with_DA_label_color.shape[ : 2 ] == road_detection_label.shape[ : 2 ] , "Shape of lidar point visualization and road detection prediction have to be same that is shape of lidar point visualization is : " + str( lidar_point_visualization_with_DA_label_color.shape ) + " and shape of road detection prediction is : " + str( road_detection_prediction.shape ) 
                            
                            for x_lidar_point_visualization in range(lidar_pts_visualization_this_epoch.shape[0]) :
                            
                                for y_lidar_point_visualization in range( lidar_pts_visualization_this_epoch.shape[1]) :
                                
                                    #if lidar_pts_visualization_this_epoch[ x_lidar_point_visualization ][ y_lidar_point_visualization ][0] == 0  :
                                    
                                    if road_detection_label[ x_lidar_point_visualization ][ y_lidar_point_visualization ] == 1 :
                                    
                                        #print( "Find a DA voxel with raw LiDAR point in the BEV voxel" )
                                        
                                        if lidar_point_visualization_with_DA_label_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ][0] == 0 :
                                        
                                        
                                        
                                            lidar_point_visualization_with_DA_label_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [0,0,255] #lidar_point_visualization_with_DA_label_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ]
                                            
                                            
                                        else :
                                        
                                            lidar_point_visualization_with_DA_label_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = [ 120,120,120 ]
                                        
                                                                        
                                    #else :
                                    
                                    #        lidar_point_visualization_with_DA_label_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ] = lidar_point_visualization_with_DA_label_color[ x_lidar_point_visualization ][ y_lidar_point_visualization ]

                            # Make road detection label with
                            #road_detection_prediction_with_color = np.array( [[[ 120, 120 , 120] if i < 0.9 else [255, 255, 255 ] for i in j] for j in road_detection_prediction] ).astype(np.uint8)

                            # Draw vehicle position in road detection prediction and road detection label

                            #width_of_image

                            if self.cfg.is_using_heuristic_da_detection == True:

                                road_detection_prediction_with_heuristic_da_detection_with_color = torch.squeeze( data["DA_detection_using_gaussian_kernel" ] ).detach().cpu().numpy()

                                road_detection_prediction_with_heuristic_da_detection_with_color = road_detection_prediction_with_heuristic_da_detection_with_color.astype( np.uint8 )

                            try :

                                road_detection_label_with_color = cv2.rectangle(road_detection_label_with_color, pt1= ( int( 1/2 * width_of_image - 5 ), int( 7/12 * height_of_image - 10 ) ) , pt2 = ( int( 1/2 * width_of_image + 5) , int( 7/12 * height_of_image + 10)), color = ( 0 , 0 , 255 ), thickness = -1 ) 

                                road_detection_prediction_with_color = cv2.rectangle(road_detection_prediction_with_color, pt1= ( int( 1/2 * width_of_image -5 ) , int( 7/12 * height_of_image - 10) ) , pt2 = ( int( 1/2 * width_of_image + 5) , int( 7/12 * height_of_image + 10 ) ), color = ( 0 , 0 , 255 ), thickness = -1 )

                                if self.cfg.is_using_heuristic_da_detection == True :
                                    
                                     road_detection_prediction_with_heuristic_da_detection_with_color = cv2.rectangle(road_detection_prediction_with_heuristic_da_detection_with_color, pt1= ( int( 1/2 * width_of_image -5 ) , int( 7/12 * height_of_image - 10) ) , pt2 = ( int( 1/2 * width_of_image + 5) , int( 7/12 * height_of_image + 10 ) ), color = ( 0 , 0 , 255 ), thickness = -1 )

                            except Exception as e :

                                print( "Cannot draw Autonomous Vehicle in Drivable Area BEV Map  because : " + str( e ))

                            #print( "Shape of road detection label is : " + str( road_detection_label_with_color.shape ) + " and shape of road detection prediction is : " + str( road_detection_prediction_with_color.shape ))

                            #lidar_points_data = torch.squeeze( data[ "lidar_data" ] )

                            #lidar_points_data = torch.concatenate( [ lidar_points_data[ : , 0 ]* ( 1 / self.cfg.list_grid_xy[0] ) , lidar_points_data[ : , 1 ]* ( 1/ self.cfg.list_grid_xy[1] ) , lidar_points_data[ : ,2 : ]], dim= 1 )

                            #print( "Shape of lidar points is : " + str( lidar_points_data.shape ))

                            """

                            for i, lidar_point in enumerate( torch.squeeze( lidar_points_data )) :

                                #print( "Lidar point is : " + str( lidar_point ))

                                if (( lidar_point[0] >= -width_of_image/2 ) and
                                    ( lidar_point[0] <= width_of_image/2 ) and
                                    ( lidar_point[1] >= -height_of_image*5/12) and 
                                    ( lidar_point[1] <= height_of_image*7/12 )) : 

                                    #print( "Writes lidar point number : " + str( i )) 

                                    road_detection_label_with_color = cv2.circle(road_detection_label_with_color,  ( int( (lidar_point[0] + width_of_image/2)), int( (lidar_point[1] + height_of_image*7/12))), radius_lidar_point, ( 0 , 0 , int(100 + ( -lidar_point[2] - 10 )* 3 )))

                                    road_detection_prediction_with_color = cv2.circle(road_detection_prediction_with_color, ( int( (lidar_point[0] + width_of_image/2)), int( (lidar_point[1] + height_of_image*7/12))), radius_lidar_point, ( 0 , 0 , int( 100 + ( -lidar_point[2] - 10 )* 3 )))
                            """

                            #resize, first image
                            image1 = Image.fromarray( road_detection_label_with_color ).resize((1000 , 1200))
                            image2 = Image.fromarray( road_detection_prediction_with_color ).resize( ( 1000,1200 ))
                            #print( "Size of prediction image is : " + str( np.array( image2 ).shape ))
                            image1_size = image1.size
                            image2_size = image2.size
                            new_image = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                            new_image.paste(image1,(0,0))
                            new_image.paste(image2,( image1_size[0], 0 ))

                            if self.cfg.is_making_video_result == True :
                                print( "Making drivable area detection result video for LiDAR point cloud : " + str( i ))

                                lidar_point_visualization_with_DA_label_color = cv2.rectangle(lidar_point_visualization_with_DA_label_color, pt1= ( int( 1/2 * width_of_image -5 ) , int( 7/12 * height_of_image - 10) ) , pt2 = ( int( 1/2 * width_of_image + 5) , int( 7/12 * height_of_image + 10 ) ), color = ( 0 , 0 , 120 ), thickness = -1 ) 

                                image1 = Image.fromarray( lidar_point_visualization_with_DA_label_color ).resize((1000 , 1200))

                                if self.cfg.is_using_heuristic_da_detection == True :
                                    # Then visualize DA detection using heuristic DA detection
                                    image2 = Image.fromarray( road_detection_prediction_with_heuristic_da_detection_with_color ).resize(( 1000,1200) )
                                    image1_size = image1.size
                                    image2_size = image2.size
                                    new_image_with_Heuristic_DA_detection = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                                    new_image_with_Heuristic_DA_detection.paste(image1,(0,0))
                                    new_image_with_Heuristic_DA_detection.paste(image2,( image1_size[0], 0 ))
                                    new_image_with_Heuristic_DA_detection = np.array( new_image_with_Heuristic_DA_detection )

                                    image1 = Image.fromarray( new_image_with_Heuristic_DA_detection )

                                image2 = Image.fromarray( road_detection_prediction_with_color ).resize( ( 1000,1200 ))
                                #print( "Size of prediction image is : " + str( np.array( image2 ).shape ))
                                image1_size = image1.size
                                image2_size = image2.size
                                new_image_with_LiDAR_Points_Visualization = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                                new_image_with_LiDAR_Points_Visualization.paste(image1,(0,0))
                                new_image_with_LiDAR_Points_Visualization.paste(image2,( image1_size[0], 0 ))
                                new_image_with_LiDAR_Points_Visualization = np.array( new_image_with_LiDAR_Points_Visualization )



                                image_ground_point_projected_to_image = np.array( torch.squeeze( data[ "ground_point_projection_to_image" ] ).cpu() ).astype( np.uint8 )

                                #resize, first image
                                image1 = Image.fromarray( image_ground_point_projected_to_image ).resize((480 , 1280))
                                image2 = Image.fromarray( new_image_with_LiDAR_Points_Visualization ).resize( ( 3150,1280 ))
                                #print( "Size of prediction image is : " + str( np.array( image2 ).shape ))
                                image1_size = image1.size
                                image2_size = image2.size
                                new_image_ground_point_projection_to_image_for_video_result = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                                new_image_ground_point_projection_to_image_for_video_result.paste(image1,(0,0))
                                new_image_ground_point_projection_to_image_for_video_result.paste(image2,( image1_size[0], 0 ))
                                
                                new_image_ground_point_projection_to_image_for_video_result_image = cv2.cvtColor( np.array( new_image_ground_point_projection_to_image_for_video_result.resize( ( width , height )) ), cv2.COLOR_BGR2RGB )

                                print( "Size of New Image Ground Point Project5ion to Image for video result is : " + str( np.array( new_image_ground_point_projection_to_image_for_video_result ).shape ))
                                
                                try :
                                    self.video.write( new_image_ground_point_projection_to_image_for_video_result_image )
                                    
                                    print( "Finish adding Drivable Area detection to Drivable Area Detection video prediction ")
                                except Exception as e :
                                    print( "Couldnt make Drivable Area prediction video because : " + str( e ) )

                            if self.cfg.is_visualize_global_attention == True :
                            
                                #resize, first image
                                image1 = new_image #Image.fromarray( road_detection_label_with_color ).resize((1000 , 1200))
                                image2 = Image.fromarray( np.array( global_attention_visualization_with_color ) ).resize( ( 1000,1200 ))
                                #print( "Size of prediction image is : " + str( np.array( image2 ).shape ))
                                image1_size = image1.size
                                image2_size = image2.size
                                new_image = Image.new('RGB',(image1_size[0] + image2_size[0], image1_size[1]), (250,250,250))
                                new_image.paste(image1,(0,0))
                                new_image.paste(image2,( image1_size[0], 0 ))

                            list_of_visualized_prediction_image.append( np.array( new_image.resize( size=(2000 , 1200 ) , resample= Image.NEAREST  )  ) )

                            #list_of_visualized_prediction_image_key.append( data[ "key" ][ batch_idx ].cpu() )

            self.val_bar.update( 1 )


                
        print( "Evaluation time for 3D DA Prediction using Global Feature Correlator is : " + str( sum_of_time_for_prediction ))
        
        if self.cfg.is_making_video_result == True :

            print( "Finish making Drivable Area Detection video for Argoverse Dataset")

            cv2.destroyAllWindows()
            self.video.release() 

        # Save visualized validation to folder
                            
        if ( is_visualized_result == True ) :

            # Visualizing some Lane Detection dataset
    
            sns.set_theme()

            f, axarr = plt.subplots( len( list_of_visualized_prediction_image ), figsize = ( 20 , 60 ))
            
            plt.title( "Visualization of Road Detection Epoch : {}".format( epoch ))
            plt.axis('off')

            for i in range( len( list_of_visualized_prediction_image )  ):

                matplotlib.image.imsave( name_of_visualization_folder + 'Road_Detection_Prediction_{}_Epoch_{}.png'.format( i , epoch ), list_of_visualized_prediction_image[i] )

                axarr[ i ].imshow(  list_of_visualized_prediction_image[i] )
                axarr[ i ].set_title( "Lane Image Data Label and Prediction Data : " + str( list_of_visualized_result[ i ]) )
                axarr[ i ].set_axis_off()
                

            f.tight_layout()
            #plt.show()

            if is_save_visualization == True : 
                f.savefig( name_of_visualization_folder + "visualization_epoch_" + str( epoch ) + ".png" )
                #plt.savefig( name_of_visualization_folder + "visualization_epoch_" + str( epoch ) + "image_saved.png" )
                print( "Save image visualization to : " + str( name_of_visualization_folder + "visualization_epoch_" + str( epoch ) + ".png"   ))

            return plt #plt
                
    
    def save_ckpt(self, epoch, is_best=False):
        save_model(self.net, self.optimizer, self.scheduler, epoch, self.cfg.log_dir, is_best=is_best)

    ### Small dataset ###
    def train_epoch_small(self, epoch, train_loader, maximum_batch = 200):
        self.net.train()
        self.batch_bar = tqdm(total = maximum_batch, desc='batch', position=1)

        for i, data in enumerate(train_loader):
            if i > maximum_batch:
                break
            
            data = self.to_cuda(data)
            output = self.net(data)
            self.optimizer.zero_grad()
            loss = output['loss']
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            if self.warmup_scheduler:
                self.warmup_scheduler.dampen()

            ### Logging ###
            log_train = f'epoch={epoch}/{self.cfg.epochs}, loss={loss.detach().cpu()}'
            self.info_bar.set_description_str(log_train)
            self.write_to_log(log_train + '\n', os.path.join(self.log_dir, 'train.txt'))
            ### Logging ###

            self.batch_bar.update(1)

    def train_small(self, train_batch = 200, valid_samples = 80):
        train_loader = build_dataloader(self.cfg.dataset.train, self.cfg, is_train=True)

        for epoch in range(self.cfg.epochs):
            self.train_epoch_small(epoch, train_loader, train_batch)
            if (epoch + 1) % self.cfg.save_ep == 0 or epoch == self.cfg.epochs - 1:
                self.save_ckpt(epoch)
            if (epoch + 1) % self.cfg.eval_ep == 0 or epoch == self.cfg.epochs - 1:
                self.validate(epoch, is_small=True, valid_samples=valid_samples)
    
    def load_ckpt(self, path_ckpt):
        trained_model = torch.load(path_ckpt)

        #remove_prefix = 'module.'
        #trained_model["net"] = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in trained_model["net"].items()}
        
        #print( "List of trained model keys : " + str( trained_model["net"].keys()))

        self.net.load_state_dict(trained_model['net'], strict= False )#strict=True)

    def process_one_sample(self, sample, is_calc_f1=False, is_get_features=False, is_measure_ms=False, is_get_attention_score=False):
        self.net.eval()
        with torch.no_grad():
            if is_measure_ms:
                t_start = torch.cuda.Event(enable_timing=True)
                t_end = torch.cuda.Event(enable_timing=True)
                t_start.record()
            
            if is_get_features:
                output = self.net(sample, is_get_features=True)
            elif is_get_attention_score:
                output = self.net(sample)
                print(getattr(self.net, 'module'))
            else:
                output = self.net(sample)

            if is_measure_ms:
                t_end.record()
                torch.cuda.synchronize()
                t_infer = t_start.elapsed_time(t_end)
                print(f'* inference time = {t_infer}')
            
            lane_maps = output['lane_maps']

            conf_label = lane_maps['conf_label'][0]
            cls_label = lane_maps['cls_label'][0]
            conf_pred = lane_maps['conf_pred'][0]
            conf_by_cls = lane_maps['conf_by_cls'][0]

            cls_idx = lane_maps['cls_idx'][0]
            conf_cls_idx = lane_maps['conf_cls_idx'][0]

            conf_pred_raw = lane_maps['conf_pred_raw'][0]

            rgb_cls_label = lane_maps['rgb_cls_label'][0]
            rgb_cls_idx = lane_maps['rgb_cls_idx'][0]
            rgb_conf_cls_idx = lane_maps['rgb_conf_cls_idx'][0]
            
            new_output = dict()

            if is_get_features:
                new_output.update({'features': output['features']})

            if is_calc_f1:
                acc_0, pre_0, rec_0, f1_0 = calc_measures(conf_label, conf_pred, 'conf', image_size= self.cfg.image_size)
                acc_1, pre_1, rec_1, f1_1 = calc_measures(conf_label, conf_by_cls, 'conf' , image_size= self.cfg.image_size)

                acc_2, pre_2, rec_2, f1_2 = calc_measures(cls_label, cls_idx, 'cls', image_size= self.cfg.image_size)
                acc_3, pre_3, rec_3, f1_3 = calc_measures(cls_label, conf_cls_idx, 'cls' , image_size= self.cfg.image_size)

                new_output['accuracy'] = np.array([acc_0, acc_1, acc_2, acc_3])
                new_output['precision'] = np.array([pre_0, pre_1, pre_2, pre_3])
                new_output['recall'] = np.array([rec_0, rec_1, rec_2, rec_3])
                new_output['f1'] = np.array([f1_0, f1_1, f1_2, f1_3])

            new_output['rgb_cls_label'] = rgb_cls_label
            new_output['rgb_cls_idx'] = rgb_cls_idx
            new_output['rgb_conf_cls_idx'] = rgb_conf_cls_idx
            new_output['conf_raw'] = conf_pred_raw

            new_output['conf_label'] = conf_label
            new_output['cls_label'] = cls_label
            
            new_output['conf_pred'] = conf_pred
            new_output['conf_by_cls'] = conf_by_cls

            new_output['cls_idx'] = cls_idx
            new_output['conf_cls_idx'] = conf_cls_idx

        return new_output

    def process_one_sample_for_2stage(self, sample):
        self.net.eval()
        with torch.no_grad():
            output = self.net(sample)
            
            lane_maps = output['lane_maps']
            
            conf_label = lane_maps['conf_label'][0]
            cls_label = lane_maps['cls_label'][0]
            conf_pred = lane_maps['conf_pred'][0]
            conf_by_cls = lane_maps['conf_by_cls'][0]

            cls_idx = lane_maps['cls_idx'][0]
            conf_cls_idx = lane_maps['conf_cls_idx'][0]

            conf_pred_raw = lane_maps['conf_pred_raw'][0]

            rgb_cls_label = lane_maps['rgb_cls_label'][0]
            rgb_cls_idx = lane_maps['rgb_cls_idx'][0]
            rgb_conf_cls_idx = lane_maps['rgb_conf_cls_idx'][0]

            conf_pred_1 = lane_maps['conf_pred_1'][0]
            rgb_conf_cls_idx_1 = lane_maps['rgb_conf_cls_idx_1'][0]
            
            new_output = dict()

            conf_pred = lane_maps['conf_pred_1'][0]
            conf_by_cls = lane_maps['conf_by_cls_1'][0]

            acc_0, pre_0, rec_0, f1_0 = calc_measures(conf_label, conf_pred, 'conf' , image_size= self.cfg.image_size)
            acc_1, pre_1, rec_1, f1_1 = calc_measures(conf_label, conf_by_cls, 'conf' , image_size= self.cfg.image_size)

            acc_2, pre_2, rec_2, f1_2 = calc_measures(cls_label, cls_idx, 'cls' , image_size= self.cfg.image_size)
            acc_3, pre_3, rec_3, f1_3 = calc_measures(cls_label, conf_cls_idx, 'cls' , image_size= self.cfg.image_size)

            new_output['accuracy'] = np.array([acc_0, acc_1, acc_2, acc_3])
            new_output['precision'] = np.array([pre_0, pre_1, pre_2, pre_3])
            new_output['recall'] = np.array([rec_0, rec_1, rec_2, rec_3])
            new_output['f1'] = np.array([f1_0, f1_1, f1_2, f1_3])

            new_output['rgb_cls_label'] = rgb_cls_label
            new_output['rgb_cls_idx'] = rgb_cls_idx
            new_output['rgb_conf_cls_idx'] = rgb_conf_cls_idx
            new_output['conf_raw'] = conf_pred_raw

            new_output['conf_label'] = conf_label
            new_output['cls_label'] = cls_label
            
            new_output['conf_pred'] = conf_pred
            new_output['conf_by_cls'] = conf_by_cls

            new_output['cls_idx'] = cls_idx
            new_output['conf_cls_idx'] = conf_cls_idx

            new_output['conf_pred_1'] = conf_pred_1
            new_output['rgb_conf_cls_idx_1'] = rgb_conf_cls_idx_1

            new_output['conf_cls_idx_1'] = lane_maps['conf_cls_idx_1'][0]

            acc_0, pre_0, rec_0, f1_0 = calc_measures(conf_label, conf_pred, 'conf' , image_size= self.cfg.image_size)
            acc_1, pre_1, rec_1, f1_1 = calc_measures(conf_label, conf_by_cls, 'conf' , image_size= self.cfg.image_size)

        return new_output

    def infer_lane(self, path_ckpt=None, mode_imshow=None, is_calc_f1=False):
        self.val_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=True)
        
        if path_ckpt:
            trained_model = torch.load(path_ckpt)
            self.net.load_state_dict(trained_model['net'], strict=True)

        self.net.eval()
        for i, data in enumerate(self.val_loader):
            data = self.to_cuda(data)
            with torch.no_grad():
                output = self.net(data)
                # print(output)

                lane_maps = output['lane_maps']
                # print(lane_maps.keys())

                for batch_idx in range(len(output['conf'])):
                    conf_label = lane_maps['conf_label'][batch_idx]
                    cls_label = lane_maps['cls_label'][batch_idx]
                    conf_pred = lane_maps['conf_pred'][batch_idx]
                    conf_by_cls = lane_maps['conf_by_cls'][batch_idx]
                    cls_idx = lane_maps['cls_idx'][batch_idx]
                    conf_cls_idx = lane_maps['conf_cls_idx'][batch_idx]
                    
                    if is_calc_f1:
                        list_conf = [conf_pred, conf_by_cls, conf_label]
                        desc_list_conf = ['conf:\t\t', 'conf_by_cls:\t', 'checking_conf:\t']
                        list_cls = [cls_idx, conf_cls_idx, cls_label]
                        desc_list_cls = ['cls_idx:\t', 'conf_cls_idx:\t', 'checking_cls:\t']

                        print('\n### Confidence ###')
                        for temp_conf, temp_desc in zip(list_conf, desc_list_conf):
                            acc, pre, rec, f1 = calc_measures(conf_label, temp_conf, 'conf' , image_size= self.cfg.image_size)
                            acc_s, pre_s, rec_s, f1_s = calc_measures(conf_label, temp_conf, 'conf', is_wo_offset=True , image_size= self.cfg.image_size)
                            print(f'{temp_desc}acc={acc},acc_strict={acc_s}')
                            print(f'{temp_desc}pre={pre},pre_strict={pre_s}')
                            print(f'{temp_desc}rec={rec},rec_strict={rec_s}')
                            print(f'{temp_desc}f1={f1},f1_strict={f1_s}')

                        print('### Classification ###')
                        for temp_cls, temp_desc in zip(list_cls, desc_list_cls):
                            acc, pre, rec, f1 = calc_measures(cls_label, temp_cls, 'cls' , image_size= self.cfg.image_size)
                            acc_s, pre_s, rec_s, f1_s = calc_measures(cls_label, temp_cls, 'cls', is_wo_offset=True , image_size= self.cfg.image_size)
                            print(f'{temp_desc}acc={acc},acc_strict={acc_s}')
                            print(f'{temp_desc}pre={pre},pre_strict={pre_s}')
                            print(f'{temp_desc}rec={rec},rec_strict={rec_s}')
                            print(f'{temp_desc}f1={f1},f1_strict={f1_s}')
                    
                    conf_pred_raw = lane_maps['conf_pred_raw'][batch_idx]

                    if mode_imshow == 'cls' or mode_imshow == 'all':
                        cls_pred_raw = lane_maps['cls_pred_raw'][batch_idx]
                        for j in range(7):
                            cv2.imshow(f'cls {j}', cls_pred_raw[j])
                    
                    if mode_imshow == 'conf' or mode_imshow == 'all':
                        cv2.imshow('conf_raw', conf_pred_raw)
                        cv2.imshow('conf', conf_pred.astype(np.uint8)*255)
                        cv2.imshow('conf_by_cls', conf_by_cls.astype(np.uint8)*255)
                        cv2.imshow('label', conf_label.astype(np.uint8)*255)

                    if mode_imshow == 'rgb' or mode_imshow == 'all':
                        rgb_cls_label = lane_maps['rgb_cls_label'][batch_idx]
                        rgb_cls_idx = lane_maps['rgb_cls_idx'][batch_idx]
                        rgb_conf_cls_idx = lane_maps['rgb_conf_cls_idx'][batch_idx]
                        cv2.imshow('rgb_cls_label', rgb_cls_label)
                        cv2.imshow('rgb_cls_idx', rgb_cls_idx)
                        cv2.imshow('rgb_conf_cls_idx', rgb_conf_cls_idx)

                    if not mode_imshow == None:
                        cv2.waitKey(0)
    
    def eval_conditional(self, save_path = None):
        self.cfg.batch_size = 1
        self.cfg.is_eval_conditional = True
        self.test_dataset = build_dataset(self.cfg.dataset.test, self.cfg)
        self.test_loader = build_dataloader(self.cfg.dataset.test, self.cfg, is_train=False)

        list_conf_f1 = []

        list_cls_f1 = []
        list_runtime = []
        error = 0
        cumsum = 0
        cnt = 0
        self.net.eval()

        for idx_batch, data in enumerate(tqdm(self.test_loader)):
            condition = np.array(data['meta']['description'], dtype = 'object')
            with torch.no_grad():
                output = self.net(data)
                lane_maps = output['lane_maps']
                for batch_idx in range(len(output['conf'])):
                    conf_label = lane_maps['conf_label'][batch_idx]
                    cls_label = lane_maps['cls_label'][batch_idx]
                    conf_pred = lane_maps['conf_pred'][batch_idx]
                    cls_idx = lane_maps['cls_idx'][batch_idx]

                    _, _, _, f1 = calc_measures(conf_label, conf_pred, 'conf' , image_size= self.cfg.image_size)
                    list_conf_f1.append([f1, condition])
                    cumsum += f1
                    cnt+=1
                    list_cls_f1.append([0, condition])
        print('Total Error: ', error)
        list_runtime = np.array(list_runtime)

        # Conditional Metrics
        conditions = ['daylight', 'night', 'urban', 'highway', 'lightcurve', 'curve', 'merging', 'occ0', 'occ1', 'occ2', 'occ3', 'occ4', 'occ5', 'occ6']
        
        list_cls_f1 = np.array(list_cls_f1, dtype = 'object')
        list_conf_f1 = np.array(list_conf_f1, dtype = 'object')     

        cond_dic_conf = dict()
        cond_dic_cls = dict()
        for condition in conditions:
            cond_dic_conf[condition] = []
            cond_dic_cls[condition] = []

        cond_dic_conf['normal'] = []
        cond_dic_cls['normal'] = []

        for i in range(len(list_conf_f1)):
            for condition in conditions:
                if condition in list_conf_f1[i, 1]:
                    cond_dic_conf[condition].append(list_conf_f1[i, 0])
                    cond_dic_cls[condition].append(list_cls_f1[i, 0])

        for i in range(len(list_conf_f1)):
            if not(('merging' in list_conf_f1[i,1]) or ('lightcurve' in list_conf_f1[i,1]) or ('curve' in list_conf_f1[i,1])):
                cond_dic_conf['normal'].append(list_conf_f1[i, 0])
                cond_dic_cls['normal'].append(list_cls_f1[i, 0])
        
        ### Logging ###
        conf_f1 = np.round(np.mean(list_conf_f1[:, 0])*100, 3) 
        cls_f1 =  np.round(np.mean(list_cls_f1[:, 0]) *100, 3)

        res_dic_conf_f1 = dict()
        res_dic_cls_f1 = dict()

        log_val = f'\noverall: {conf_f1}, {cls_f1} '
        for condition in conditions:
            if(len(cond_dic_conf[condition]) > 0):
                res_dic_conf_f1[condition] = np.round(np.mean(cond_dic_conf[condition])*100, 3)
                res_dic_cls_f1[condition] = np.round(np.mean(cond_dic_cls[condition])*100, 3)
            else:       
                res_dic_conf_f1[condition] = -999
                res_dic_cls_f1[condition] = -999

            log_val = log_val + condition + ' ' + str(res_dic_conf_f1[condition]) + ', ' + str(res_dic_cls_f1[condition]) + ' '

        # Normal, Occ456
        res_dic_conf_f1['normal'] = np.round(np.mean(cond_dic_conf['normal']*100),3)
        res_dic_cls_f1['normal'] = np.round(np.mean(cond_dic_cls['normal']*100), 3)
        res_dic_conf_f1['occ456'] = (res_dic_conf_f1['occ4']*128 + res_dic_conf_f1['occ5']*41 + res_dic_conf_f1['occ6']*3)/(128+41+3)
        res_dic_cls_f1['occ456'] = (res_dic_cls_f1['occ4']*128 + res_dic_cls_f1['occ5']*41 + res_dic_cls_f1['occ6']*3)/(128+41+3)

        log_val = log_val + 'Normal ' + str(res_dic_conf_f1['normal']) + ', ' + str(res_dic_cls_f1['normal']) + ' '
        log_val = log_val + 'Occ456 ' + str(res_dic_conf_f1['occ456']) + ', ' + str(res_dic_cls_f1['occ456'])

        print(log_val)
        if(save_path == None):
            self.write_to_log(log_val + '\n', os.path.join(self.log_dir, 'val.txt'))
        else:
            self.write_to_log(log_val + '\n', save_path)

        self.val_info_bar.set_description_str(log_val)
        ### Logging ###
