'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
import torch
import torch.nn as nn
import numpy as np

from baseline.models.registry import HEADS

from torch.utils.tensorboard import SummaryWriter

import os
from datetime import datetime

@HEADS.register_module
class GridSeg(nn.Module):
    
    def __init__(self,
                num_1=1024,
                num_2=2048,
                num_classes=7,
                cfg=None,
                focal_loss_alpha = None,
                focal_loss_gamma = None,
                tensorboard_dir = ".",
                image_size = None):
        super(GridSeg, self).__init__()
        self.cfg=cfg
        
        self.is_visualize_global_attention = self.cfg.is_visualize_global_attention
        
        self.act_sigmoid = nn.Sigmoid()

        self.conf_predictor = nn.Sequential(
            nn.Conv2d(num_1, num_2, 1),
            nn.Conv2d(num_2, 1, 1)
        )

        self.class_predictor = nn.Sequential(
            nn.Conv2d(num_1, num_2, 1),
            nn.Conv2d(num_2, num_classes, 1)
        )

        self.num_classes = num_classes

        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

        self.tensorboard_dir = tensorboard_dir + "/tensorboard_log_{}/".format( str( datetime.now() ))

        os.makedirs( self.tensorboard_dir , exist_ok= True)

        self.tensorboard_writer = SummaryWriter( log_dir= self.tensorboard_dir )

        self.step = 0

        self.image_size = image_size



    def forward(self, x):
        conf_output = self.act_sigmoid(self.conf_predictor(x))
        class_output = self.class_predictor(x)

        #print( "Shape of the Confidence output is : {} and Class Output is : {}".format( conf_output.shape , class_output.shape ))

        if self.image_size :
            # Upsampling the output to make same with Road Detection map labels
            if conf_output.shape[ 2 : 4] != self.image_size :

                conf_output = nn.Upsample( self.image_size , mode= "nearest")( conf_output )

            if class_output.shape[ 2 : 4] != self.image_size :

                class_output = nn.Upsample( self.image_size , mode="nearest" )( class_output )

        out = torch.cat((class_output, conf_output), 1)
        
        
        if self.is_visualize_global_attention == False :
            return out
        else :
            return tuple((out,x))        
    """
    def label_formatting(self, raw_label):
        # Output image: top-left of the image is farthest-left
        num_of_labels = len(raw_label)
        label_tensor = np.zeros((num_of_labels, 2, 144, 144), dtype = np.longlong)

        for k in range(num_of_labels):
            label_temp = np.zeros((144,144,2), dtype = np.longlong)
            label_data = raw_label[k]

            for i in range(144):
                for j in range(144):

                    y_idx = 144 - i - 1
                    x_idx = 144 - j - 1
                    # y_idx = i
                    # x_idx = j

                    line_num = int(label_data[i][j])
                    if line_num == 255:
                        label_temp[y_idx][x_idx][1] = 0
                        # classification
                        label_temp[y_idx][x_idx][0] = 6
                    else: # 클래스
                        # confidence
                        label_temp[y_idx][x_idx][1] = 1
                        # classification
                        label_temp[y_idx][x_idx][0] = line_num

            label_tensor[k,:,:,:] = np.transpose(label_temp, (2, 0, 1))

        return(torch.tensor(label_tensor))
    """
    
    def label_formatting(self, raw_label, is_get_label_as_tensor = True , new_image_size = None):
        # Output image: top-left of the image is farthest-left
        num_of_labels = len(raw_label)
        if new_image_size :
            label_tensor = np.zeros((num_of_labels, 2, new_image_size[0], new_image_size[1] ), dtype = np.longlong)
        else : 
            label_tensor = np.zeros((num_of_labels, 2, self.row_size, self.row_size), dtype = np.longlong)

        for k in range(num_of_labels):

            if new_image_size :

                #print( "Upsampling image BEV detection to : " + str( new_image_size ))

                row_size = new_image_size[0]
                column_size = new_image_size[1]
                label_temp = np.zeros((row_size,column_size,2), dtype = np.longlong)
                label_data = raw_label[k]

                #print( "Label data is : {} with total drivable area : {}".format( str( label_data ) , torch.sum( label_data ) ))

                for i in range( row_size ):
                    for j in range( column_size ):

                        y_idx = row_size - i - 1
                        x_idx = column_size - j - 1

                        line_num = int(label_data[i][j])
                        if line_num == 255:
                            label_temp[y_idx][x_idx][1] = 0
                            # classification
                            label_temp[y_idx][x_idx][0] = 1 #6 
                        else: # class
                            # confidence
                            label_temp[y_idx][x_idx][1] = 1
                            # classification
                            label_temp[y_idx][x_idx][0] = line_num

                label_tensor[k,:,:,:] = np.transpose(label_temp, (2, 0, 1))
            else :

                label_temp = np.zeros((self.row_size,self.row_size,2), dtype = np.longlong)
                label_data = raw_label[k]

                for i in range(self.row_size):
                    for j in range(self.row_size):

                        y_idx = self.row_size - i - 1
                        x_idx = self.row_size - j - 1

                        line_num = int(label_data[i][j])
                        if line_num == 255:
                            label_temp[y_idx][x_idx][1] = 0
                            # classification
                            label_temp[y_idx][x_idx][0] = 6
                        else: # class
                            # confidence
                            label_temp[y_idx][x_idx][1] = 1
                            # classification
                            label_temp[y_idx][x_idx][0] = line_num
                

        if is_get_label_as_tensor:
            return torch.tensor(label_tensor)
        else:
            return label_tensor
    
    def loss(self, out, batch):
        train_label = batch['label']

        # Change drivable area label to K- Lane line format

        if torch.max( train_label ) != 255 :

            #print( "Changing drivable area label to K-Lane drivable area label format")

            train_label[ train_label == 0 ] = 255
            train_label[ train_label == 1 ] = 0
            
        #lanes_label = train_label[:,:, :144]
        lanes_label = train_label[:,:, :]
        lanes_label = self.label_formatting(lanes_label ,  new_image_size = train_label.shape[1 : 3] ) #channel0 = line number, channel1 = confidence

        #y_pred_cls = out[:, 0:7, :, :]
        #y_pred_conf = out[:, 7, :, :]
        
        if isinstance( out, tuple ) :
            out = out[0]
        

        y_pred_cls = out[:, 0:1, :, :]
        y_pred_conf = out[:, 1, :, :]

        


        y_label_cls = lanes_label[:, 0, :, :].cuda(  int( self.cfg.gpus_ids.split("'")[0]) )
        y_label_conf = lanes_label[:, 1, :, :].cuda(  int( self.cfg.gpus_ids.split("'")[0]) )

        if y_pred_cls.shape[ 2 ] != y_label_cls.shape[ 1 ] :

            #print( "Dimension of Road detection prediction before upsampling : {} ".format( y_pred_cls.shape ))

            y_pred_cls = nn.Upsample( size= y_label_cls.shape[ 1 : 3 ], mode="nearest" )( y_pred_cls )

            #print( "Dimension of Road detection prediction after upsampling : {}".format( y_pred_cls.shape ))

        if y_pred_conf.shape[ 2 ] != y_label_conf.shape[ 1 ] :

            y_pred_conf = nn.Upsample( size = y_label_conf.shape[ 1 : 3 ] , mode="nearest" )( torch.unsqueeze( y_pred_conf , dim= 0 ) )

        cls_loss = 0

        if self.num_classes >= 3 :
            cls_loss += nn.CrossEntropyLoss()(y_pred_cls.cpu(), y_label_cls.cpu())
        else :

            if self.focal_loss_alpha : 
                cross_entropy_loss = nn.BCEWithLogitsLoss()(torch.squeeze( y_pred_cls ).float().cuda(int( self.cfg.gpus_ids.split("'")[0])), torch.squeeze( y_label_cls ).float( ).cuda( int( self.cfg.gpus_ids.split("'")[0]) ))
                pt = torch.exp(-cross_entropy_loss) # prevents nans when probability 0
                F_loss = (1-pt)**self.focal_loss_gamma * cross_entropy_loss
                cls_loss += F_loss

            else :
                cls_loss += nn.BCELoss()(y_pred_cls.cpu(), y_label_cls.cpu())


        ## Dice Loss ###
        y_pred_conf = y_pred_conf.cuda( int( self.cfg.gpus_ids.split("'")[0]) )
        y_label_conf = y_label_conf.cuda( int( self.cfg.gpus_ids.split("'")[0]) )
        numerator = 2 * torch.sum(torch.mul(y_pred_conf, y_label_conf))
        denominator = torch.sum(torch.square(y_pred_conf)) + torch.sum(torch.square(y_label_conf)) + 1e-6
        dice_coeff = numerator / denominator
        conf_loss = (1 - dice_coeff)

        loss = conf_loss + cls_loss

        ret = {'loss': loss, 'loss_stats': {'conf': conf_loss, 'cls': cls_loss}}

        self.tensorboard_writer.add_scalar( "Total_Loss/Train", loss , self.step )
        self.tensorboard_writer.add_scalar( "Confidence_Loss/Train", conf_loss , self.step )
        self.tensorboard_writer.add_scalar( "Classification_Loss/Train", cls_loss , self.step )

        self.step = self.step + 1

        return ret

    def get_lane_map_numpy_with_label(self, output, data, is_flip=True, is_img=False):
        '''
        * in : output feature map
        * out: lane map with class or confidence
        *       per batch
        *       ### Label ###
        *       'conf_label': (144, 144) / 0, 1
        *       'cls_label': (144, 144) / 0, 1, 2, 3, 4, 5(lane), 255(ground)
        *       ### Raw Prediction ###
        *       'conf_pred_raw': (144, 144) / 0 ~ 1
        *       'cls_pred_raw': (7, 144, 144) / 0 ~ 1 (softmax)
        *       ### Confidence ###
        *       'conf_pred': (144, 144) / 0, 1 (thresholding)
        *       'conf_by_cls': (144, 144) / 0, 1 (only using cls)
        *       ### Classification ###
        *       'cls_idx': (144, 144) / 0, 1, 2, 3, 4, 5(lane), 255(ground)
        *       'conf_cls_idx': (144, 144) / (get cls idx in conf true positive)
        *       ### RGB Image ###
        *       'rgb_img_cls_label': (144, 144, 3)
        *       'rgb_img_cls_idx': (144, 144, 3)
        *       'rgb_img_conf_cls_idx': (144, 144, 3)
        '''
        lane_maps = dict()

        # for batch
        list_conf_label = []
        list_cls_label = []
        list_conf_pred_raw = []
        list_conf_pred = []
        list_cls_pred_raw = []
        list_cls_idx = []
        list_conf_by_cls = []
        list_conf_cls_idx = []

        batch_size = len(output['conf'])
        for batch_idx in range(batch_size):
            cls_label = data['label'][batch_idx].cpu().numpy()

            #print( "Shape of the Label is " + str( cls_label.shape ))
            conf_label = np.where(cls_label == 255, 0, 1)

            conf_pred_raw = output['conf'][batch_idx].cpu().numpy()
            if is_flip:
                conf_pred_raw = np.flip(np.flip(conf_pred_raw, 0),1)
            conf_pred = np.where(conf_pred_raw > self.cfg.conf_thr, 1, 0)
            cls_pred_raw = torch.nn.functional.softmax(output['cls'][batch_idx], dim=0)
            cls_pred_raw = cls_pred_raw.cpu().numpy()
            #print( "Shape of the Predicted Label is : " + str( cls_pred_raw.shape ))
            if is_flip:
                cls_pred_raw = np.flip(np.flip(cls_pred_raw, 1),2)
            cls_idx = np.argmax(cls_pred_raw, axis=0)
            cls_idx[np.where(cls_idx==6)] = 255
            conf_by_cls = cls_idx.copy()
            conf_by_cls = np.where(conf_by_cls==255, 0, 1)
            conf_cls_idx = cls_idx.copy()
            conf_cls_idx[np.where(conf_pred==0)] = 255

            list_cls_label.append(cls_label)
            list_conf_label.append(conf_label)
            list_conf_pred_raw.append(conf_pred_raw)
            list_conf_pred.append(conf_pred)
            list_cls_pred_raw.append(cls_pred_raw)
            list_cls_idx.append(cls_idx)
            list_conf_by_cls.append(conf_by_cls)
            list_conf_cls_idx.append(conf_cls_idx)

        lane_maps.update({
            'conf_label': list_conf_label,
            'cls_label': list_cls_label,
            'conf_pred_raw': list_conf_pred_raw,
            'cls_pred_raw': list_cls_pred_raw,
            'conf_pred': list_conf_pred,
            'conf_by_cls': list_conf_by_cls,
            'cls_idx': list_cls_idx,
            'conf_cls_idx': list_conf_cls_idx,
        })

        if is_img:
            list_rgb_img_cls_label = []
            list_rgb_img_cls_idx = []
            list_rgb_img_conf_cls_idx = []

            for batch_idx in range(batch_size):
                list_rgb_img_cls_label.append(
                    self.get_rgb_img_from_cls_map(list_cls_label[batch_idx]))
                list_rgb_img_cls_idx.append(
                    self.get_rgb_img_from_cls_map(list_cls_idx[batch_idx]))
                list_rgb_img_conf_cls_idx.append(
                    self.get_rgb_img_from_cls_map(list_conf_cls_idx[batch_idx]))
            
            lane_maps.update({
                'rgb_cls_label': list_rgb_img_cls_label,
                'rgb_cls_idx': list_rgb_img_cls_idx,
                'rgb_conf_cls_idx': list_rgb_img_conf_cls_idx,
            })

        return lane_maps

    def get_rgb_img_from_cls_map(self, cls_map):
        temp_rgb_img = np.zeros((144, 144, 3), dtype=np.uint8)
               
        for j in range(144):
            for i in range(144):
                idx_lane = int(cls_map[j,i])
                temp_rgb_img[j,i,:] = self.cfg.cls_lane_color[idx_lane] \
                                        if not (idx_lane == 255) else (0,0,0)

        return temp_rgb_img
        
