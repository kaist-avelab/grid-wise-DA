# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

import glob
import logging
import os
from typing import Dict, Iterator, List, Optional, Union, cast

import numpy as np

import torch

import argoverse.data_loading.object_label_record as object_label
from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.data_loading.pose_loader import get_city_SE3_egovehicle_at_sensor_t, read_city_name
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.calibration import Calibration, load_calib, load_image
from argoverse.utils.camera_stats import CAMERA_LIST, RING_CAMERA_LIST, STEREO_CAMERA_LIST
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3

# For making drivable area label
from scipy.spatial.transform import Rotation as R
from PIL import Image, ImageOps
from PIL import ImageDraw 

from argoverse.map_representation.map_api import ArgoverseMap

from argoverse.visualization.ground_visualization import draw_ground_pts_in_image

import copy
import math

import pickle

import os
import os.path as osp

from baseline.datasets.registry import DATASETS
from baseline.heuristic_da_detection import drivable_area_prediction_using_non_dl_lidar

LIST_OF_LOG_FOR_VISUALIZATION = [ "e17eed4f-3ffd-3532-ab89-41a3f24cf226" ,
                                 "b3def699-884b-3c9e-87e1-1ab76c618e0b" ,
                                 "af706af1-a226-3f6f-8d65-b1f4b9457c48",
                                 "11953248-1195-1195-1195-511954366464",
                                 "64c12551-adb9-36e3-a0c1-e43a0e9f3845"
                                 ]


logger = logging.getLogger(__name__)

@DATASETS.register_module
class ArgoverseRoadDetection:
    def __init__(self, data_root: str , split, mode_item='pillar', description=None, cfg=None , av_hd_map_dir = "/media/ave/sub4/Ofel/argoverse-api/map_files/" , max_number_of_log : int = 22 , is_random_log = False , number_batch= None, is_eval_visualization_dataset : bool = False , LIST_OF_LOG_FOR_VISUALIZATION : List[ str ] = None , range_drivable_area_map = None) -> None:
        # initialize class member
        self.CAMERA_LIST = CAMERA_LIST

        self.root_dir: str = data_root
        self.max_number_of_log = max_number_of_log
        #print( "Maximum number of log for Argoverse Road Detection is : " + str( self.max_number_of_log ))
        self.is_random_log = is_random_log
        self.number_batch = number_batch
        self.is_eval_visualization_dataset = is_eval_visualization_dataset
        self.range_drivable_area_map = range_drivable_area_map

        #if self.is_eval_visualization_dataset == True :

            #assert LIST_OF_LOG_FOR_VISUALIZATION is not None

        self.LIST_OF_LOG_FOR_VISUALIZATION = LIST_OF_LOG_FOR_VISUALIZATION

        
        #self._log_list = None
        #self._log_list = self.log_list
        self._image_list: Optional[Dict[str, Dict[str, List[str]]]] = None
        self._image_list_sync: Optional[Dict[str, Dict[str, List[np.ndarray]]]] = None
        self._lidar_list: Optional[Dict[str, List[str]]] = None
        #self._lidar_list = self.lidar_list
        self._image_timestamp_list: Optional[Dict[str, Dict[str, List[int]]]] = None
        self._timestamp_image_dict: Optional[Dict[str, Dict[str, Dict[int, str]]]] = None
        self._image_timestamp_list_sync: Optional[Dict[str, Dict[str, List[int]]]] = None
        self._lidar_timestamp_list: Optional[Dict[str, List[int]]] = None
        self._timestamp_lidar_dict: Optional[Dict[str, Dict[int, str]]] = None
        self._label_list: Optional[Dict[str, List[str]]] = None
        self._calib: Optional[Dict[str, Dict[str, Calibration]]] = None  # { log_name: { camera_name: Calibration } }
        self._city_name = None
        self.counter: int = 0

        self.image_count: int = 0
        self.lidar_count: int = 0

        self._log_list: Optional[List[str]] = None

        self.current_log = self.log_list[self.counter]

        print( "Log List of Argoverse Dataset : " + str( self.log_list ))

        self._len_number_frame_dataset = self.len_number_frame_dataset

        print( "Number of total dataset : " + str( self._len_number_frame_dataset ))

        #if cfg :
        #    self.max_number_of_log = cfg.max_number_of_log
        #else :


        

        assert self.image_list is not None
        assert self.lidar_list is not None
        assert self.label_list is not None

        # load calibration file
        self.calib_filename: str = os.path.join(self.root_dir, self.current_log, "vehicle_calibration_info.json")

        # lidar @10hz, ring camera @30hz, stereo camera @5hz
        self.num_lidar_frame: int = len(self.lidar_timestamp_list)
        self.num_ring_camera_frame: int = len(self.image_timestamp_list[RING_CAMERA_LIST[0]])
        self.num_stereo_camera_frame: int = len(self.image_timestamp_list[STEREO_CAMERA_LIST[0]])

        self.sync: SynchronizationDB = SynchronizationDB(data_root)

        # Get drivable area map for labelling

        self.avmap = ArgoverseMap( root = av_hd_map_dir )

        self.current_city_name = self.city_name

        self.drivable_rasterized_map = self.avmap.get_rasterized_driveable_area( self.current_city_name )[0][ : , : ]
        self.drivable_rasterized_map_rotation_matrix = np.array( self.avmap.get_rasterized_driveable_area( self.current_city_name )[1][ : , : ])

        assert self.image_list_sync is not None
        assert self.calib is not None

        self.cfg = cfg
        self.data_root = data_root
        self.mode_item = mode_item
        self.training = 'train' in split
        
        self.num_seq = int( max_number_of_log ) if isinstance( max_number_of_log , int ) else len( max_number_of_log ) if isinstance( max_number_of_log , list ) else len( self._log_list )#len(os.listdir(osp.join(root_dir, 'train1')))

        self._list_of_frame_index_log = self.list_of_frame_index_log

        self.is_using_ground_point_projection_to_image = self.cfg.is_using_ground_point_projection_to_image

        if self.is_using_ground_point_projection_to_image == True :

            self.LIST_OF_GROUND_POINTS_PROJECTED_TO_CAMERA = [ "ring_front_center", "ring_side_left" , "ring_front_right" , "ring_rear_left"]

    @property
    def city_name(self) -> str:
        """get city name of the current log

        Returns:
            city_name: city name of the current log, either 'PIT' or 'MIA'
        """
        return read_city_name(os.path.join(self.root_dir, self.current_log, "city_info.json"))

    @property
    def calib(self) -> Dict[str, Calibration]:
        """get calibration dict for current log

        Returns:
            calib: Calibration object for the current log
        """
        self._ensure_calib_is_populated()
        assert self._calib is not None
        return self._calib[self.current_log]

    def _ensure_calib_is_populated(self) -> None:
        """load up calibration object for all logs

        Returns:
            None
        """
        if self._calib is None:
            self._calib = {}
            for log in self.log_list:
                calib_filename = os.path.join(self.root_dir, log, "vehicle_calibration_info.json")
                self._calib[log] = load_calib(calib_filename)

    @property
    def log_list(self) -> List[str]:
        """return list of log (str) in the current dataset directory

        Returns:
            log_list: list of string representing log id
        """
        if self._log_list is None:

            def valid_log(log: str) -> bool:
                return os.path.exists(os.path.join(self.root_dir, log, "vehicle_calibration_info.json"))

            self._log_list = [x for x in os.listdir(self.root_dir) if valid_log(x)]

            print( "Number of Log List is : " + str( len( self._log_list )))

            if self.is_eval_visualization_dataset == True :

                #assert self.LIST_OF_LOG_FOR_VISUALIZATION is not None

                if self.LIST_OF_LOG_FOR_VISUALIZATION is not None :

                    self._log_list = LIST_OF_LOG_FOR_VISUALIZATION 

                    print( "Argoverse Dataset for visualization with Visualization Log : " + str( self._log_list ))

                    print( "----------------------------------")

                    return self.log_list
            
            if self.max_number_of_log is not None :

                print( "Maximum number of log in the dataset is : " + str( self.max_number_of_log ))

                if isinstance( self.max_number_of_log , int ) :

                    if self.is_random_log == True :

                        first_len_of_log = len( self._log_list )
                    
                        self._log_list = [ self._log_list[i] for i in  np.linspace( 0 , first_len_of_log - 1 , self.max_number_of_log  ).astype( int )  ] #self._log_list[ np.linspace( 0 , first_len_of_log - 1 , self.max_number_of_log  ).astype( int ) ]

                    else :
            
            	        self._log_list = self._log_list[ : self.max_number_of_log ]
                        
                else :

                    self._log_list = self._log_list[ int( self.max_number_of_log[0] ) : int( self.max_number_of_log[1] ) ]

        return self._log_list

    @property
    def len_number_frame_dataset( self ) -> int :

        log_list = self.log_list

        #print( "Log list is : " + str( log_list ))

        all_lidar_list = self.all_lidar_list()

        all_bev_tensor_list = self.all_bev_tensor_list() # In some cased delete LiDAR files and convert to BEV tensor,
                                                            # So change the list of lidar to list of BEV tensor

        all_lidar_list_number = int( np.sum( np.array([len(all_lidar_list[log_list[i]]) for i in range( len( self.log_list))])))

        all_bev_tensor_list_number =  int( np.sum( np.array([len(all_bev_tensor_list[log_list[i]]) for i in range( len( self.log_list))])))

        #print( "Total number of lidar list : " + str( all_lidar_list_number )+ " and total number of BEV Tensor List : " + str( all_bev_tensor_list_number))

        #print( "All lidar list is : " + str( all_lidar_list))

        #return int( np.sum( np.array([len(all_lidar_list[log_list[i]]) for i in range( len( self.log_list))])))
        
        if self.number_batch :
            return ( self.number_batch * ( max( all_lidar_list_number , all_bev_tensor_list_number) // self.number_batch ))
        return max( all_lidar_list_number , all_bev_tensor_list_number)
    
    @property
    def list_of_frame_index_log( self ) -> List[int] :

        all_lidar_list = self.all_lidar_list()

        list_of_frame_index_log = []

        for log_index in range( len( self._log_list )):

            if log_index == 0 :

                list_of_frame_index_log.append( 0 )
            else :
                list_of_frame_index_log.append( list_of_frame_index_log[ log_index- 1] + len( all_lidar_list[ self.log_list[ log_index - 1 ]]))

        return list_of_frame_index_log
    
    @property
    def image_list(self) -> Dict[str, List[str]]:
        """return list of all image path (str) for all cameras for the current log

        Returns:
            image_list: dictionary of list of image, with camera name as key
        """
        if self._image_list is None:
            self._image_list = {}
            for log in self.log_list:
                self._image_list[log] = {}
                for camera in CAMERA_LIST:
                    self._image_list[log][camera] = sorted(
                        glob.glob((os.path.join(self.root_dir, log, camera, "*.jpg")))
                    )
                    self.image_count += len(self._image_list[log][camera])
        return self._image_list[self.current_log]

    @property
    def image_list_sync(self) -> Dict[str, List[np.ndarray]]:
        """return list of image path (str) for all cameras for the current log.

        The different between image_list and image_list_sync is that image_list_sync
        syncronizes the image to lidar frame.

        Returns:
            image_list_sync: dictionary of list of image, with camera name as key. Each camera will have the same
                             number of images as #lidar frame.
        """
        logging.info("syncronizing camera and lidar sensor...")
        self._ensure_lidar_timestamp_list_populated()
        assert self._lidar_timestamp_list is not None

        if self._image_list_sync is None:

            self._image_list_sync = {}
            self._image_timestamp_list_sync = {}
            for log in self.log_list:

                self._image_list_sync[log] = {}
                self._image_timestamp_list_sync[log] = {}

                for camera in CAMERA_LIST:
                    self._image_timestamp_list_sync[log][camera] = cast(
                        List[int],
                        list(
                            filter(
                                lambda x: x is not None,
                                (
                                    self.sync.get_closest_cam_channel_timestamp(x, camera, log)
                                    for x in self._lidar_timestamp_list[log]
                                ),
                            )
                        ),
                    )

                    self._image_list_sync[log][camera] = [
                        self.get_image_at_timestamp(x, camera=camera, log_id=log, load=False)
                        for x in self._image_timestamp_list_sync[log][camera]
                    ]

        return self._image_list_sync[self.current_log]

    @property
    def image_timestamp_list_sync(self) -> Dict[str, List[int]]:
        """return list of image timestamp (str) for all cameras for the current log.

        The different between image_timestamp and image_timestamp_list_sync is that image_timestamp_list_sync
        synchronizes the image to the lidar frame.

        Returns:
            image_timestamp_list_sync: dictionary of list of image timestamp, with camera name as key.
                                       Each camera will have the same number of image timestamps as #lidar frame.
        """
        assert self.image_list_sync is not None
        assert self._image_timestamp_list_sync is not None
        return self._image_timestamp_list_sync[self.current_log]

    @property
    def lidar_list(self) -> List[str]:
        """return list of lidar path (str) of the current log                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

        Returns:
            lidar_list: list of lidar path for the current log
        """
        if self._lidar_list is None:
            self._lidar_list = {}
            for log in self.log_list:
                self._lidar_list[log] = sorted(glob.glob(os.path.join(self.root_dir, log, "lidar", "*.ply")))

                self.lidar_count += len(self._lidar_list[log])
        return self._lidar_list[self.current_log]
    
    def all_lidar_list(self) -> Dict[str , List[str]]:
        """return list of lidar path (str) of the current log

        Returns:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            lidar_list: list of lidar path for the current log
        """
        if self._lidar_list is None:
            self._lidar_list = {}
            for log in self.log_list:
                self._lidar_list[log] = sorted(glob.glob(os.path.join(self.root_dir, log, "lidar", "*.ply")))

                self.lidar_count += len(self._lidar_list[log])
            if self.lidar_count > 0 :

                if ((self.is_eval_visualization_dataset == True) & (self.LIST_OF_LOG_FOR_VISUALIZATION is not None )) :
                    # Then all lidar list is only Log of List Eval Visualization

                    for log in self.log_list :

                        self._lidar_list[ log ] = self._lidar_list[ log ][ : 1 ] # Take only first LiDAR frame of every log in LIST_OF_LOG_VISUALIZED                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               

                    self.lidar_count= len( self.log_list ) * 1 
        return self._lidar_list #[self.current_log]
    
    def all_bev_tensor_list(self) -> Dict[str , List[str]]:
        """return list of lidar path (str) of the current log

        Returns:
            lidar_list: list of lidar path for the current log
        """
        #if self.lidar_count == 0 :
        self._lidar_list = {}
        for log in self.log_list:
            #print( "Directory of BEV tensor is : " + str( os.path.join(self.root_dir, log, "BEV_tensor_folder") ))
            #print( "Directory of BEV tensor is : {}/{}/{}".format( self.root_dir , log , "BEV_tensor_folder" ))
            self._lidar_list[log] = sorted(glob.glob(os.path.join(self.root_dir, log, "BEV_tensor_folder", "*.pickle")))

            self._lidar_list[ log ] = [str( i ).replace( ".pickle" , '.ply') for i in self._lidar_list[ log ]]

            #print( "Directory of BEV tensor is : " + str( os.path.join(self.root_dir, log, "BEV_tensor_folder", "*.pickle") ))

            self.lidar_count += len(self._lidar_list[log])

        if self.lidar_count > 0 :

            if ((self.is_eval_visualization_dataset == True ) & ( self.LIST_OF_LOG_FOR_VISUALIZATION is not None )) :
                # Then all lidar list is only Log of List Eval Visualization

                for log in self.log_list :

                    self._lidar_list[ log ] = self._lidar_list[ log ][ : 1 ]

                self.lidar_count= len( self.log_list ) * 1 

        
        #print( "List of BEV tensor lidar : " + str( self._lidar_list ))
        return self._lidar_list #[self.current_log]


    @property
    def label_list(self) -> List[str]:
        """return list of label path (str) of the current log

        Returns:
            label: list of label path for the current log
        """
        if self._label_list is None:
            self._label_list = {}
            for log in self.log_list:
                self._label_list[log] = sorted(
                    glob.glob(os.path.join(self.root_dir, log, "per_sweep_annotations_amodal", "*.json"))
                )

        return self._label_list[self.current_log]

    @property
    def image_timestamp_list(self) -> Dict[str, List[int]]:
        """return dict of list of image timestamp (str) for all cameras for the current log.

        Returns:
            image_timestamp_list: dictionary of list of image timestamp for all cameras
        """
        assert self.image_list is not None
        assert self._image_list is not None

        if self._image_timestamp_list is None:
            self._image_timestamp_list = {}
            for log in self.log_list:
                self._image_timestamp_list[log] = {}
                for camera in CAMERA_LIST:
                    self._image_timestamp_list[log][camera] = [
                        int(x.split("/")[-1][:-4].split("_")[-1]) for x in self._image_list[log][camera]
                    ]

        return self._image_timestamp_list[self.current_log]

    @property
    def timestamp_image_dict(self) -> Dict[str, Dict[int, str]]:
        """return dict of list of image path (str) for all cameras for the current log, index by timestamp.

        Returns:
            timestamp_image_dict: dictionary of list of image path for all cameras, with timestamp as key
        """
        if self._timestamp_image_dict is None:
            assert self.image_timestamp_list is not None
            assert self._image_timestamp_list is not None
            assert self.image_list is not None
            assert self._image_list is not None

            self._timestamp_image_dict = {}

            for log in self.log_list:
                self._timestamp_image_dict[log] = {}
                for camera in CAMERA_LIST:
                    self._timestamp_image_dict[log][camera] = {
                        self._image_timestamp_list[log][camera][i]: self._image_list[log][camera][i]
                        for i in range(len(self._image_timestamp_list[log][camera]))
                    }

        return self._timestamp_image_dict[self.current_log]

    @property
    def timestamp_lidar_dict(self) -> Dict[int, str]:
        """return dict of list of lidar path (str) for the current log, index by timestamp.

        Returns:
            timestamp_lidar_dict: dictionary of list of lidar path, with timestamp as key
        """
        if self._timestamp_lidar_dict is None:
            assert self._lidar_timestamp_list is not None
            assert self._lidar_list is not None

            self._timestamp_lidar_dict = {}

            for log in self.log_list:
                self._timestamp_lidar_dict[log] = {
                    self._lidar_timestamp_list[log][i]: self._lidar_list[log][i]
                    for i in range(len(self._lidar_timestamp_list[log]))
                }

        return self._timestamp_lidar_dict[self.current_log]

    @property
    def lidar_timestamp_list(self) -> List[int]:
        """return list of lidar timestamp

        Returns:
            lidar_timestamp_list: list of lidar timestamp (at 10hz)
        """
        self._ensure_lidar_timestamp_list_populated()
        assert self._lidar_timestamp_list is not None
        return self._lidar_timestamp_list[self.current_log]

    def _ensure_lidar_timestamp_list_populated(self) -> None:
        """load up lidar timestamp for all logs

        Returns:
            None
        """
        assert self.lidar_list is not None
        assert self._lidar_list is not None

        if self._lidar_timestamp_list is None:
            self._lidar_timestamp_list = {}
            for log in self.log_list:
                self._lidar_timestamp_list[log] = [
                    int(x.split("/")[-1][:-4].split("_")[-1]) for x in self._lidar_list[log]
                ]

    def __iter__(self) -> Iterator["ArgoverseTrackingLoader"]:
        self.counter = -1

        return self

    def __next__(self , av_hd_map_dir = "/home/ofel04/argoverse-api/map_files/" ) -> "ArgoverseTrackingLoader":
        self.counter += 1

        if self.counter >= len(self):
            raise StopIteration
        else:
            self.current_log = self.log_list[self.counter]
            self.num_lidar_frame = len(self.lidar_timestamp_list)
            self.num_ring_camera_frame = len(self.image_timestamp_list[RING_CAMERA_LIST[0]])
            self.num_stereo_camera_frame = len(self.image_timestamp_list[STEREO_CAMERA_LIST[0]])
            
            avmap = ArgoverseMap( root = av_hd_map_dir )
            
            self.current_city_name = self.city_name
            self.drivable_rasterized_map = avmap.get_rasterized_driveable_area( self.current_city_name )[0][ : , : ]
            self.drivable_rasterized_map_rotation_matrix = np.array( avmap.get_rasterized_driveable_area( self.current_city_name )[1][ : , : ])
            
            print( "Self is : " + str( self ))
            return self

    def __len__(self) -> int:
        #return len(self.log_list)
        #print( "Number of frame dataset : " + str( self.len_number_frame_dataset ))

        return self.len_number_frame_dataset

    def __str__(self) -> str:
        frame_lidar = self.num_lidar_frame
        frame_image_ring = self.num_ring_camera_frame
        frame_image_stereo = self.num_stereo_camera_frame

        num_images = [len(self.image_list[cam]) for cam in CAMERA_LIST]

        num_annotations = [len(object_label.read_label(label)) for label in self.label_list]

        start_time = self.lidar_timestamp_list[0]
        end_time = self.lidar_timestamp_list[-1]

        time_in_sec = (end_time - start_time) * 10 ** (-9)
        return f"""
--------------------------------------------------------------------
------Log id: {self.current_log}
--------------------------------------------------------------------
Time: {time_in_sec} sec
# frame lidar (@10hz): {frame_lidar}
# frame ring camera (@30hz): {frame_image_ring}
# frame stereo camera (@5hz): {frame_image_stereo}

Total images: {sum(num_images)}
Total bounding box: {sum(num_annotations)}
        """
    
    # Function to take BEV Drivable Label from Drivable Rasterized Map

    def get_rasterized_drivabel_area_label( self , key : int , map_range : list = [ -50 , -50 , 50 , 70] , grid_size : list = [0.2, 0.2]) -> np.array :

        # Get matrix rotation of Ego Vehicle Coordinate to City Coordinate
    
        city_to_egovehicle_se3 = self.get_pose(key)

        # Get ego vehicle city coordinate and rotation

        x,y,_ = city_to_egovehicle_se3.translation

        #print( "Location of ego- vehicle : x = {}, y = {}".format( x , y ) )

        ego_vehicle_rotation = city_to_egovehicle_se3.rotation

        pose_rotation_matrix_to_yaw = R.from_matrix( ego_vehicle_rotation )

        yaw_angle = pose_rotation_matrix_to_yaw.as_euler( "zyx" , degrees = True )[0]


        # Get Drivable area in Raster Map

        x_raster_map_coordinate = x + self.drivable_rasterized_map_rotation_matrix[0][2]
        y_raster_map_coordinate = y + self.drivable_rasterized_map_rotation_matrix[1][2]
        
        drivable_area_map_crop_size = max( abs( map_range[0]) , abs(map_range[1]) , abs( map_range[2] ) , abs( map_range[3] ))


        img = Image.fromarray( self.drivable_rasterized_map ).crop( (( x_raster_map_coordinate-drivable_area_map_crop_size) , ( y_raster_map_coordinate -drivable_area_map_crop_size ) , ( x_raster_map_coordinate + drivable_area_map_crop_size ) , ( y_raster_map_coordinate +  drivable_area_map_crop_size )))#.rotate( -yaw_angle/ math.pi * 180 + 180)

        img = img.resize((int( 2*drivable_area_map_crop_size * 1/grid_size[0]), int( 2*drivable_area_map_crop_size* 1/grid_size[1])), resample=Image.BOX)


        #img_with_ground_height = Image.fromarray( ground_heigh_raster_map_with_color ).crop( (( x_raster_map_coordinate-80) , ( y_raster_map_coordinate -80 ) , ( x_raster_map_coordinate + 80 ) , ( y_raster_map_coordinate +  80 ))).rotate( -yaw_angle/ math.pi * 180 + 180)

        img = np.array( img )

        img = ImageOps.mirror( Image.fromarray( img ).rotate( yaw_angle + 90))

        rotated_bev_image_shape = np.array( img ).shape

        img = img.crop( ( rotated_bev_image_shape[0]/2 + map_range[0] * 1/grid_size[0] , rotated_bev_image_shape[1]/2 - map_range[3] * 1/grid_size[1] , rotated_bev_image_shape[0]/2 + map_range[2] * 1/grid_size[0]  , rotated_bev_image_shape[1]/2 - map_range[1] * 1/grid_size[1]))

        img = img.resize((int( 100 * 1/grid_size[0]), int( 120 * 1/grid_size[1])), resample=Image.BOX)

        return np.array( img )

    def get_drivable_area_label_from_pickle( self , idx : int ) -> str :
        
        name_of_drivabel_area_label_pickle = self.root_dir + "/" + str( self.current_log ) + "/BEV_drivable_area_label/" + self._lidar_list[ self.current_log ][idx].split( "/" )[-1].replace(".ply", "" ) + ".pickle"
    
        return str( name_of_drivabel_area_label_pickle )

    def get_drivable_area_projected_to_camera( self, idx : int ) -> str :

        name_of_drivable_area_segmentation = self.root_dir + "/" + str( self.current_log ) + "/DA_Projection_to_Camera/" + self._lidar_list[ self.current_log ][ idx ].split( "/" )[-1].replace( ".ply" , "") + ".pickle"

        return str( name_of_drivable_area_segmentation )

    def __getitem__(self, key: int ) -> "ArgoverseTrackingLoader":

        # Find the log index

        counter_log = 0

        for frame_index_log in self.list_of_frame_index_log[ 1 : ] :

            if key < frame_index_log :

                break

            counter_log = counter_log + 1

        self.counter = counter_log #key
        #print( "Key is : " + str( key))
        self.current_log = self.log_list[self.counter]
        self.num_lidar_frame = len(self.lidar_timestamp_list)
        self.num_ring_camera_frame = len(self.image_timestamp_list[RING_CAMERA_LIST[0]])
        self.num_stereo_camera_frame = len(self.image_timestamp_list[STEREO_CAMERA_LIST[0]])

        key_in_current_log = key - self.list_of_frame_index_log[ self.counter ]

        #print( "List of frame index log : " + str( self.list_of_frame_index_log ))
        #print( "Global Key is : {} Key in current log : {}".format( key , key_in_current_log))
        #self.bev_map_drivable_area_label = self.get_rasterized_drivabel_area_label( key_in_current_log ) #key ) # Get Drivable Area Label for map

        sample = dict()

        # Extract BEV tensor pillars

        with open( self.get_bev_tensor_lidar_from_pickle( key_in_current_log ) , "rb") as f :

            bev_tensor = pickle.load( f , encoding = "latin1" )

        try :
            bev_tensor_pillar = np.squeeze( bev_tensor[0] , axis = 0 )
            bev_tensor_pillar_indices = np.squeeze( bev_tensor[1] , axis = 0 )
        except :
            bev_tensor_pillar = bev_tensor[0]
            bev_tensor_pillar_indices = bev_tensor[1]

        #print( "Size of BEV tensor pillar : " + str( bev_tensor_pillar.shape))

        #sample[ "key" ] = "log:{}_frame:{}".format( self.current_log , key_in_current_log )

        sample[ "pillars" ] = bev_tensor_pillar
        sample[ "pillar_indices"] =  bev_tensor_pillar_indices

        #sample[ "lidar_data" ] = self.get_lidar_in_rasterized_map_coordinate(idx= key_in_current_log , log_id= self.current_log)
        
        if self.range_drivable_area_map is None :

            with open( self.get_drivable_area_label_from_pickle( key_in_current_log ) , "rb" ) as f :

                drivable_area_label = pickle.load( f , encoding= "latin1" )
                
                

        else :
        
            drivable_area_map_range = [ - self.range_drivable_area_map[0]/2 , self.range_drivable_area_map[0]/2 , -self.range_drivable_area_map[1]/2 , self.range_drivable_area_map[1]/2]
        
            drivable_area_label = self.get_rasterized_drivabel_area_label( key_in_current_log , map_range = drivable_area_map_range )
        
        sample[ "label" ] = drivable_area_label

        if self.is_using_ground_point_projection_to_image == True :

            try :
                # Check if Argoverse Log have LiDAR point cloud or not

                lidar_pts =  self.get_lidar( idx = key_in_current_log )
            except :
                lidar_pts = None

            

            if lidar_pts is not None :
                list_of_projection_ground_point_to_image = [ draw_ground_pts_in_image(
                self.sync,
                #self.get_lidar( idx = key_in_current_log ),
                lidar_pts,
                self.get_pose(key_in_current_log),
                self.avmap,
                self.current_log,
                self.lidar_timestamp_list[key_in_current_log],
                self.city_name,
                self.data_root,
                self.cfg.experiment_name,
                camera = str( i ),
                is_save_image_ground_point_projection_to_image = False

            
            ) for i in self.LIST_OF_GROUND_POINTS_PROJECTED_TO_CAMERA ]
            
            
            else :

                list_of_projection_ground_point_to_image = [ self.get_image( idx= key_in_current_log , camera = str( i )) for i in self.LIST_OF_GROUND_POINTS_PROJECTED_TO_CAMERA ]
            
            # Write text cmarea projection image point to ground information

            #list_of_projection_ground_point_to_image = [ Image.fromarray( i ) for i in list_of_projection_ground_point_to_image ]

            #draw_image_front_camera_projection = ImageDraw.Draw(list_of_projection_ground_point_to_image[0] )

            #font = ImageFont.truetype("sans-serif.ttf", 16)

            #draw_image_front_camera_projection.text((0, 0),"Front Camera",(0,0,0),font=font)

            #list_of_projection_ground_point_to_image[0] =






            #resize, first image
            for i, image_ground_point_projection in enumerate( list_of_projection_ground_point_to_image ) :

                #print( "Prcoessing ground point projection to camera : " + str( self.LIST_OF_GROUND_POINTS_PROJECTED_TO_CAMERA[i]))

                if i == 0 :
                    # Then new_image is the first projection groun point to image

                    new_image = image_ground_point_projection 

                    continue

                if image_ground_point_projection is None :

                    image_ground_point_projection = np.ones( (1280,1920,3)).astype( np.uint8 )

                image1 = Image.fromarray( new_image ).resize((1920 , int(i* 1280)))
                image2 = Image.fromarray( image_ground_point_projection ).resize( ( 1920,1280 ))
                #print( "Size of prediction image is : " + str( np.array( image2 ).shape ))
                image1_size = image1.size
                image2_size = image2.size
                new_image = Image.new('RGB',(image1_size[0], image1_size[1] + image2_size[1]), (250,250,250))
                new_image.paste(image1,(0,0))
                new_image.paste(image2,( 0, image1_size[1] ))

                new_image = np.array( new_image ).astype( np.uint8 )



            sample[ "ground_point_projection_to_image" ] = new_image 

            # Make LiDAR points visualizationwith DA label

            try :

                lidar_pts_this_frame = self.get_lidar( key_in_current_log )
                    
                lidar_point_visualization_with_colored = lidar_pts_visualization( lidar_pts_this_frame )

                sample[ "LiDAR_points_visualization" ] = lidar_point_visualization_with_colored
            except :

                sample[ "LiDAR_points_visualization" ] = np.array([[[255, 255 , 255] for i in range(500)] for j in range( 600 )])
            
            #drivable_area_label = self.get_rasterized_drivabel_area_label( key_in_current_log )

            if self.cfg.is_using_heuristic_da_detection == True :
            
                if self.training == False :

                    da_detection_result_using_gaussian_kernel = drivable_area_prediction_using_non_dl_lidar( self , key_in_current_log  ,  HEIGHT_MEAN_TRESHOLD = 0.5, BAYESIAN_GAUSSIAN_KERNEL_RADIUS= 5 , LIDAR_POINT_FILTERING_HEIGHT_TRESHOLD = 5 )

                    da_detection_result_using_gaussian_kernel_with_color = np.array( [[ [120 , 120 , 120 ] if i == 1 else [ 255, 255 , 255 ] for i in j ] for j in da_detection_result_using_gaussian_kernel ] )

                    sample[ "DA_detection_using_gaussian_kernel"] = da_detection_result_using_gaussian_kernel_with_color

            #return frame_lidar_points, (ground_height_bev_map,drivable_area_label, lidar_point_visualization_with_colored )



        #print( "Sum of drivable area : " + str( np.sum( sample["label"])))
      
        return sample #self

    def get(self, log_id: str) -> "ArgoverseTrackingLoader":
        """get ArgoverseTrackingLoader object with current_log set to specified log_id

        Args:
            log_id: log id
        Returns:
            ArgoverseTrackingLoader: with current_log set to log_id
        """
        self.current_log = log_id
        self.num_lidar_frame = len(self.lidar_timestamp_list)
        self.num_ring_camera_frame = len(self.image_timestamp_list[RING_CAMERA_LIST[0]])
        self.num_stereo_camera_frame = len(self.image_timestamp_list[STEREO_CAMERA_LIST[0]])
        return self

    def get_image_list(self, camera: str, log_id: Optional[str] = None, load: bool = False) -> List[str]:
        """Get list of image/or image path

        Args:
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_list is not None
        assert self._image_list is not None

        if log_id is None:
            log_id = self.current_log
        if load:
            return [self.get_image(i, camera) for i in range(len(self._image_list[log_id][camera]))]

        return self._image_list[log_id][camera]

    def get_image_list_sync(self, camera: str, log_id: Optional[str] = None, load: bool = False) -> List[str]:
        """Get list of image/or image path in lidar index

        Args:
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_list_sync is not None
        assert self._image_list_sync is not None

        if log_id is None:
            log_id = self.current_log

        if load:
            return [self.get_image_sync(i, camera) for i in range(len(self._image_list_sync[log_id][camera]))]

        return self._image_list_sync[log_id][camera]

    def get_image_at_timestamp(
        self,
        timestamp: int,
        camera: str,
        log_id: Optional[str] = None,
        load: bool = True,
    ) -> Optional[Union[str, np.ndarray]]:
        """get image or image path at a specific timestamp

        Args:
            timestamp: timestamp
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.timestamp_image_dict is not None
        assert self._timestamp_image_dict is not None

        if log_id is None:
            log_id = self.current_log
        assert self.timestamp_image_dict is not None
        try:
            image_path = self._timestamp_image_dict[log_id][camera][timestamp]
        except KeyError:
            logging.error(f"Cannot find {camera} image at timestamp {timestamp} in log {log_id}")
            return None

        if load:
            return load_image(image_path)
        return image_path

    def get_image(
        self, idx: int, camera: str, log_id: Optional[str] = None, load: bool = True
    ) -> Union[str, np.ndarray]:
        """get image or image path at a specific index (in image index)

        Args:
            idx: image based 0-index
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_timestamp_list is not None
        assert self._image_timestamp_list is not None
        assert self.image_list is not None
        assert self._image_list is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._image_timestamp_list[log_id][camera])
        image_path = self._image_list[log_id][camera][idx]

        if load:
            return load_image(image_path)
        return image_path

    def get_image_sync(
        self, idx: int, camera: str, log_id: Optional[str] = None, load: bool = True
    ) -> Union[str, np.ndarray]:
        """get image or image path at a specific index (in lidar index)

        Args:
            idx: lidar based 0-index
            camera: camera based on camera_stats.CAMERA_LIST
            log_id: log_id, if not specified will use self.current_log
            load: whether to return image array (True) or image path (False)

        Returns:
            np.array: list of image path (str or np.array)),
        """
        assert self.image_timestamp_list_sync is not None
        assert self._image_timestamp_list_sync is not None
        assert self.image_list_sync is not None
        assert self._image_list_sync is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._image_timestamp_list_sync[log_id][camera])
        image_path = self._image_list_sync[log_id][camera][idx]

        if load:
            return load_image(image_path)
        return image_path

    def get_lidar(self, idx: int, log_id: Optional[str] = None, load: bool = True , is_return_intensity = False) -> Union[str, np.ndarray]:
        """Get lidar corresponding to frame index idx (in lidar frame).

        Args:
            idx: Lidar frame index
            log_id: ID of log to search (default: current log)
            load: whether to load up the data, will return path to the lidar file if set to false

        Returns:
            Either path to lidar at a specific index, or point cloud data if load is set to True
        """
        assert self.lidar_timestamp_list is not None
        assert self._lidar_timestamp_list is not None
        assert self.lidar_list is not None
        assert self._lidar_list is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._lidar_timestamp_list[log_id])

        if load:
                        
            return load_ply(self._lidar_list[log_id][idx].replace("BEV_tensor_folder", "lidar").replace(".pickle", ".ply") , is_return_intensity = is_return_intensity)
        return self._lidar_list[log_id][idx]
    
    def get_lidar_in_rasterized_map_coordinate( self , idx : int , log_id: Optional[str] = None, load: bool = True , is_return_intensity = True) -> np.ndarray :
        # Function to get LiDAR points in a frame in City Coordinate

        lidar_points_in_ego_vehicle_coordinate = self.get_lidar( idx , log_id , load , is_return_intensity=is_return_intensity )

        city_to_egovehicle_se3 = self.get_pose(idx)

        x,y,_ = city_to_egovehicle_se3.translation

        lidar_pts_for_bev = copy.deepcopy(lidar_points_in_ego_vehicle_coordinate)[ : , :3 ]
        lidar_pts_in_city_coordinate = city_to_egovehicle_se3.transform_point_cloud(
                lidar_pts_for_bev
            )  # put into city coords
        
        #print( "Shape of LiDAR points in City Coordinate : " + str( lidar_pts_in_city_coordinate.shape ))
        #print( "Shape of LiDAR points in Ego Vehicle Coordinate : " + str( lidar_points_in_ego_vehicle_coordinate.shape ))
        
        if lidar_points_in_ego_vehicle_coordinate.shape[1] == 4 : 
            lidar_pts_in_city_coordinate = np.concatenate( [ lidar_pts_in_city_coordinate[ : ] , lidar_points_in_ego_vehicle_coordinate[ : , 3 : ] ] , axis = 1 )
        
        if lidar_pts_in_city_coordinate.shape[1] == 4 :

            lidar_pts_in_city_coordinate = lidar_pts_in_city_coordinate - [[ x , y , 0 , 0] for i in range( lidar_pts_in_city_coordinate.shape[0]) ]
        else :

            lidar_pts_in_city_coordinate = lidar_pts_in_city_coordinate - [[ x , y , 0] for i in range( lidar_pts_in_city_coordinate.shape[0]) ]


        # Rotate LiDAR point in ego Vehicle orientation
            
        ego_vehicle_rotation = city_to_egovehicle_se3.rotation

        pose_rotation_matrix_to_yaw = R.from_matrix( ego_vehicle_rotation )

        yaw_angle = pose_rotation_matrix_to_yaw.as_euler( "zyx" , degrees = False )[0]

        yaw_angle = yaw_angle + 0.5 * math.pi

        lidar_pts_new_coordinate = lidar_pts_in_city_coordinate[ : , 0 : 2].dot( np.array( [[math.cos( yaw_angle ) , -math.sin( yaw_angle )] , [ math.sin( yaw_angle ) , math.cos( yaw_angle )]]))

        #lidar_pts_new_coordinate[ : , 2 : ] = lidar_pts_in_city_coordinate[ : , 2 : ]

        #print( "Shape of LiDAR Point in City Coordinate : " + str( lidar_pts_new_coordinate.shape ))

        lidar_pts_new_coordinate = np.concatenate( [ np.array( lidar_pts_new_coordinate ) , np.array( lidar_pts_in_city_coordinate )[ : , 2 : ]]  , axis = 1 )

        # Mirror LiDAR points in y- axis to make LiDAR BEV map coordinate same exactly with ego- vehicle coordinate
            
        lidar_pts_new_coordinate = np.concatenate( [ -1* lidar_pts_new_coordinate[ : , 0 ].reshape(-1, 1) , -1* lidar_pts_new_coordinate[ : , 1 ].reshape(-1, 1) , lidar_pts_new_coordinate[ : , 2 : ] ] , axis = 1 )

        return lidar_pts_new_coordinate

    def get_bev_tensor_lidar_from_pickle( self , idx : int ) -> str :
    
        name_of_bev_tensor_file = self.root_dir + "/" + str( self.current_log ) + "/BEV_tensor_folder/" + self._lidar_list[ self.current_log ][idx].split( "/" )[-1].replace(".ply", "" ) + ".pickle"
        
        #print( "Name of BEV tensor file : " + str( name_of_bev_tensor_file ))
        return name_of_bev_tensor_file
    	
    def get_label_object(self, idx: int, log_id: Optional[str] = None) -> List[ObjectLabelRecord]:
        """Get label corresponding to frame index idx (in lidar frame).

        Args:
            idx: Lidar frame index
            log_id: ID of log to search (default: current log)

        Returns:
            List of ObjectLabelRecord info for a particular index
        """
        assert self.lidar_timestamp_list is not None
        assert self._lidar_timestamp_list is not None
        assert self.label_list is not None
        assert self._label_list is not None

        if log_id is None:
            log_id = self.current_log

        assert idx < len(self._lidar_timestamp_list[log_id])

        return object_label.read_label(self._label_list[log_id][idx])
        

    def get_calibration(self, camera: str, log_id: Optional[str] = None) -> Calibration:
        """Get calibration corresponding to the camera.

        Args:
            camera: name of the camera; one of::

               ["ring_front_center",
                "ring_front_left",
                "ring_front_right",
                "ring_rear_left",
                "ring_rear_right",
                "ring_side_left",
                "ring_side_right",
                "stereo_front_left",
                "stereo_front_right"]

            log_id: ID of log to search (default: current log)

        Returns:
            Calibration info for a particular index
        """
        self._ensure_calib_is_populated()
        assert self._calib is not None

        if log_id is None:
            log_id = self.current_log

        return self._calib[log_id][camera]

    def get_pose(self, idx: int, log_id: Optional[str] = None) -> Optional[SE3]:
        """Get pose corresponding to an index in a particular log_id.

        Args:
            idx: Lidar frame index
            log_id: ID of log to search (default: current log)

        Returns:
            Pose for a particular index
        """

        # Change all dataset index to current log index
        
        if log_id is None:
            log_id = self.current_log
        self._ensure_lidar_timestamp_list_populated()
        assert self._lidar_timestamp_list is not None

        timestamp = self._lidar_timestamp_list[log_id][idx]

        return get_city_SE3_egovehicle_at_sensor_t(timestamp, self.root_dir, log_id)

    def get_idx_from_timestamp(self, timestamp: int, log_id: Optional[str] = None) -> Optional[int]:
        """Get index corresponding to a timestamp in a particular log_id.

        Args:
            timestamp: Timestamp to search for
            log_id: ID of log to search (default: current log)

        Returns:
            Index in the log if found, or None if not found.
        """
        if log_id is None:
            log_id = self.current_log
        self._ensure_lidar_timestamp_list_populated()
        assert self._lidar_timestamp_list is not None

        for i in range(len(self._lidar_timestamp_list[log_id])):
            if self._lidar_timestamp_list[log_id][i] == timestamp:
                return i
        return None

    def print_all(self) -> None:
        assert self.image_timestamp_list is not None
        assert self.lidar_timestamp_list is not None
        print("#images:", self.image_count)
        print("#lidar:", self.lidar_count)

def lidar_pts_visualization( lidar_pts , voxel_size = [ 0.2 , 0.2 ] , lidar_range = [-50 , 70 , -50 , 50 ] , HEIGHT_MEAN_TRESHOLD = -1 , HEIGHT_VARIANCE_TRESHOLD = 0.01 , DIFFERENT_MAX_MIN_HEIGHT_TRESHOLD = 0.2 , BAYESIAN_GAUSSIAN_KERNEL_RADIUS = 4 , LIDAR_POINT_FILTERING_HEIGHT_TRESHOLD = 10 ) :

    lidar_pts = lidar_pts #argoverse_data.get_lidar( index_lidar )

    lidar_range = [-50 , 70 , -50 , 50 ] 

    # Filtering the lidar height below height treshold

    lidar_pts = lidar_pts[ lidar_pts[ : , 2 ] <= LIDAR_POINT_FILTERING_HEIGHT_TRESHOLD ]

    #print( "Lidar points at beginning : " + str( lidar_pts ))
    if lidar_range[0] < 0 :
        lidar_range[1] = lidar_range[1] + -1* lidar_range[0]
        lidar_pts[ : , 0 ] = lidar_pts[ : , 0 ] + -1 * lidar_range[0]
        lidar_range[0] = 0        

        

    if lidar_range[2] < 0 :
        lidar_range[3] = lidar_range[3] + -1* lidar_range[2]
        lidar_pts[ : , 1 ] = lidar_pts[ : , 1 ] + -1* lidar_range[2]
        lidar_range[2] = 0       

    #print( "Lidar points after offset : " + str( lidar_pts ))

    
    length_of_pillar_bev = int( ( lidar_range[ 1] - lidar_range[0] )/ voxel_size[0] )
    width_of_pillar_bev = int( ( lidar_range[ 3 ] - lidar_range[2] )/ voxel_size[1] )
    
    list_of_points_in_tensor = [[[] for i in range( width_of_pillar_bev ) ] for j in range( length_of_pillar_bev )]

    for lidar_point in lidar_pts :

        X_coordinate_lidar_point = int( lidar_point[0]//voxel_size[0] )

        if (( X_coordinate_lidar_point < 0 ) | ( X_coordinate_lidar_point >= length_of_pillar_bev ))  :
            continue
            
        Y_coordinate_lidar_point = int( lidar_point[1]//voxel_size[1] )

        if (( Y_coordinate_lidar_point < 0 ) | ( Y_coordinate_lidar_point >= width_of_pillar_bev )) :
            continue

        #print( "Select lidar points : " + str( lidar_point ))

        list_of_points_in_tensor[ X_coordinate_lidar_point][ Y_coordinate_lidar_point ].append( lidar_point )

    # Measure mean and height variance of every grid
        
    list_of_lidar_point_in_voxel = [[0 for i in range( width_of_pillar_bev ) ] for j in range( length_of_pillar_bev )]
        
    for x in range( length_of_pillar_bev ) :

        for y in range( width_of_pillar_bev ) :

            if len( list_of_points_in_tensor[x][y] ) > 0 :

                #print( "List of points are : " + str( list_of_points_in_tensor[x][y] ))

                list_of_lidar_point_in_voxel[ x][y] = 1


    #assert None not in list_of_ground_and_none_ground_mean_and_variance_height

    list_of_lidar_point_in_voxel_with_color = [[[0 , 255 , 0] if i==1 else [ 255 , 255 , 255 ] for i in j ] for j in list_of_lidar_point_in_voxel ]

    list_of_lidar_point_in_voxel_with_color = np.array( list_of_lidar_point_in_voxel ).astype( np.uint8 )

    image_of_lidar_point_visualization = np.array( ImageOps.flip( ImageOps.mirror( Image.fromarray( list_of_lidar_point_in_voxel_with_color ))))
    
    #print( "Sum of voxel with lidar points in lidar points visualization are : " + str( np.

    image_of_lidar_point_visualization_with_color = [[[0 , 0 , 255] if i==1 else [ 255 , 255 , 255 ] for i in j ] for j in np.array( image_of_lidar_point_visualization ) ]

    return np.array( image_of_lidar_point_visualization_with_color )

