import pandas as pd
import numpy as np

import os
import pickle

from argoverse.data_loading.argoverse_road_detection_loader import ArgoverseRoadDetectionLoader

from argoverse.visualization.ground_visualization import draw_ground_pts_in_image

from argoverse.map_representation.map_api import ArgoverseMap

from scipy.spatial.transform import Rotation as R

from math import atan2
from PIL import Image, ImageOps
import cv2

import math


def convert_lidar_points_to_bev_tensors( lidar_pts , voxel_size = [ 0.2 , 0.2 ] , max_number_pillars = 21000 , max_number_points = 32 , lidar_range = [-50 , 50 , -50 , 70 ]):

    if lidar_range[0] < 0 :
        lidar_range[1] = lidar_range[1] + -1* lidar_range[0]
        lidar_pts[ : , 0 ] = lidar_pts[ : , 0 ] + -1 * lidar_range[0]
        lidar_range[0] = 0        

        

    if lidar_range[2] < 0 :
        lidar_range[3] = lidar_range[3] + -1* lidar_range[2]
        lidar_pts[ : , 1 ] = lidar_pts[ : , 1 ] + -1* lidar_range[2]
        lidar_range[2] = 0       

        

    
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
        

    list_of_bev_tensor = []

    list_of_bev_tensor_indices = []
    
    for x in range( length_of_pillar_bev ) :

        for y in range( width_of_pillar_bev ) :

            if len( list_of_points_in_tensor[x][y] ) > 0 :
    
                if len( list_of_points_in_tensor[x][y] ) > max_number_points :
    
                    list_of_selected_index = np.linspace( 0 , len( list_of_points_in_tensor[x][y] ) -1 , max_number_points ).astype(int)

                    #print( "List of selected index : " + str( list_of_selected_index ))
    
                    selected_points_in_pillar = np.array( list_of_points_in_tensor[x][y])[list_of_selected_index]
    
                    # Extract different of points coordinate to all points in pillar and different of points coordinate to all pillar BEV center
    
                    average_coordinate_selected_points_in_pillar = selected_points_in_pillar[ : , : ].mean( axis = 0 )

                    average_coordinate_selected_points_in_pillar[ 3 ] = 0
    
                    different_coordinate_selected_points_in_pillar = selected_points_in_pillar - average_coordinate_selected_points_in_pillar
    
                    center_of_bev_pillar = [ x* voxel_size[0] + voxel_size[0]/2 , y* voxel_size[1] + voxel_size[1]/2 ]
    
                    different_coordinate_selected_points_in_pillar_to_bev_pillar_center = selected_points_in_pillar[ : , : 2 ] - center_of_bev_pillar
    
                    extracted_selected_points_in_pillar = np.concatenate( [ selected_points_in_pillar , different_coordinate_selected_points_in_pillar[ : , :3] , different_coordinate_selected_points_in_pillar_to_bev_pillar_center ] , axis = 1 )
    
                    list_of_bev_tensor.append( extracted_selected_points_in_pillar )
    
                else :
    
                    selected_points_in_pillar = np.array( list_of_points_in_tensor[x][y][ : ] )
    
                    #print( "Selected points in pillar : " + str( selected_points_in_pillar ))
    
                    # Extract different of points coordinate to all points in pillar and different of points coordinate to all pillar BEV center
    
                    average_coordinate_selected_points_in_pillar = selected_points_in_pillar[ : , : ].mean( axis = 0 )

                    #print( "Average coordinate selected points in pillar : " + str( average_coordinate_selected_points_in_pillar ))

                    average_coordinate_selected_points_in_pillar[3] = 0
    
                    different_coordinate_selected_points_in_pillar = selected_points_in_pillar - average_coordinate_selected_points_in_pillar
    
                    center_of_bev_pillar = [ x* voxel_size[0] + voxel_size[0]/2 , y* voxel_size[1] + voxel_size[1]/2 ]
    
                    different_coordinate_selected_points_in_pillar_to_bev_pillar_center = selected_points_in_pillar[ : , : 2 ] - center_of_bev_pillar
    
                    extracted_selected_points_in_pillar = np.concatenate( [ selected_points_in_pillar , different_coordinate_selected_points_in_pillar[ : , :3] , different_coordinate_selected_points_in_pillar_to_bev_pillar_center ] , axis = 1 )
    
                    #for padding_bev_tensor in range( max_number_points - len( list_of_points_in_tensor[x][y] ) ) :
    
                    extracted_selected_points_in_pillar = np.append(extracted_selected_points_in_pillar , np.array( [[ 0 for i in range(9) ] for j in range( max_number_points - len( list_of_points_in_tensor[x][y]))]).reshape( -1,9) , axis = 0)
    
                    #print( "Final extracted points in pillar : " + str( extracted_selected_points_in_pillar ))
                    list_of_bev_tensor.append(extracted_selected_points_in_pillar )
                        
                list_of_bev_tensor_indices.append(np.array([ 0 , x , y ] ))

    list_of_bev_tensor = np.stack( list_of_bev_tensor , axis = 0 )
    list_of_bev_tensor_indices = np.stack( list_of_bev_tensor_indices , axis = 0)

    #print( "List of BEV tensor : " + str( list_of_bev_tensor ))
    #print( "List of BEV tensor indices : " + str( list_of_bev_tensor_indices ))

    print( "Dimension of BEV tensor pillars without additional pillars : " + str( list_of_bev_tensor.shape ))

                                                   
    if list_of_bev_tensor.shape[0] > max_number_pillars :

        selected_pillar_indices = np.linspace( 0 , list_of_bev_tensor.shape[0] - 1  , max_number_pillars ).astype( int )

        list_of_bev_tensor = list_of_bev_tensor[ selected_pillar_indices , : ]

        list_of_bev_tensor_indices = list_of_bev_tensor_indices[ selected_pillar_indices ,  :  ]

    else :

        #for padding_bev_tensor_pillar in range( max_number_pillars - list_of_bev_tensor.shape[0] ) :

        number_of_padding_tensor = max_number_pillars - list_of_bev_tensor.shape[0]

        list_of_bev_tensor = np.append( list_of_bev_tensor , np.array( [[[0 for i in range(9)] for j in range( max_number_points ) ] for l in range( number_of_padding_tensor )] ).reshape( number_of_padding_tensor , max_number_points , -1 ) , axis = 0 )

        #padding_list_of_bev_tensor  = np.array( [[0 for i in range(3)] for j in range( max_number_pillars - list_of_bev_tensor.shape[0] )])

        #print( "Dimension of padding list of BEV tensor : " + str( padding_list_of_bev_tensor.shape ))
                                               
        list_of_bev_tensor_indices = np.append( list_of_bev_tensor_indices , np.array( [[0 for i in range(3)] for j in range( number_of_padding_tensor )] ), axis = 0)

    list_of_bev_tensor = list_of_bev_tensor.reshape( 1 , max_number_pillars , max_number_points , 9)
    list_of_bev_tensor_indices = list_of_bev_tensor_indices.reshape( 1 , max_number_pillars , 3 )

    print( "Dimension of BEV tensor : " + str( list_of_bev_tensor.shape ))
    print( "Dimension of BEV tensor Index : " + str( list_of_bev_tensor_indices.shape ))
    
    return( list_of_bev_tensor , list_of_bev_tensor_indices )

def visualize_drivable_area_in_image( idx : int , argoverse_data , is_road_segmentation = False):

    # Get matrix rotation of Ego Vehicle Coordinate to City Coordinate
    
    city_to_egovehicle_se3 = argoverse_data.get_pose(idx)

    # Get ego vehicle city coordinate and rotation

    x,y,_ = city_to_egovehicle_se3.translation

    #ego_vehicle_rotation = city_to_egovehicle_se3.rotation

    #yaw_angle = atan2(ego_vehicle_rotation[ 0,0 ], ego_vehicle_rotation[1,0])

    #pose_rotation_matrix_to_yaw = R.from_matrix( ego_vehicle_rotation )

    #yaw_angle = pose_rotation_matrix_to_yaw.as_euler( "zyx" , degrees = True )[0]

    city_name = argoverse_data.city_name

    #log_index = argoverse_data.counter

    # Get Drivable area in Raster Map

    #x_raster_map_coordinate = x - 642
    #y_raster_map_coordinate = y + 211

    drivable_rasterized_map_rotation_matrix = np.array( avmap.get_rasterized_driveable_area( argoverse_data.city_name )[1][ : , : ])#.astype( np.uint8 )

    drivable_rasterized_map = am.get_rasterized_driveable_area(city_name= city_name  )[0]

    x_raster_map_coordinate = int( x + drivable_rasterized_map_rotation_matrix[0][2])
    y_raster_map_coordinate = int( y + drivable_rasterized_map_rotation_matrix[1][2] )

    #print( "Position of ego vehicle is x: " + str( x_raster_map_coordinate ) + " y : " + str( y_raster_map_coordinate )) 

    bev_area_around_vehicle = np.array( [[i , j ] for i in range ( (int(x) -100), ( int(x) + 100), ) for j in range( (int(y) -100) , (int( y ) + 100) ) if (( i >= 1 ) & ( i < drivable_rasterized_map.shape[1]) & ( j>= 1 ) & ( j < drivable_rasterized_map.shape[0] ))] )# if drivable_rasterized_map[ i,j][1] > 0 ])

    # convert list of bev pointr into 4 corner of BEV grids

    bev_corners_around_vehicle = np.array( [[[ i - 0.5 , j - 0.5 ] , [i + 0.5 , j - 0.5 ], [i - 0.5 , j + 0.5 ] , [ i + 0.5 , j+0.5 ]] for [i, j] in bev_area_around_vehicle ]).reshape( -1 , 2 )
    
    #print( "Shape of bev area around vehicle : " + str( bev_area_around_vehicle.shape ))
    
    #bev_area_around_vehicle = [ bev_area_point for bev_area_point in bev_area_around_vehicle[0] if (( bev_area_around_vehicle[ 0 ] >= 0 ) & ( bev_area_around_vehicle[0] < drivable_rasterized_map.shape[1]) & ( bev_area_around_vehicle[1] >= 0 ) & ( bev_area_around_vehicle[1] < drivable_rasterized_map.shape[0] ))]
    #print( "BEV area around vehicle : " + str( bev_corners_around_vehicle ) + " with shape of : " + str( bev_corners_around_vehicle.shape ))
    #print( "Number of drivable area around vehicle : " + str( bev_area_around_vehicle.shape ))

    # Filter BEV point around autonomous vehicle who is drivable area

    #bev_area_around_vehicle = np.array( [bev_point for bev_point in bev_area_around_vehicle if drivable_rasterized_map[ int( bev_point[1]) , int( bev_point[0])] == 1 ])
    
    bev_area_around_vehicle_height = np.array( am.get_ground_height_at_xy( bev_corners_around_vehicle , city_name )).reshape(-1 , 1)

    bev_area_around_vehicle_with_ground_height = np.append( bev_corners_around_vehicle , bev_area_around_vehicle_height , axis = -1)

    bev_area_around_vehicle_points = city_to_egovehicle_se3.inverse_transform_point_cloud(
        bev_area_around_vehicle_with_ground_height
    )

    #bev_area_around_vehicle_points = bev_area_around_vehicle_points[ bev_area_around_vehicle_points[ : , 2 ] <= 0 ]
    #print( "BEV area around vehicle in lidar point is : " + str( bev_area_around_vehicle_points ))

    bev_area_grid_corners_around_vehicle_points = bev_area_around_vehicle_points.reshape( -1 , 4 , 3 )

    if bev_area_grid_corners_around_vehicle_points.shape[ 0 ] == 0 :

        #print( "There is no BEV points around Autonomous Vehicle" )

        return -100 , -100 

    #print( "BEV area grid corners around vehicle points are : " + str( bev_area_grid_corners_around_vehicle_points ))

    try : 

        img, img_for_segmentation = draw_ground_pts_in_image(
                argoverse_data.sync,
                bev_area_around_vehicle_points if not is_road_segmentation else bev_area_grid_corners_around_vehicle_points, #bev_area_grid_corners_around_vehicle_points ,
                city_to_egovehicle_se3,
                am,
                argoverse_data.current_log,
                argoverse_data.lidar_timestamp_list[idx],
                city_name,
                tracking_train_dataset_dir,
                experiment_prefix = "Covert_Drivable_Area_Map_to_Drivable_Area_Image",
                camera = 'ring_front_center',
                is_road_segmentation = is_road_segmentation
            )
        
    except Exception as e :
#1.0
        print( "Error projecting DA to image : " + str(e) )

        img = argoverse_data.get_image( idx , camera= "ring_front_center")

        img_for_segmentation = np.ones( img.shape )

        return -100, -100

        

    return img, img_for_segmentation

def visualize_drivable_undrivable_area_for_ground_height_estimation( idx , argoverse_data , log_id=None, log_index = 0 , radius_lidar_point = 2 , is_show_image_visualization = True ) :

    # Get matrix rotation of Ego Vehicle Coordinate to City Coordinate
    
    city_to_egovehicle_se3 = argoverse_data.get_pose(idx)

    # Get ego vehicle city coordinate and rotation

    x,y,_ = city_to_egovehicle_se3.translation

    ego_vehicle_rotation = city_to_egovehicle_se3.rotation

    yaw_angle = atan2(ego_vehicle_rotation[ 0,0 ], ego_vehicle_rotation[1,0])

    pose_rotation_matrix_to_yaw = R.from_matrix( ego_vehicle_rotation )

    yaw_angle = pose_rotation_matrix_to_yaw.as_euler( "zyx" , degrees = True )[0]

    city_name = argoverse_data.city_name

    drivable_rasterized_map_rotation_matrix = np.array( avmap.get_rasterized_driveable_area( city_name )[1][ : , : ])#.astype( np.uint8 )


    x_raster_map_coordinate = x + drivable_rasterized_map_rotation_matrix[0][2]
    y_raster_map_coordinate = y + drivable_rasterized_map_rotation_matrix[1][2]

    bev_area_around_vehicle = np.array( [[i , j ] for i in range ( int( x)- 100 , int(x) + 100) for j in range( int( y) - 100 , int( y ) + 100)] )# if drivable_rasterized_map[ i,j][1] > 0 ])

    #print( "Shape of BEV area around vehicle is : " + str( bev_area_around_vehicle.shape ))
          
    #print( "Number of drivable area around vehicle : " + str( bev_area_around_vehicle.shape ))
    
    bev_area_around_vehicle_height = np.array( am.get_ground_height_at_xy( bev_area_around_vehicle , city_name )).reshape(-1 , 1)

    #print( "Height of BEV area around vehicle is : " + str( bev_area_around_vehicle_height ))

    bev_area_around_vehicle_with_ground_height = np.append( bev_area_around_vehicle , bev_area_around_vehicle_height , axis = 1)

    #print( "LiDAR points of drivable area around ego- vehicle : " + str( bev_area_around_vehicle_with_ground_height ))

    bev_area_around_vehicle_points = city_to_egovehicle_se3.inverse_transform_point_cloud(
        bev_area_around_vehicle_with_ground_height
    )[: , 2]

    bev_area_around_vehicle_points = np.array( [ 1/(1+ math.exp(-1*bev_area_height ))  if not np.isnan( bev_area_height) else 1 for bev_area_height in bev_area_around_vehicle_points] ).reshape(  200 , 200 )

    #print( "BEV area around vehicle points in lidar coordinate is : " + str( bev_area_around_vehicle_points ) + "\nWith shape of BEV area around vehicle points are : " + str( bev_area_around_vehicle_points.shape ))

    bev_vehicle_height_around_autonomous_vehicle_with_colored_images = np.array( [[[ i*255 , 0 , 0 ] for i in j ] for j in bev_area_around_vehicle_points ] ).astype( np.uint8 ).repeat( 5 , axis = 0 ).repeat( 5 , axis = 1 ) 
    
    img = ImageOps.mirror( Image.fromarray( bev_vehicle_height_around_autonomous_vehicle_with_colored_images ).rotate( yaw_angle + 90))#.resize( 1000 , 1000)

    rotated_bev_image_shape = img.size
    
    img = img.crop( ( rotated_bev_image_shape[0]/2 - 50 * 5 , rotated_bev_image_shape[1]/2 - 70 * 5 , rotated_bev_image_shape[0]/2 + 50 * 5  , rotated_bev_image_shape[1]/2 + 50 * 5))

    bev_vehicle_height_around_autonomous_vehicle_in_image = np.array( img )

    print( "Shape of the image is : "  + str( bev_vehicle_height_around_autonomous_vehicle_in_image.shape ))

    bev_vehicle_height_around_autonomous_vehicle = [[-1*math.log(255/i[0] - 1 ) if i[0] < 255 else 100 for i in j ] for j in bev_vehicle_height_around_autonomous_vehicle_in_image ]
    
    return bev_vehicle_height_around_autonomous_vehicle , bev_vehicle_height_around_autonomous_vehicle_in_image
                                  
    
    #return new_image #img


#def main() :

tracking_train_dataset_dir = "/media/ave/sub4/Ofel/tracking_train1_with_lidar/argoverse-tracking/train1/"#"../../tracking_train1_v1.1/argoverse-tracking/train1/"#"/media/ofel04/66ECDBFDECDBC609/tracking_train1_v1.1/argoverse-tracking/train1/" #'/home/ofel04/Downloads/tracking_train1_v1.1/argoverse-tracking/train1'

av_hd_map_dir = "../map_files/"

argoverse_loader = ArgoverseRoadDetectionLoader( tracking_train_dataset_dir )

max_number_log_extracted = 20 #100 #len( argoverse_loader.log_list )

am = ArgoverseMap( root = av_hd_map_dir )

# Find drivable point in LiDAR point cloud



avmap = ArgoverseMap( root = av_hd_map_dir )


for index_log_argoverse in range( 0 , max_number_log_extracted  ) :

    log_id = argoverse_loader.log_list[ index_log_argoverse ]

    # Create folder for BEV tensor in the log
    
    range_of_bev_tensor = [[200 , 200] , [400,400]]
    
    dict_of_file_bev_tensor = dict()
    
    for bev_tensor_range in range_of_bev_tensor :

	    name_of_bev_tensor_folder =  tracking_train_dataset_dir + "/" + str( log_id ) + "/BEV_tensor_folder_range_{}_{}/".format( bev_tensor_range[0] , bev_tensor_range[1] )

	    os.makedirs( name_of_bev_tensor_folder , exist_ok=True )
	    
	    dict_of_file_bev_tensor[ str( bev_tensor_range ) ] = name_of_bev_tensor_folder

    name_of_bev_drivable_area_label_folder = tracking_train_dataset_dir + "/" + str( log_id ) + "/BEV_drivable_area_label/"

    os.makedirs( name_of_bev_drivable_area_label_folder , exist_ok= True )

    name_of_drivable_area_projection_to_camera_folder = tracking_train_dataset_dir + "/" + str( log_id ) + "/DA_Projection_to_Camera/"

    #name_of_drivable_area_projection_to_camera_folder = tracking_train_dataset_dir + "/" + str( log_id ) + "/DA_Detection_in_Camera/"

    os.makedirs( name_of_drivable_area_projection_to_camera_folder , exist_ok= True )

    name_of_ground_height_bev_map_folder = tracking_train_dataset_dir + "/" + str( log_id ) + "/Ground_Height_BEV_Map/"

    #name_of_drivable_area_projection_to_camera_folder = tracking_train_dataset_dir + "/" + str( log_id ) + "/DA_Detection_in_Camera/"

    os.makedirs( name_of_ground_height_bev_map_folder , exist_ok= True )

    print( "Making BEV tensor for Log Argoverse Dataset : " + str( index_log_argoverse ))

    argoverse_data = argoverse_loader[ index_log_argoverse ]

    number_samples = len( argoverse_loader._lidar_timestamp_list[ str( log_id ) ] )

    for i , frame_argoverse_index in enumerate( range( number_samples )) :

        
        for bev_tensor_range in range_of_bev_tensor :
        
        
            lidar_pts = argoverse_data.get_lidar_in_rasterized_map_coordinate( frame_argoverse_index )

            bev_tensor = convert_lidar_points_to_bev_tensors( lidar_pts , lidar_range = [ -bev_tensor_range[0]/2 , bev_tensor_range[0]/2, -bev_tensor_range[1]/2 , bev_tensor_range[1]/2])

            #print( "List of Argoverse 1 LiDAR data : " + str( argoverse_data._lidar_list ))
		
            name_of_bev_tensor_folder = dict_of_file_bev_tensor[ str( bev_tensor_range ) ]

            name_of_lidar_frame_file = name_of_bev_tensor_folder + str( argoverse_data._lidar_list[ log_id ][ frame_argoverse_index ] ).split( "/")[-1].replace( ".ply" , "" ) + ".pickle"

            with open( name_of_lidar_frame_file , 'wb+') as handle:
            
            	pickle.dump( bev_tensor, handle)

            	print( "Success convert LiDAR points into BEV tensor log {} frame number {} to file : {}".format( index_log_argoverse , frame_argoverse_index , name_of_lidar_frame_file ))
            	print( "-----------------------------------------" )
        
        
        """
        # Makin tensor label for drivable area map

        drivable_area_label = argoverse_data.get_rasterized_drivabel_area_label( key = frame_argoverse_index )

        name_of_drivable_area_label = name_of_bev_drivable_area_label_folder + str( argoverse_data._lidar_list[ log_id ][ frame_argoverse_index ] ).split( "/")[-1].replace( ".ply" , "" ) + ".pickle"

        with open( name_of_drivable_area_label , 'wb+') as handle:
            pickle.dump( drivable_area_label, handle)

        print( "Sum of drivable area is : " + str( np.sum( drivable_area_label )))
        

        print( "Succes creating BEV drivable area label log {} frame number : {} to file : {}".format( index_log_argoverse , frame_argoverse_index , name_of_drivable_area_label ) )
        print( "-------------------------------------------")
        """

        """

        # Making drivable area projection to camera for camera- based DA detection

        img , img_for_segmentation = visualize_drivable_area_in_image( idx = frame_argoverse_index , argoverse_data= argoverse_data , is_road_segmentation= True )

        if isinstance( img , int ) :

            print( "There is no BEV around autonomous vehicle for Log ID : {} and Frame : {}".format( log_id , frame_argoverse_index ))

            continue

        img_for_segmentation = np.array( img_for_segmentation )

        name_of_drivable_area_projection_to_camera = name_of_drivable_area_projection_to_camera_folder + str( argoverse_data._lidar_list[ log_id ][ frame_argoverse_index ]).split( "/" )[ -1 ].replace( ".ply" , "") + ".npy"

        np.save( name_of_drivable_area_projection_to_camera , img_for_segmentation )

        print( "Sum of drivable area is : " + str( np.sum( img_for_segmentation ) )  )

        print( "Success creating cemera drivable area projection label log {} frame number {} to file {}".format( index_log_argoverse , frame_argoverse_index , name_of_drivable_area_projection_to_camera ))

        print( "---------------------------------------------")

        """
        
        """

        # Making Ground Height Map for Point- wise DA detection using LiDAR

        ground_height_drivable_area,_= visualize_drivable_undrivable_area_for_ground_height_estimation( idx = frame_argoverse_index , argoverse_data= argoverse_data  )

        ground_height_drivable_area = np.array( ground_height_drivable_area )

        name_of_ground_heigh_bev_file = name_of_ground_height_bev_map_folder + str( argoverse_data._lidar_list[ log_id ][ frame_argoverse_index ]).split( "/" )[ -1 ].replace( ".ply" , "") + ".npy"

        np.save( name_of_ground_heigh_bev_file , ground_height_drivable_area )

        print( "Number of drivable area is : " + str( ground_height_drivable_area[ ground_height_drivable_area < 100 ].shape )  )

        print( "Success creating ground height label log {} frame number {} to file {}".format( index_log_argoverse , frame_argoverse_index , name_of_ground_heigh_bev_file ))

        print( "---------------------------------------------")

        """



    #name_of_lidar_folder = tracking_train_dataset_dir + "/" + str( log_id ) + "/lidar/"

    argoverse_loader.__next__()

    #os.system( "sudo rm -r {}".format( name_of_lidar_folder )) #os.removedirs( name_of_lidar_folder )

    #print( "Delete LiDAR files in log : " + str( log_id )) 

        

            
