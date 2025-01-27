# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>
"""OpenCV plotting utility functions."""

from typing import List, Tuple

import cv2
import numpy as np

from .calibration import CameraConfig, proj_cam_to_uv
from .frustum_clipping import clip_segment_v3_plane_n

# For drawing road segmentation for road detection
from shapely.geometry import Polygon


def add_text_cv2(img: np.ndarray, text: str, x: int, y: int, color: Tuple[int, int, int], thickness: int = 3) -> None:
    """Add text to image using OpenCV. Color should be BGR order"""
    img = cv2.putText(
        img,
        text,
        (x, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=thickness,
        color=color,
        lineType=cv2.LINE_AA,
    )


def draw_clipped_line_segment(
    img: np.ndarray,
    vert_a: np.ndarray,
    vert_b: np.ndarray,
    camera_config: CameraConfig,
    linewidth: int,
    planes: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    color: Tuple[int, int, int],
) -> None:
    """Plot the portion of a line segment that lives within a parameterized 3D camera frustum.

    Args:
        img: Numpy array of shape (M,N,3)
        vert_a: first point, in the camera coordinate frame.
        vert_b: second point, in the camera coordinate frame.
        camera_config: CameraConfig object
        linewidth: integer, linewidth for plot
        planes: frustum clipping plane parameters
        color: RGB 3-tuple
    """
    clip_vert_a, clip_vert_b = clip_segment_v3_plane_n(vert_a.copy(), vert_b.copy(), planes.copy())
    if clip_vert_a is None or clip_vert_b is None:
        return

    uv_a, _, _, _ = proj_cam_to_uv(clip_vert_a.reshape(1, 3), camera_config)
    uv_b, _, _, _ = proj_cam_to_uv(clip_vert_b.reshape(1, 3), camera_config)

    uv_a = uv_a.squeeze()
    uv_b = uv_b.squeeze()
    cv2.line(
        img,
        (int(uv_a[0]), int(uv_a[1])),
        (int(uv_b[0]), int(uv_b[1])),
        color,
        linewidth,
    )


def draw_point_cloud_in_img_cv2(img: np.ndarray, xy: np.ndarray, colors: np.ndarray, radius: int = 5 , draw_road_segmentation = False ) -> np.ndarray:
    """Plot a point cloud in an image by drawing small circles centered at (x,y) locations.

    Note these are not (u,v) but rather (v,u) coordinate pairs.

    Args:
        img: Array of shape (M, N, 3), representing an image with channel order BGR, per the OpenCV convention
        xy: Array of shape (K, 2) representing the center coordinates of each circle
        colors: Array of shape (K, 3), with BGR values in [0, 255] representing the fill color for each circle
        radius: radius of all circles

    Returns:
        img: Array of shape (M, N, 3), with all circles plotted
    """
    list_of_highest_area_with_road = [ None for i in range( img.shape[ 1 ])]

    
    #print( "Image Shape : " + str( img.shape ))

    if draw_road_segmentation == False :
    
        for i, (x, y) in enumerate(xy):
            rgb = colors[i]
            rgb = tuple([int(intensity) for intensity in rgb])
            img = cv2.circle(img, (x, y), radius, tuple(rgb), -1)
            
            
            if draw_road_segmentation == True : 
                #print( x )
                if  ( list_of_highest_area_with_road[ y ] is None ) :
                    list_of_highest_area_with_road[ y ] = x
                
            
                elif  ( x < list_of_highest_area_with_road[ y ] ) :
                    list_of_highest_area_with_road[ y ] = x 
            
        #print( "Image of Shape is : " + str( img.shape) )
        if draw_road_segmentation == True :
            # Make area road with color Green
            for x_axis , highest_y_value in enumerate( list_of_highest_area_with_road ) :
                if highest_y_value is not None :
                    img[ highest_y_value : ][ x_axis] = [ 0 , 255 , 0 ]

    else :
        # Draw drivable area grid instead of drivable area points

        img_for_segmentation = np.ascontiguousarray( np.zeros( img.shape )).astype( "uint8" ).copy()

        #print( "Shape of autonomous vehicle image is : " + str( img.shape ) + " and shape of image for segmentation is : " + str( img_for_segmentation.shape ))

        xy = xy.reshape( -1, 4 , 2 )

        #print( "List of drivable area grid corners are : {} with Maximum : {} and Minimum : {}".format( xy , xy[ : , : , 0 ].max() , xy[ : , : , 1 ].max()))

        height_image , width_image = img.shape[ : 2 ]

        #print( "Shape of the image is width image : " + str( width_image ) + " and height image : " + str( height_image ))

        # Filter point in the image

        #xy = xy[ ( xy[ : , : , 0 ] >= 0 ) & ( xy[ : , : , 0 ] <= width_image ) & ( xy[ : , : , 1 ] >= 0 ) & ( xy[ : , : , 1 ] <= height_image ) ]

        xy = np.array( [ grid_corners for grid_corners in xy if ( grid_corners[ : , 0 ].max() >= 0 ) & ( grid_corners[ : , 0 ].min() <= width_image ) & ( grid_corners[ : , 1 ].max() >= 0 ) & ( grid_corners[ : , 1 ].min() <= height_image) & ( grid_corners[ : , 1 ].min() >= height_image/2) ])

        xy[ : , : , 0 ] = np.clip(xy[ : , : , 0 ] , 0 , width_image )

        xy[ : , : , 1 ] = np.clip(xy[ : , : , 1 ] , 0 , height_image )

        #print( "List of drivable area grids after filtering : " + str( xy ) + " with shape :  " + str( xy.shape ))

        for i, (top_left , top_right , below_left , below_right ) in enumerate(xy):
            #rgb = colors[i]
            #rgb = tuple([int(intensity) for intensity in rgb])

            polygon = Polygon([tuple( top_left ), tuple( top_right ), tuple( below_right ), tuple( below_left )])
            int_coords = lambda x: np.array(x).round().astype(np.int32)
            exterior = [int_coords(polygon.exterior.coords)]

            img = cv2.fillPoly(img.copy(), exterior, color=(0, 255, 0))

            img_for_segmentation = cv2.fillPoly(img_for_segmentation.copy().astype( img.dtype ), exterior, color=(255, 255, 255))

            img_for_segmentation = np.array( img_for_segmentation )

        # Change drivable area label to drivable area label or not
        img_for_segmentation = np.array( [[0 if pixel_image_value.sum() == 0 else 1 for pixel_image_value in i] for i in img_for_segmentation ])


            #img = cv2.circle(img, (x, y), radius, tuple(rgb), -1)

            # Return image with drivable area and image with drivable area segmentation

        
        return img, img_for_segmentation 

            
        """
        if draw_road_segmentation == True :
            # Make area road with color Green
            for x_axis , highest_y_value in enumerate( list_of_highest_area_with_road ) :
                if highest_y_value is not None :
                    img[ highest_y_value : ][ x_axis] = [ 0 , 255 , 0 ]
        """
                
                        	       
    return img


def draw_polyline_cv2(
    line_segments_arr: np.ndarray,
    image: np.ndarray,
    color: Tuple[int, int, int],
    im_h: int,
    im_w: int,
) -> None:
    """Draw a polyline onto an image using given line segments.

    Args:
        line_segments_arr: Array of shape (K, 2) representing the coordinates of each line segment
        image: Array of shape (M, N, 3), representing a 3-channel BGR image
        color: Tuple of shape (3,) with a BGR format color
        im_h: Image height in pixels
        im_w: Image width in pixels
    """
    for i in range(line_segments_arr.shape[0] - 1):
        x1 = line_segments_arr[i][0]
        y1 = line_segments_arr[i][1]
        x2 = line_segments_arr[i + 1][0]
        y2 = line_segments_arr[i + 1][1]

        x_in_range = (x1 >= 0) and (x2 >= 0) and (y1 >= 0) and (y2 >= 0)
        y_in_range = (x1 < im_w) and (x2 < im_w) and (y1 < im_h) and (y2 < im_h)

        if x_in_range and y_in_range:
            # Use anti-aliasing (AA) for curves
            image = cv2.line(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_polygon_cv2(points: np.ndarray, image: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Draw a polygon onto an image using the given points and fill color.

    These polygons are often non-convex, so we cannot use cv2.fillConvexPoly().
    Note that cv2.fillPoly() accepts an array of array of points as an
    argument (i.e. an array of polygons where each polygon is represented
    as an array of points).

    Args:
        points: Array of shape (N, 2) representing all points of the polygon
        image: Array of shape (M, N, 3) representing the image to be drawn onto
        color: Tuple of shape (3,) with a BGR format color

    Returns:
        image: Array of shape (M, N, 3) with polygon rendered on it
    """
    points = np.array([points])
    points = points.astype(np.int32)
    image = cv2.fillPoly(image, points, color)  # , lineType[, shift]]) -> None
    return image


def plot_bbox_polygon_cv2(img: np.ndarray, track_id: str, color: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Draw a colored bounding box with a red border.

    We use OpenCV's rectangle rendering to draw the thin red border.

    Args:
        img: Array of shape (M,N,3) represnenting the image to plot the bounding box onto
        track_id: The track id to use as a label
        color: Numpy Array of shape (3,) with a BGR format color
        bbox: Numpy array, containing values xmin, ymin, xmax, ymax. Note that the requested color is placed in
            xmax-1 and ymax-1, but not beyond. in accordance with Numpy indexing implementation).
            All values on the border (touching xmin, or xmax, or ymin, or ymax along an edge) will be colored red.

    Returns:
        img: Array of shape (M, N, 3)
    """
    xmin, ymin, xmax, ymax = bbox.astype(np.int32).squeeze()

    bbox_h, bbox_w, _ = img[ymin:ymax, xmin:xmax].shape
    tiled_color = np.tile(color.reshape(1, 1, 3), (bbox_h, bbox_w, 1))
    img[ymin:ymax, xmin:xmax, :] = (img[ymin:ymax, xmin:xmax, :] + tiled_color) / 2.0

    white = (255, 255, 255)
    plot_x = xmin + 10
    plot_y = ymin + 25
    img = cv2.putText(
        img,
        str(track_id),
        (plot_x, plot_y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        thickness=5,
        color=white,
    )

    red = (255, 0, 0)
    # Use the default thickness value of 1. Negative values, like CV_FILLED, would provide a filled rectangle instead.
    img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=red)
    return img


def get_img_contours(img: np.ndarray) -> np.ndarray:
    """
    Uses

    Ref: Suzuki, S. and Abe, K., Topological Structural Analysis of Digitized Binary Images
    by Border Following. CVGIP 30 1, pp 32-46 (1985)

    Args:
        img: binary image with zero and one values

    Returns:
        contours: Numpy array
    """
    imgray = img.copy() * 255
    threshold_val = 127
    max_binary_val = 255
    ret, thresh = cv2.threshold(imgray, threshold_val, max_binary_val, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
