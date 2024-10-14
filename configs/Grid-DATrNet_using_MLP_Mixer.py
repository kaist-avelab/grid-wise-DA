'''
* Copyright (c) AVELab, KAIST. All rights reserved.
* author: Donghee Paek & Kevin Tirta Wijaya, AVELab, KAIST
* e-mail: donghee.paek@kaist.ac.kr, kevin.tirta@kaist.ac.kr
'''
seed = 2024#2021
load_from = None
finetune_from = None
log_dir = './logs'
view = False

experiment_name = "Training_Road_Detection_Argoverse_Dataset_using_Feature_Extractor_MLP_Mixer"

is_visualize_global_attention = False

GPUS_EN = '2'

gpus_ids =  GPUS_EN
is_save_visualization = True 


net = dict(
    type='Detector',
    head_type= "seg" , #'row',
    loss_type='row_ce'
)

#max_number_log = 23 #Maximum number log used for make training Road Detection in Argoverse 1 Dataset faster

pcencoder = dict(
    type='PointPillars',
    max_points_per_pillar = 32,
    num_features = 9,
    num_channels = 64,
    Xn = 1200,
    Yn = 1000
    #resnet='resnet34',
    #pretrained=False,
    #replace_stride_with_dilation=[False, True, False],
    #out_conv=True,
    #in_channels=[64, 128, 256, -1]
)
featuremap_out_channel = 64

filter_mode = 'xyz'
list_filter_roi = [-50, 70, -50, -50, -2.0, 1.5] #[0.02, 46.08, -11.52, 11.52, -2.0, 1.5]  # get rid of 0, 0 points
list_roi_xy = [ -50, 70, -50, -50]#[0.0, 46.08, -11.52, 11.52]
list_grid_xy = [ 0.2, 0.2 ]#[0.04, 0.02]
list_img_size_xy = [1200 , 1000]#[1152, 1152]

backbone = dict(
    type='MixSegNet', # GFC-T
    image_size=200, #[240,200],
    patch_size=8, #8,
    channels=64,
    dim=512,
    depth=3,
    #heads=16,
    output_channels=1024,
    expansion_factor=4,
    #dim_head=64,
    dropout=0.,
    #emb_dropout=0., # TBD
    #is_with_shared_mlp=True,
    is_using_convolution_pooling = True
)

"""
heads = dict(
    type='RowSharNotReducRef',
    dim_feat=8, # input feat channels
    row_size=200, #144,
    dim_shared=512,
    lambda_cls=1.,
    thr_ext = 0.3,
    off_grid = 2,#2,
    dim_token = 1024,
    tr_depth = 1,
    tr_heads = 16,
    tr_dim_head = 64,
    tr_mlp_dim = 2048,
    tr_dropout = 0.,
    tr_emb_dropout = 0.,
    is_reuse_same_network = False,
    number_class_target = 1
)
"""

heads = dict( 
    type = "GridSeg",
    num_1 = 1024,
    num_2 = 2048,
    num_classes = 2,
    focal_loss_alpha = 0.25,
    focal_loss_gamma = 2, #2,
    tensorboard_dir = "./tensorboard_dir",
    image_size = [600,500]
)

conf_thr = 0.5
view = True

# BGR Format to OpenCV
cls_lane_color = [
    (0, 0, 255),
    (0, 50, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 0, 100)
]

optimizer = dict(
  type = 'Adam', #'AdamW',
  lr = 0.0001,
)

epochs = 35
batch_size =4 #
total_iter = (13122 // batch_size) * epochs
scheduler = dict(
    type = 'CosineAnnealingLR',
    T_max = total_iter
)

image_size = [600, 500] #Image of the output of Road Detection

eval_ep = 1
save_ep = 1

### Setting Here ###
dataset_path = "../tracking_train1_v1.1/argoverse-tracking/train1"#'./data/KLane' # '/media/donghee/HDD_0/KLane'
### Setting Here ###
dataset_type = "ArgoverseRoadDetection"#'KLane'
dataset = dict(
    train=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='train',
        mode_item='pillar',
        max_number_of_log = 20, #None, # Max number of log training for make training Argoverse 1 Dataset in Road Detection faster
        number_batch = 8
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_path,
        split='test',
        mode_item='pillar',
        max_number_of_log = 10 ,#2
        is_random_log = False ,
        number_batch = 1,
        is_eval_visualization_dataset = True, 
        LIST_OF_LOG_FOR_VISUALIZATION = [ "e17eed4f-3ffd-3532-ab89-41a3f24cf226" ,
                                "b3def699-884b-3c9e-87e1-1ab76c618e0b" ,
                                "af706af1-a226-3f6f-8d65-b1f4b9457c48",
                                "11953248-1195-1195-1195-511954366464",
                                "64c12551-adb9-36e3-a0c1-e43a0e9f3845"
                                ]

    )
)
workers=12 #12

# For evaluating Road Detection
is_visualized_result = True
