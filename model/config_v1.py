class DefaultConfig_v1():
    #backbone
    pretrained=False
    freeze_stage_1=False
    freeze_bn=False

    #fpn
    fpn_out_channels=256
    use_p5=True
    
    #head
    class_num=1
    use_GN_head=True
    prior=0.01
    add_centerness=True   # fcos v1: True, fcos v2: True
    cnt_on_reg=False # fcos v1: False, fcos v2: True
    sample_radiu_ratio = -1 # fcos v1: -1, fcos v2: default 1.5
    norm_reg_targets = False # fcos v1: False, fcos v2: True

    #loss
    centerness_loss_sqrt = False # fcos v1: False, fcos v2: True
    reg_IoU = "iou" # fcos v1: IoU, fcos v2: giou

    #training
    strides=[8,16,32,64,128]
    limit_range=[[-1,64],[64,128],[128,256],[256,512],[512,999999]]

    #inference
    score_threshold=0.05
    nms_iou_threshold=0.6
    max_detection_boxes_num=1000