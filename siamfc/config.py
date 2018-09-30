
class Config:
    # dataset related
    exemplar_size = 127
    instance_size = 255
    context_amount = 0.5
    proc_num_thread = 16

    # training related
    sample_pairs_per_video = 14
    train_ratio = 0.9
    frame_range = 50
    train_batch_size = 8
    valid_batch_size = 8
    train_num_workers = 16
    valid_num_workers = 8
    lr = 1e-2
    momentum = 0.9
    weight_decay = 5e-4
    step_size = 1
    gamma = 0.8685
    epoch = 50
    seed = 3214
    log_dir = './models/logs'
    radius = 16
    response_scale = 1e-3
    stretch=True
    scale=True


    # tracking related
    interp_method = 'bicubic'
    scale_step = 1.0375
    num_scale = 3
    scale_lr = 0.59
    response_up_stride = 16
    response_sz = 17
    train_response_sz = 15
    window_influence = 0.176
    scale_penalty = 0.9745
    total_stride = 8
    model_path = '/home/zhangfangyi/codes/SiamFC-PyTorch/models/siamfc.pth'
    gpu_id = 0

    """
    # tracking related
    interp_method = 'bicubic'
    scale_step = 1.02
    num_scale = 3
    scale_lr = 0.59
    response_up_stride = 16
    response_sz = 17
    window_influence = 0.1
    scale_penalty = 0.95
    total_stride = 8
    """


config = Config()
