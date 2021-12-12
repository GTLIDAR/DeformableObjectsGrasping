import sys
import functools
import torch.nn as nn
sys.path.append("..")
from Models import swin_transformer, vivit,vivit_two ,vivit_FDP, vivit_FDP_two, basic_CNN, swin_transformer_two, timeSformer_orig, timeSformer_orig_two
# Todo: add other model architecture and put them under ../Models

# Build and return the model here based on the configuration.
def get_model(config_data,use_gpu):
    model_type = config_data['Model_name']
    Image_width = config_data['Image_width']
    Image_height = config_data['Image_height']
    video_length = config_data['video_length']
    Scale_ratio = config_data['scale_ratio']
    resized_width = Image_width * Scale_ratio
    resized_height = Image_height * Scale_ratio
    Model = None
    # You may add more parameters if you want
    # Based on the Image_width, Image_height, Scale_ratio, video_length, some hyperparameter can be computed here
    if model_type == 'vivit_fdp_two':
        img_size  = (int(resized_height),int(resized_width))
        patch_size = (12, 16)
        in_chans = 3
        num_cls = 2
        emb_dim=256
        depth=8
        num_heads=16
        mlp_ratio=4
        dropout = config_data['mlp_drop']
        attn_dropout = config_data['attn_drop']
        num_frames = video_length

        # if some parameters need to be computed, add them here:
        Model = vivit_FDP_two.VIVIT(img_size=img_size, patch_size=patch_size, in_chans=in_chans,num_classes=num_cls, embed_dim=emb_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None, drop_rate=dropout, attn_drop_rate=attn_dropout, drop_path_rate=dropout, num_frames=num_frames, dropout=0.,use_gpu=use_gpu)

        model_params = {'img_size':img_size,'patch_size':patch_size, 'in_chans':in_chans,'num_cls': num_cls, 'emb_dim':emb_dim, 'depth':depth, 'num_heads':num_heads,'mlp_ratio':mlp_ratio, 'dropout':dropout, 'attn_drop_rate':attn_dropout, 'num_frames':num_frames}

    elif model_type == 'basic_CNN':  # the benchmark for comparison
        # The bacis CNN + LSTM (the same one in slip detection paper)
        base_network = config_data['base_network']
        pretrained_ = config_data['pretrained']
        frozen_weights = config_data['frozen_weights']  # False -> all parameters are learnable
        rnn_input_size = 64 # CNN output size
        rnn_hidden_size = 64 # LSTM hidden size
        rnn_num_layers = 1 # LSTM layers
        num_classes = 2 # slip or not
        dropout_CNN = config_data["CNN_drop"] # dropout value for CNN
        dropout_LSTM = config_data["LSTM_drop"] # dropout value for LSTM
        Model = basic_CNN.Basic_CNN(base_network = base_network, pretrained = pretrained_, rnn_input_size = rnn_input_size,
                         rnn_hidden_size = rnn_hidden_size, rnn_num_layers = rnn_num_layers, num_classes = num_classes,
                          use_gpu=use_gpu, frozen_weights=frozen_weights, dropout_CNN = dropout_CNN, dropout_LSTM = dropout_LSTM, video_length = video_length)
        model_params = {'base_network': base_network, 'rnn_input_size': rnn_input_size, 'rnn_hidden_size': rnn_hidden_size, 'rnn_num_layers': rnn_num_layers,
                        'num_classes': num_classes, 'dropout_CNN': dropout_CNN, 'dropout_LSTM': dropout_LSTM}

    elif model_type == 'timeSformer_orig_two':
        img_size  = (int(resized_height),int(resized_width))
        patch_size = (12, 16)
        in_chans = 3
        num_cls = 2
        emb_dim=256
        depth=8
        num_heads=16
        mlp_ratio=4
        dropout = config_data['mlp_drop']
        attn_dropout = config_data['attn_drop']
        num_frames = video_length
        # if some parameters need to be computed, add them here:
        Model = timeSformer_orig_two.TimeSFormer(img_size=img_size, patch_size=patch_size, in_chans=in_chans,num_classes=num_cls, embed_dim=emb_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=False, qk_scale=None, drop_rate=dropout, attn_drop_rate=attn_dropout, drop_path_rate=dropout,hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=num_frames, attention_type='divided_space_time', dropout=0.,use_gpu=use_gpu)
        model_params = {'img_size':img_size,'patch_size':patch_size, 'in_chans':in_chans,'num_cls': num_cls, 'emb_dim':emb_dim, 'depth':depth, 'num_heads':num_heads,'mlp_ratio':mlp_ratio, 'dropout':dropout, 'attn_drop_rate':attn_dropout, 'num_frames':num_frames}

    else:
        raise NotImplementedError("Model Factory Not Implemented")

    # Find total parameters and trainable parameters in the model
    total_params = sum(p.numel() for p in Model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in Model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    return Model, model_params
