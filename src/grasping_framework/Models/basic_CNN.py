'''
Nerual Network architecture
'''
import torch
from torch import nn
from torchvision.models import vgg19_bn, vgg16_bn, inception_v3, alexnet, resnet18, resnet34, resnet50
from torchsummary import summary
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Basic_network(nn.Module):  # after fc, 64 values
    def __init__(self, base_network='vgg_16', pretrained = False, frozen_weights = False, dropout = 0.5, rnn_input_size = 64):
        super(Basic_network, self).__init__()
        # Define CNN to extract features.
        self.features = None # load any specified network
        if base_network == 'vgg_16':
            self.features = vgg16_bn(pretrained=pretrained)  # the initial_value
            # To delete fc8
            self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])  #discard the last two layers in the original nerual network (but keep all the CNN hidden layers)
            for param in self.features.parameters():
                param.requires_grad = not frozen_weights
            # nn.Sequential -> Re-combine all the layers to be a fc classifier
            self.fc = nn.Sequential(nn.Linear(4096*2, rnn_input_size))  # *2 -> combine tactile & visual (this one should be trained)
            # RNN input size: 64
        elif base_network == 'vgg_19':  #three more layers
            self.features = vgg19_bn(pretrained=pretrained)
            # To delete fc8
            self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])
            for param in self.features.parameters():
                param.requires_grad = not frozen_weights
            self.fc = nn.Sequential(nn.Linear(4096*2, rnn_input_size))
        elif base_network == 'resnet18':  
            self.features = resnet18(pretrained=pretrained)
            self.features.fc = nn.Linear(self.features.fc.in_features, 4096)
            for param in self.features.parameters():
                param.requires_grad = not frozen_weights
            self.fc = nn.Sequential(nn.Linear(4096*2, rnn_input_size))
        elif base_network == 'inception_v3':
            #TODO It is unreliable.
            self.features = inception_v3(pretrained=pretrained)
            self.features.aux_logits = True
            # To delete the last layer.
            self.features.fc = nn.Sequential(*list(self.features.fc.children())[:-1])  #
            for param in self.features.parameters():
                param.requires_grad = not frozen_weights
            self.fc = nn.Sequential(nn.Linear(2048*2, rnn_input_size))
        elif base_network == 'debug':
            self.features = alexnet(pretrained=pretrained)  #the first one
            # To delete the last layer
            self.features.classifier = nn.Sequential(*list(self.features.classifier.children())[:-2])
            for param in self.features.parameters():
                param.requires_grad = not frozen_weights
            self.fc = nn.Sequential(nn.Linear(4096*2, rnn_input_size))
        self.dropout_ = nn.Dropout(dropout)
        assert self.features, "Illegal CNN network name!"  # non-defined network

    def forward(self, x1, x2, y1, y2):  #input two images at a time (fusion)
        features_1 = self.features(x1)
        features_2 = self.features(y1)
        features_pinching = torch.cat((features_1, features_2), 1)  # concatenate the two rows  1 -> add along the column
        features_pinching = features_pinching.view(features_pinching.size(0), -1)   #return a tensor of a different shape  -1 > infer the other dimention, which means you can always find a correct shape
        features_pinching = self.fc(features_pinching)
        features_pinching = self.dropout_(features_pinching)
        features_3 = self.features(x2)
        features_4 = self.features(y2)
        features_sliding = torch.cat((features_3, features_4), 1)  # concatenate the two rows  1 -> add along the column
        features_sliding = features_sliding.view(features_sliding.size(0), -1)   #return a tensor of a different shape  -1 > infer the other dimention, which means you can always find a correct shape
        features_sliding = self.fc(features_sliding)
        features_sliding = self.dropout_(features_sliding)
        return features_pinching, features_sliding # (64,1) tensor for LSTM model

# RNN Classifier  This is the model that needs to be trained
class RNN_network(nn.Module):
    def __init__(self, input_size=64, hidden_size=64, num_layers=2, num_embedding=20, use_gpu=False, dropout=0.8):
        # input size 64 -> dimension of input tensor
        # hidden sieze 64 -> dimension of outputs of each layer
        # class -> slip or not (two classes)
        super(RNN_network, self).__init__()  # a subclass of NN.module and forward function is callable.
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.lstm_1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        #batch_size are important -> input CNN feature (batch_size, seq_length, hidden_size)
        #if we do not set batch_size, LSTM would not take the first dim to be batch_size (instead of taking it as seq_length)

        # self.lstm_2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,)
        self.fc = nn.Linear(hidden_size, num_embedding)  # project to 2-label data
        # self.dropout_1 = nn.Dropout(dropout)
        # self.dropout_2 = nn.Dropout(dropout)
        self.h0 = None  #h0 -> init
        self.c0 = None  #c0 -> init

    def forward(self, x):
        #input x dim: batch_size(always to be the first one), seq_length, hidden_size
        # Set initial hidden and cell states
        # len(x) -> batch size == x.size(0)
        # x.size() -> batch_size seq_length(for i range(8)) num_components
        # c is intermediate variable
        # h is hidden output
        # Init to be zero is a common standard
        if self.use_gpu:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to('cuda')
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm_1(x, (h0, c0))  # out: tensor of shape
        # x: (batch_size, seq_length, hidden_size) the last layer's output
        #_ contains both h and c (after training)
        # out = self.dropout_1(x)
        # out, _ = self.lstm_2(x, (h0, c0))
        # out = self.dropout_2(out)
        # Decode the hidden state of the last time step (last sequence: -1)
        # And only use this outputs to compute the gradient for BP
        out = self.fc(out[:, -1, :]) # project to 2-label data
        return out

class Basic_CNN(nn.Module):
    def __init__(self, base_network='vgg_16', pretrained=False, rnn_input_size=64, rnn_hidden_size=64,
                 rnn_num_layers=1, num_classes=3, num_embedding = 20, use_gpu=False, frozen_weights = False, dropout_CNN=0.5, dropout_LSTM=0.8, video_length = 8):  #classes -> slip or not
        super(Basic_CNN, self).__init__()
        self.cnn_network = Basic_network(base_network=base_network, pretrained=pretrained, frozen_weights=frozen_weights, dropout=dropout_CNN, rnn_input_size = rnn_input_size)
        self.rnn_network_pinching = RNN_network(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                                       num_embedding=num_embedding, use_gpu=use_gpu, dropout=dropout_LSTM)
        self.rnn_network_sliding = RNN_network(input_size=rnn_input_size, hidden_size=rnn_hidden_size, num_layers=rnn_num_layers,
                                       num_embedding=num_embedding, use_gpu=use_gpu, dropout=dropout_LSTM)
        self.video_length = video_length
        self.use_gpu = use_gpu  
        self.act_fc1 = nn.Linear(1, 8)  # action(1 * 2; gripper & ratio) -> 1 * 16
        self.act_fc2 = nn.Linear(8, 20)  # action(1 * 2) -> 1 * 16

        self.pred1 = nn.Linear(60, 30)
        self.pred2 = nn.Linear(30, 15)
        #Classifier head
        self.head = nn.Linear(15,num_classes) if num_classes > 0 else nn.Identity()

    # x1 -> visual pinching
    # x2 -> visual sliding
    # y1 -> tactile pinching
    # y2 -> tactile sliding
    def forward(self, x1, x2, y1, y2, thresh):
        # if use_gpu, copy the input data to GPU space
        for i in range(self.video_length):   # get 8 * (64, 1) tensor (a sequence and then fed into LSTM)
            if self.use_gpu:
                thresh = thresh.to('cuda')
                features_pinching, features_sliding = self.cnn_network(x1[:,:,i,:,:].to('cuda'), y1[:,:,i,:,:].to('cuda'), x2[:,:,i,:,:].to('cuda'), y2[:,:,i,:,:].to('cuda'))
            else:
                features_pinching, features_sliding = self.cnn_network(x1[:,:,i,:,:], y1[:,:,i,:,:], x2[:,:,i,:,:], y2[:,:,i,:,:])
            if i == 0:
                cnn_features_pinching = features_pinching.unsqueeze(1)
                cnn_features_sliding = features_sliding.unsqueeze(1)
            else:
                cnn_features_pinching = torch.cat([cnn_features_pinching, features_pinching.unsqueeze(1)], dim=1)
                cnn_features_sliding = torch.cat([cnn_features_sliding, features_sliding.unsqueeze(1)], dim=1)
        # cnn_features = torch.FloatTensor(cnn_features)
        # if self.use_gpu:
        #     cnn_features = cnn_features.to(device)
        # cnn_features = cnn_features.reshape([-1, 8, 64])
        output_pinching = self.rnn_network_pinching(cnn_features_pinching)
        output_sliding = self.rnn_network_sliding(cnn_features_sliding)
        fused = torch.cat((output_pinching, output_sliding), dim=-1)
        embed_act = self.act_fc1(thresh)
        embed_act = F.leaky_relu(embed_act)
        embed_act = self.act_fc2(embed_act)
        # print(pinching.shape)
        # print(sliding.shape)
        # print(fused.shape)
        # print(embed_act.shape)
        pred_input = torch.cat((fused, embed_act), dim = -1)
        predction = F.leaky_relu(self.pred1(pred_input))
        predction = F.leaky_relu(self.pred2(predction))
        predction = self.head(predction)
        return predction

    def init_weights(self):  # no pretrained model (but )
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_init_weights)

