import os 
import sys
import yaml
import time

import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data.dataloader import DataLoader

from utils import data_loader
from utils import model_factory
from utils import model_factory_single

cuda_avail = torch.cuda.is_available()

def test_net(params, NN_model):
    #Determine whether to use GPU
    if cuda_avail:
        print("Cuda Available. Setting Device=CUDA")
        device = torch.device("cuda:0")
    else:
        print("Setting Device=CPU")
        device = torch.device("cpu")

    # Init loss func.
    loss_function = nn.CrossEntropyLoss()
    if cuda_avail:
        NN_model.cuda()
        loss_function = loss_function.cuda()

    # Dataloader
    test_dataset = data_loader.Tactile_Vision_dataset(params["scale_ratio"], params["video_length"], data_path=params['Test_data_dir'])
    test_data_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True,
                                  num_workers=params['num_workers'])

    # do one feed-forward on the test dataset
    test_total_loss = 0.0
    test_total_acc = 0.0
    test_total = 0.0
    NN_model.eval()
    start_time = time.time()
    with torch.no_grad():
        for rgb_imgs, tactile_imgs, label in test_data_loader:
            if config_loaded['Modality'] == "Combined":
                output = NN_model(rgb_imgs, tactile_imgs)
            elif config_loaded['Modality'] == "Visual":
                output = NN_model(rgb_imgs)
            elif config_loaded['Modality'] == "Tactile":
                output = NN_model(tactile_imgs)
            if cuda_avail:
                label = label.cuda()
            # print("Elapsed time: %f" % (time.time() - start_time))
            loss = loss_function(output, label)
            _, predicted = torch.max(output.data, 1)
            test_total_acc += (predicted == label).sum().item()
            test_total_loss += float(loss.data)
            test_total += len(label)
    test_total_loss = test_total_loss / test_total
    test_total_acc = test_total_acc / test_total
    print('Test Loss: %.3f, Test Acc: %.3f' % (test_total_loss, test_total_acc))
    print("Elapsed time: %f" % (time.time() - start_time))

if __name__ == "__main__":
    model_name = sys.argv[1] # load model  (*.pt)
    yaml_file = sys.argv[2] # specify the yaml file, corresponding to the model_name
    if os.path.exists(yaml_file):
        with open(yaml_file) as stream:
            config_loaded = yaml.safe_load(stream)
    else:
        print("The yaml file does not exist!")
        sys.exit()
    # Todo, read the model_params.json to specify the model parameters
    if config_loaded['Modality'] == "Combined":
        NN_model, model_params = model_factory.get_model(config_loaded, cuda_avail)
    elif config_loaded['Modality'] == "Visual" or config_loaded['Modality'] == "Tactile":
        NN_model, model_params = model_factory_single.get_model(config_loaded, cuda_avail)
    if cuda_avail:
        NN_model.load_state_dict(torch.load(model_name)['model'])
    else:
        NN_model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'))['model'])
    test_net(config_loaded, NN_model)


