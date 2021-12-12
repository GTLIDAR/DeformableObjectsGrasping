import os
import sys
import time
import yaml
import json
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

from utils import new_data_loader
from utils import model_factory
# from utils import model_factory_single
from utils import log_record

cuda_avail = torch.cuda.is_available()


def plot_loss_accuracy(train_loss, test_loss, train_acc, test_acc, save_path, colors,
                       loss_legend_loc='upper center', acc_legend_loc='upper left',
                       fig_size=(20, 10), sub_plot1=(1, 2, 1), sub_plot2=(1, 2, 2)):
    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()

    plt.subplot(sub_plot1[0], sub_plot1[1], sub_plot1[2])

    t_loss = np.array(train_loss)
    v_loss = np.array(test_loss)
    x_train = range(t_loss.size)
    x_val = range(v_loss.size)

    min_train_loss = t_loss.min()

    min_val_loss = v_loss.min()

    plt.plot(x_train, train_loss, linestyle='-', color='tab:{}'.format(colors[0]),
             label="TRAIN LOSS ({0:.4})".format(min_train_loss))
    plt.plot(x_val, test_loss, linestyle='--', color='tab:{}'.format(colors[0]),
             label="TEST LOSS ({0:.4})".format(min_val_loss))

    plt.xlabel('epoch no.')
    plt.ylabel('loss')
    plt.legend(loc=loss_legend_loc)
    plt.title('Training and Test Loss')

    plt.subplot(sub_plot2[0], sub_plot2[1], sub_plot2[2])

    t_acc = np.array(train_acc)
    v_acc = np.array(test_acc)
    x_train = range(t_acc.size)
    x_val = range(v_acc.size)

    max_train_acc = t_acc.max()
    max_val_acc = v_acc.max()

    plt.plot(x_train, train_acc, linestyle='-', color='tab:{}'.format(colors[0]),
             label="TRAIN ACC ({0:.4})".format(max_train_acc))
    plt.plot(x_val, test_acc, linestyle='--', color='tab:{}'.format(colors[0]),
             label="VALID ACC ({0:.4})".format(max_val_acc))

    plt.xlabel('epoch no.')
    plt.ylabel('accuracy')
    plt.legend(loc=acc_legend_loc)
    plt.title('Training and Test Accuracy')

    file_path = os.path.join(save_path, 'loss_acc_plot.png')
    fig.savefig(file_path)

    return


def train_net(params):
    # Determine whether to use GPU
    if params['use_gpu'] == 1:
        print("GPU:" + str(params['use_gpu']))

    if params['use_gpu'] == 1 and cuda_avail:
        print("use_gpu=True and Cuda Available. Setting Device=CUDA")
        device = torch.device("cuda:0")  # change the GPU index based on the availability
        use_gpu = True
    else:
        print("Setting Device=CPU")
        device = torch.device("cpu")
        use_gpu = False

    # Create dir to save model and other training artifacts
    if 'save_dir' in params.keys():
        model_save_dir = os.path.join(params['save_dir'], params['Model_name'])
        if (os.path.exists(model_save_dir) == False):
            os.mkdir(model_save_dir)
        dt = datetime.now()
        dt_string = dt.strftime("%d_%m_%Y__%H_%M_%S")
        exp_save_dir = os.path.join(model_save_dir, dt_string)
        os.mkdir(exp_save_dir)

        # create a log file to record the terminal info
        log_record.create_log(exp_save_dir)

        # Save config used for this experiment
        yaml_path = os.path.join(exp_save_dir, 'config.yaml')
        with open(yaml_path, 'w') as outfile:
            yaml.dump(params, outfile, default_flow_style=False)

    # Set seed
    if params['use_random_seed'] == 0:
        torch.manual_seed(params['seed'])

    # Create network & Init Layer weights
    if params['Modality'] == "Combined":
        NN_model, model_params = model_factory.get_model(params, use_gpu)
    
    # Save model params used for this experiment
    if 'save_dir' in params.keys():
        model_params_path = os.path.join(exp_save_dir, 'model_params.json')
        with open(model_params_path, 'w') as outfile:
            json.dump(model_params, outfile)

    if params['skip_init_in_train'] == 0:
        NN_model.init_weights()
    # TODO: Load previous model, if any

    # Init optimizer & loss func.
    loss_function = nn.CrossEntropyLoss()
    if use_gpu:
        NN_model = NN_model.cuda()
        loss_function = loss_function.cuda()
    optimizer = optim.Adam(NN_model.parameters(), lr=float(params['lr']))

    # Dataloader
    train_dataset = new_data_loader.Tactile_Vision_dataset(params['Fruit_type'], params["Tactile_scale_ratio"], params["Visual_scale_ratio"], params["video_length"], data_path=params['Train_data_dir'])
    test_dataset = new_data_loader.Tactile_Vision_dataset(params['Test_type'], params["Tactile_scale_ratio"], params["Visual_scale_ratio"], params["video_length"], data_path=params['Train_data_dir'])
    train_data_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])
    test_data_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'])

    # Test a single feed-forward only process (uncomment this block if you pass the test)
    data = next(iter(train_data_loader))
    print("Print feed-forward test results:")
    # print(data[0].shape) # visual pinching
    # print(data[1].shape) # visual sliding
    # print(data[2].shape) # tactile pinching
    # print(data[3].shape) # tactile sliding
    # print(data[4])  # label
    # print(data[5].shape)  # threshold
    if params['Modality'] == "Combined":
        output = NN_model(data[0], data[1], data[2], data[3], data[5])
    _, predicted = torch.max(output.data, 1)
    print(output)  # Transformer output
    print(predicted)  # prediction
    print(data[4])  # Label data
    print("Pass the feed-forward test!")
    # sys.exit(0)
    # Test a single feed-forward only process
    # To record training procession
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    # Start training
    # NN_model.train()
    t_start = time.time()
    for epoch in range(params['epochs']):
        # Start
        train_total_loss = 0.0
        train_total_acc = 0.0
        train_total = 0.0
        test_total_loss = 0.0
        test_total_acc = 0.0
        test_total = 0.0 
        NN_model.train()
        for i, data in enumerate(train_data_loader):
            NN_model.zero_grad()
            # print(data[0].shape) # visual pinching
            # print(data[1].shape) # visual sliding
            # print(data[2].shape) # tactile pinching
            # print(data[3].shape) # tactile sliding
            # print(data[4])  # label
            # print(data[5].shape)  # threshold
            # one iteration
            label = data[4]
            if params['Modality'] == "Combined":
                output = NN_model(data[0], data[1], data[2], data[3], data[5])
            if use_gpu:
                label = label.to('cuda')
            loss = loss_function(output, label)
            # Backward & optimize
            loss.backward()
            optimizer.step()  # update the parameters
            # cal training acc
            _, predicted = torch.max(output.data, 1)
            train_total_acc += (predicted == label).sum().item()
            train_total_loss += float(loss.data)
            train_total += len(label)
        train_loss.append(train_total_loss / train_total)
        train_acc.append(train_total_acc / train_total)
        elapsed_time = time.time() - t_start
        speed_epoch = elapsed_time / (epoch + 1)
        speed_batch = speed_epoch / len(train_data_loader)
        eta = speed_epoch * params['epochs'] - elapsed_time
        
        NN_model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_data_loader):
                label = data[4]
                if params['Modality'] == "Combined":
                    output = NN_model(data[0], data[1], data[2], data[3], data[5])
                if use_gpu:
                    label = label.to('cuda')
                loss = loss_function(output, label)                
                _, predicted = torch.max(output.data, 1)
                test_total_acc += (predicted == label).sum().item()
                test_total_loss += float(loss.data)
                test_total += len(label)
        test_loss.append(test_total_loss / test_total)
        test_acc.append(test_total_acc / test_total)

        if epoch % params['print_interval'] == 0:
            message = '[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f, Test Loss: %.3f, Test Acc: %.3f' % (
            epoch, params['epochs'], train_loss[epoch], train_acc[epoch], test_loss[epoch], test_acc[epoch])
            log_record.update_log(exp_save_dir, message)
            print(message)
            message = "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
            elapsed_time, speed_epoch, speed_batch, eta)
            log_record.update_log(exp_save_dir, message)
            print(message)

        if epoch % 10 == 0 or test_acc[epoch] > 0.50:
            model_path = os.path.join(exp_save_dir + '/' + params['Model_name'] + str(epoch) + '.pt'.format(epoch))
            state_dict = {'model': NN_model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(state_dict, model_path)
            plot_loss_accuracy(train_loss, test_loss, train_acc, test_acc, exp_save_dir, colors=['blue'],
                               loss_legend_loc='upper center', acc_legend_loc='upper left')

    if 'save_dir' in params.keys():
        model_path = os.path.join(exp_save_dir + '/' + params['Model_name'] + '_last.pt'.format(epoch))
        state_dict = {'model': NN_model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state_dict, model_path)
        plot_loss_accuracy(train_loss, test_loss, train_acc, test_acc, exp_save_dir, colors=['blue'],
                           loss_legend_loc='upper center', acc_legend_loc='upper left')
