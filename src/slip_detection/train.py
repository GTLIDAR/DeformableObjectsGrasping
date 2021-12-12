import os
import sys
import time
import yaml
import json
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import numpy as np
import torch
from torch import optim
from torch import nn
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

from utils import data_loader
from utils import model_factory
from utils import model_factory_single
from utils import log_record

cuda_avail = torch.cuda.is_available()


def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, save_path, colors,
                       loss_legend_loc='upper center', acc_legend_loc='upper left',
                       fig_size=(20, 10), sub_plot1=(1, 2, 1), sub_plot2=(1, 2, 2)):
    plt.rcParams["figure.figsize"] = fig_size
    fig = plt.figure()

    plt.subplot(sub_plot1[0], sub_plot1[1], sub_plot1[2])

    t_loss = np.array(train_loss)
    v_loss = np.array(val_loss)
    x_train = range(t_loss.size)
    x_val = range(v_loss.size)

    min_train_loss = t_loss.min()

    min_val_loss = v_loss.min()

    plt.plot(x_train, train_loss, linestyle='-', color='tab:{}'.format(colors[0]),
             label="TRAIN LOSS ({0:.4})".format(min_train_loss))
    plt.plot(x_val, val_loss, linestyle='--', color='tab:{}'.format(colors[0]),
             label="VALID LOSS ({0:.4})".format(min_val_loss))

    plt.xlabel('epoch no.')
    plt.ylabel('loss')
    plt.legend(loc=loss_legend_loc)
    plt.title('Training and Validation Loss')

    plt.subplot(sub_plot2[0], sub_plot2[1], sub_plot2[2])

    t_acc = np.array(train_acc)
    v_acc = np.array(val_acc)
    x_train = range(t_acc.size)
    x_val = range(v_acc.size)

    max_train_acc = t_acc.max()
    max_val_acc = v_acc.max()

    plt.plot(x_train, train_acc, linestyle='-', color='tab:{}'.format(colors[0]),
             label="TRAIN ACC ({0:.4})".format(max_train_acc))
    plt.plot(x_val, val_acc, linestyle='--', color='tab:{}'.format(colors[0]),
             label="VALID ACC ({0:.4})".format(max_val_acc))

    plt.xlabel('epoch no.')
    plt.ylabel('accuracy')
    plt.legend(loc=acc_legend_loc)
    plt.title('Training and Validation Accuracy')

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
    elif params['Modality'] == "Tactile" or params['Modality'] == "Visual":
        NN_model, model_params = model_factory_single.get_model(params, use_gpu)
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
    train_dataset = data_loader.Tactile_Vision_dataset(params["scale_ratio"], params["video_length"],
                                                       data_path=params['Train_data_dir'])
    train_data_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                   num_workers=params['num_workers'])

    valid_dataset = data_loader.Tactile_Vision_dataset(params["scale_ratio"], params["video_length"],
                                                       data_path=params['Valid_data_dir'])
    valid_data_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=True,
                                   num_workers=params['num_workers'])

    test_dataset = data_loader.Tactile_Vision_dataset(params["scale_ratio"], params["video_length"],
                                                      data_path=params['Test_data_dir'])
    test_data_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True,
                                  num_workers=params['num_workers'])

    # Test a single feed-forward only process (uncomment this block if you pass the test)
    data = next(iter(train_data_loader))
    print("Print feed-forward test results:")
    # Camera images: data[0].shape ->  torch.Size([4, 3, 8, 240, 320])  batch_size, channel (RGB), depth, height, width
    # Tactile images: data[1].shape ->  torch.Size([4, 3, 8, 240, 320])  batch_size, channel (RGB), depth, height, width
    # labels: data[2].shape ->  torch.Size([4])  batch_size
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    if params['Modality'] == "Combined":
        output = NN_model(data[0], data[1])
    elif params['Modality'] == "Visual":
        output = NN_model(data[0])
    elif params['Modality'] == "Tactile":
        output = NN_model(data[1])
    _, predicted = torch.max(output.data, 1)
    print(output)  # Transformer output
    print(predicted)  # prediction
    print(data[2])  # Label data
    print("Pass the feed-forward test!")
    # sys.exit(0)
    # Test a single feed-forward only process

    # To record training procession
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    max_valid_acc = 0.0
    max_test_acc = 0.0
    # Start training
    t_start = time.time()
    for epoch in range(params['epochs']):
        # Start
        train_total_loss = 0.0
        train_total_acc = 0.0
        train_total = 0.0

        valid_total_loss = 0.0
        valid_total_acc = 0.0
        valid_total = 0.0

        NN_model.train()
        for i, data in enumerate(train_data_loader):
            NN_model.zero_grad()
            # one iteration
            rgb_imgs = data[0]
            tactile_imgs = data[1]
            label = data[2]
            if params['Modality'] == "Combined":
                output = NN_model(rgb_imgs, tactile_imgs)
            elif params['Modality'] == "Visual":
                output = NN_model(rgb_imgs)
            elif params['Modality'] == "Tactile":
                output = NN_model(tactile_imgs)
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
        if epoch % params['print_interval'] == 0:
            message = '[Epoch: %3d/%3d] Training Loss: %.3f, Training Acc: %.3f' % (
            epoch, params['epochs'], train_loss[epoch], train_acc[epoch])
            log_record.update_log(exp_save_dir, message)
            print(message)

        NN_model.eval()
        with torch.no_grad():
            for rgb_imgs, tactile_imgs, label in valid_data_loader:
                if params['Modality'] == "Combined":
                    output = NN_model(rgb_imgs, tactile_imgs)
                elif params['Modality'] == "Visual":
                    output = NN_model(rgb_imgs)
                elif params['Modality'] == "Tactile":
                    output = NN_model(tactile_imgs)
                if params['use_gpu']:
                    label = label.cuda()
                loss = loss_function(output, label)
                _, predicted = torch.max(output.data, 1)
                valid_total_acc += (predicted == label).sum().item()
                valid_total_loss += float(loss.data)
                valid_total += len(label)
        valid_loss.append(valid_total_loss / valid_total)
        valid_acc.append(valid_total_acc / valid_total)
        if epoch % params['print_interval'] == 0:
            message = '[Epoch: %3d/%3d] Validation Loss: %.3f, Validation Acc: %.3f' % (
            epoch, params['epochs'], valid_loss[epoch], valid_acc[epoch])
            log_record.update_log(exp_save_dir, message)
            print(message)
            message = "Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
                elapsed_time, speed_epoch, speed_batch, eta)
            print(message)
            log_record.update_log(exp_save_dir, message)

        # if valid_acc[epoch] > max_valid_acc:
        max_valid_acc = valid_acc[epoch]
        if params['test_eval'] == 1:
            test_total_loss = 0.0
            test_total_acc = 0.0
            test_total = 0.0
            NN_model.eval()
            with torch.no_grad():
                for rgb_imgs, tactile_imgs, label in test_data_loader:
                    if params['Modality'] == "Combined":
                        output = NN_model(rgb_imgs, tactile_imgs)
                    elif params['Modality'] == "Visual":
                        output = NN_model(rgb_imgs)
                    elif params['Modality'] == "Tactile":
                        output = NN_model(tactile_imgs)
                    if params['use_gpu']:
                        label = label.cuda()
                    loss = loss_function(output, label)
                    _, predicted = torch.max(output.data, 1)
                    test_total_acc += (predicted == label).sum().item()
                    test_total_loss += float(loss.data)
                    test_total += len(label)
            test_total_loss = test_total_loss / test_total
            test_total_acc = test_total_acc / test_total
        if test_total_acc >= max_test_acc:
            max_test_acc = test_total_acc
            message = 'Model Improved: [Epoch: %3d/%3d] Test Loss: %.3f, Test Acc: %.3f' % (
            epoch, params['epochs'], test_total_loss, test_total_acc)
            log_record.update_log(exp_save_dir, message)
            print(message)
            if 'save_dir' in params.keys():
                print(' Model improved. Saving model')
                model_path = os.path.join(exp_save_dir + '/' + params['Model_name'] + '_{:0>5}.pt'.format(epoch))
                state_dict = {'model': NN_model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state_dict, model_path)

        print(" ")

        # do one feed-forward on the test dataset
    if params['test_eval'] == 1:
        test_total_loss = 0.0
        test_total_acc = 0.0
        test_total = 0.0
        NN_model.eval()
        with torch.no_grad():
            for rgb_imgs, tactile_imgs, label in test_data_loader:
                if params['Modality'] == "Combined":
                    output = NN_model(rgb_imgs, tactile_imgs)
                elif params['Modality'] == "Visual":
                    output = NN_model(rgb_imgs)
                elif params['Modality'] == "Tactile":
                    output = NN_model(tactile_imgs)
                if params['use_gpu']:
                    label = label.cuda()
                loss = loss_function(output, label)
                _, predicted = torch.max(output.data, 1)
                test_total_acc += (predicted == label).sum().item()
                test_total_loss += float(loss.data)
                test_total += len(label)
        test_total_loss = test_total_loss / test_total
        test_total_acc = test_total_acc / test_total
        message = 'After the final epoch: Test Loss: %.3f, Test Acc: %.3f' % (test_total_loss, test_total_acc)
        log_record.update_log(exp_save_dir, message)
        print(message)

    if 'save_dir' in params.keys():
        model_path = os.path.join(exp_save_dir + '/' + params['Model_name'] + '_last.pt'.format(epoch))
        state_dict = {'model': NN_model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state_dict, model_path)
        plot_loss_accuracy(train_loss, valid_loss, train_acc, valid_acc, exp_save_dir, colors=['blue'],
                           loss_legend_loc='upper center', acc_legend_loc='upper left')
