from train import train_net
import sys
import os
import yaml

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
# or specify the configuration by running `python main.py custom`
if __name__ == "__main__":
    exp_name = 'config_cluster.yaml' # make sure config.yaml is under the same directory

    if len(sys.argv) > 1:
        exp_name = sys.argv[1]  #or we can explicitly specify the file name

    print("Running Experiment: ", exp_name)
    yaml_file = exp_name
    if os.path.exists(yaml_file):
        with open(yaml_file) as stream:
            config_loaded = yaml.safe_load(stream)
    else:
        print("The yaml file does not exist!")
        sys.exit()
    if not os.path.exists(config_loaded['save_dir']):
        os.mkdir(config_loaded['save_dir'])
    if not os.path.exists(config_loaded['save_dir']  + '/' + config_loaded['Model_name']):
        os.mkdir(config_loaded['save_dir']  + '/' + config_loaded['Model_name'])
    train_net(config_loaded) # train the model, with the specifications

