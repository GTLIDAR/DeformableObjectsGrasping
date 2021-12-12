# Deformable Objects Grasping

This repository contains code and data associated with paper *"Learning Generalizable Vision-Tactile Robotic Grasping Strategy for Deformable Objects via Transformer"*. Our paper link is: , and if you are interested in our work and would like to cite our paper, please use the following citation format:

### BibTex
```
@misc{Han2021TransGrasping,
  author = {Yunhai Han, Rahul Batra, Nathan Boyd, Tuo Zhao, Yu She, Seth Hutchinson, and Ye Zhao},
  title = {Learning Generalizable Vision-Tactile Robotic Grasping Strategy for Deformable Objects via Transformer},
  year = {2021},
  journal = {arxiv}
}
```
***

## Tasks
1) Slip Detection  
Work done as part of the paper "Slip Detection with Combined Tactile and Vision Information". They collected the slip dataset and trained a CNN+LSTM model to detect slip. We showed that the Transformer models outperform the CNN+LSTM model  in term of accuracy and efficiency. 
See their paper link: https://arxiv.org/abs/1802.10153
; See a GitHub project link: https://github.com/wkoa/slip_detection (not released from the authors)

2) Grasping Framework  
Original work done as part of our paper. we propose a Transformer-based robotic grasping framework for rigid grippers that leverage tactile and visual information for safe object grasping. Specificaylly, our framework can predict the grasping outcome given a grasping force threshold and to estimate the force threshold for safe grasping through inference. 

3) Fruit classification  
Original work done as part of our paper. Our framework can also classify the grasped fruit type using the same input data but with a different MLP network for the final fruit classification task. During training, the weights of the Transformer models frozen and only the MLP network is trained. Therefore, in order to train the model for fruit classification, make sure that you have already trained a grasping model and specify its path in the yaml file becuase the Transformer weights will be loaded.



***

## Datasets

There are two datasets 1) Slip Detection Dataset and 2) Grasping Dataset

### 1. Slip Detection Dataset
Download the whole dataset from: https://www.dropbox.com/sh/i39pjxfqiwhbu1n/AAD5FNjk-Nt28UZ8lsVuVd4ja?dl=0. 

Slip Dataset consists of video frames captured by Gelsight Tactile sensor and Realsense Camera both mounted on a robotic arm(which arm). Robot arm action is labelled either slipping or not slipping. Dataset consists of 89 everyday objects like bottles, torch, etc

### Bottle
![alt text](data_example/slip_detection/object001_gelsight.gif     "Gelsight frame" ) | ![alt text](data_example/slip_detection/object001_realsense.gif    "Realsense frame" )
:--:|:--:
Gelsight | Camera 

### Box
![alt text](data_example/slip_detection/object002_gelsight.gif     "Gelsight frame" ) | ![alt text](data_example/slip_detection/object002_realsense.gif    "Realsense frame" )
:--:|:--:
Gelsight | Camera 

### Flashlight 
![alt text](data_example/slip_detection/object003_gelsight.gif     "Gelsight frame" ) | ![alt text](data_example/slip_detection/object003_realsense.gif    "Realsense frame" )
:--:|:--:
Gelsight | Camera 

### Metal Bar
![alt text](data_example/slip_detection/object005_gelsight.gif     "Gelsight frame" ) | ![alt text](data_example/slip_detection/object005_realsense.gif    "Realsense frame" )
:--:|:--:
Gelsight | Camera 

### Battery
![alt text](data_example/slip_detection/object007_gelsight.gif     "Gelsight frame" ) | ![alt text](data_example/slip_detection/object007_realsense.gif    "Realsense frame" )
:--:|:--:
Gelsight | Camera 


### 2. Fruit Grasping Dataset
Download data from: https://www.dropbox.com/s/1eb3ufrr0gryxu8/FruitData.zip?dl=0.

Dataset consists of video frames captured by Gelsight Tactile sensor and Realsense Camera both mounted on a robotic arm(which arm). Robotic arm performs two actions pinching and sliding and frameset are captured for both actions for different fruits. For each frameset a fruit label is also provided. Dataset has 6 categories of fruits. Check more details in the whole dataset.

#### Apple
![alt text](data_example/fruit_data/apple_grasping_gelsight.gif     "Gelsight frame" ) | ![alt text](data_example/fruit_data/apple_grasping_realsense.gif    "Realsense frame" )|![alt text](data_example/fruit_data/apple_sliding_gelsight.gif      "Gelsight frame" )| ![alt text](data_example/fruit_data/apple_sliding_realsense.gif     "Realsense frame" )
:--:|:--:|:--:|:--:
Pinching Gelsight | Pinching Camera | Sliding Gelsight | Sliding Camera

#### Lemon
![alt text](data_example/fruit_data/lemon_grasping_gelsight.gif     "Gelsight frame" ) | ![alt text](data_example/fruit_data/lemon_grasping_realsense.gif    "Realsense frame" ) |![alt text](data_example/fruit_data/lemon_sliding_gelsight.gif      "Gelsight frame" ) |![alt text](data_example/fruit_data/lemon_sliding_realsense.gif     "Realsense frame" )
:--:|:--:|:--:|:--:
Pinching Gelsight | Pinching Camera | Sliding Gelsight | Sliding Camera

#### Orange
![alt text](data_example/fruit_data/orange_grasping_gelsight.gif     "Gelsight frame" ) |![alt text](data_example/fruit_data/orange_grasping_realsense.gif    "Realsense frame" ) | ![alt text](data_example/fruit_data/orange_sliding_gelsight.gif      "Gelsight frame" ) | ![alt text](data_example/fruit_data/orange_sliding_realsense.gif     "Realsense frame" )
:--:|:--:|:--:|:--:
Pinching Gelsight | Pinching Camera | Sliding Gelsight | Sliding Camera

#### Plum
![alt text](data_example/fruit_data/plum_grasping_gelsight.gif     "Gelsight frame" ) | ![alt text](data_example/fruit_data/plum_grasping_realsense.gif    "Realsense frame" ) | ![alt text](data_example/fruit_data/plum_sliding_gelsight.gif      "Gelsight frame" ) |![alt text](data_example/fruit_data/plum_sliding_realsense.gif     "Realsense frame" )
:--:|:--:|:--:|:--:
Pinching Gelsight | Pinching Camera | Sliding Gelsight | Sliding Camera

#### Tomato
![alt text](data_example/fruit_data/tomato_grasping_gelsight.gif     "Gelsight frame" )|![alt text](data_example/fruit_data/tomato_grasping_realsense.gif    "Realsense frame" )|![alt text](data_example/fruit_data/tomato_sliding_gelsight.gif      "Gelsight frame" )|![alt text](data_example/fruit_data/tomato_sliding_realsense.gif     "Realsense frame" )
:--:|:--:|:--:|:--:
Pinching Gelsight | Pinching Camera | Sliding Gelsight | Sliding Camera

#### Kiwi 
![alt text](data_example/fruit_data/kiwi_grasping_gelsight.gif     "Gelsight frame" )|![alt text](data_example/fruit_data/kiwi_grasping_realsense.gif    "Realsense frame" )|![alt text](data_example/fruit_data/kiwi_sliding_gelsight.gif      "Gelsight frame" )|![alt text](data_example/fruit_data/kiwi_sliding_realsense.gif     "Realsense frame" )
:--:|:--:|:--:|:--:
Pinching Gelsight | Pinching Camera | Sliding Gelsight | Sliding Camera


***

## Requirements
    * python3 >= 3.8.10                 
    * numpy >= 1.21.1                
    * pytorch  >= 1.9.0+cpu (cpu training&testing) or 1.9.0+cu102 (cuda training&testing)
    * opencv >= 4.5.3
    * yaml >= 5.4.1
    * json >= 2.0.9
    * matplotlib >= 3.4.2
***


## Training

### 1. Slip Detection Training 
```
1. Download slip detection data(See download instructions above)
2. cd src/grasping_framework (TODO)
3. python main.py [config_vivit.yml | config_timesformer.yml]  #Select between vivit or timeSformer model (TODO)
```

### 2. Fruit Grasping Training 
```
1. Download fruit grasping data(See download instructions above)
2. cd src/grasping_framework
3. python main.py [config_vivit.yml | config_timesformer.yml]  #Select between vivit or timeSformer model
```

### 3. Fruit Classification Training 
```
1. Download fruit grasping data(See download instructions above)
2. You can either download pretrained models or train a grapsing model(See Grasping Training instructions below) on your end.
* Download pre-trained TimeSofmer model from: https://www.dropbox.com/s/1vk8gy5e43y3xkz/timeSformer_orig_two_fruit190.pt?dl=0
* Download pre-trained viviT model from: https://www.dropbox.com/s/k8su3ayzolnoxyj/vivit_fdp_two_fruit_last.pt?dl=0 
3. Place the models into /src/fruit_classification/Pretrained and specify the model path in yaml files
4. cd src/fruit_classification
5. python main.py [config_vivit.yml | config_timesformer.yml]  #Select between vivit or timeSformer model
```

