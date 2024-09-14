# UVZ
This is the project of paper "Underwater Variable Zoom: Depth-Guided Perception Network for Underwater Image Enhancement".

# 1. Abstract 
The underwater images often suffer from color deviations and blurred details. To address these issues, many methods employ networks with an encoder/decoder structure to enhance the images. However, the direct skip connection overlooks the differences between pre- and post-features, and deep network learning introduces information loss. This paper presents an underwater image enhancement network that focuses on pre-post differences. The network utilizes a multi-scale input and output framework to facilitate the underwater image enhancement process. A novel cross-wise transformer module (CTM) is introduced to guide the interactive learning of features from different periods, thereby enhancing the emphasis on detail-degraded regions. To compensate for the information loss within the deep network, a feature supplement module (FSM) is devised for each learning stage. FSM merges the multi-scale input features, effectively enhancing the visibility of underwater images. Experimental results across several datasets demonstrate that the integrated modules yield significant enhancements in network performance. The proposed network exhibits outstanding performance in both visual comparisons and quantitative metrics. Furthermore, the network also exhibits good adaptability in additional visual tasks without the need for parameter tuning. The code and results are released in https://github.com/WindySprint/UVZ.

## Environment
```
1. Python 3.8.18
2. PyTorch 1.13.1
3. Torchvision 0.14.1
4. OpenCV-Python 4.8.1.78
5. NumPy 1.24.4
```

## Checkpoints
[Google drive](https://drive.google.com/drive/folders/1upkU-KE-dFLi2ShlF_KVFvQR6eSQzMRN?usp=drive_link) <Br>
[Baidu drive](https://pan.baidu.com/s/1467en9VX3jL0Ye6V48iA_A ) (password: 0722)

## Test
```
1. Clone repo
2. Download 'trained_model' folder and place it in repo
3. Put the images in your folder path A
4. Change 'ori_images_path' to A and 'result_path' in test.py
5. Run test.py
6. Find results in 'result_path'
```

## Train
```
Train d_net for generate depth images
1. Put the orignal images and the depth images in your folder path A and B
2. Change 'ori_images_path' to A and 'depth_images_path' to B in train_d.py
3. Run train_d.py
4. Find trained d_net in 'checkpoint_path'
```
```
Train net for generate enhanced images
1. Put the orignal images and the GT images in your folder path A and B
2. Change 'ori_images_path' to A and 'enhan_images_path' to B in train.py
3. Ensure 'd_net_name' is consistent with the folder names of d_net
4. Change 'net_name' in train.py
5. Run train.py
6. Find trained net in 'checkpoint_path'/'net_name'
```

## Resources
If you use our code, please cite our paper. Thanks! <Br>
[Paper Link](https://www.sciencedirect.com/science/article/pii/S0957417424022176)
```
@article{HUANG2025125350,
title = {Underwater variable zoom: Depth-guided perception network for underwater image enhancement},
journal = {Expert Systems with Applications},
volume = {259},
pages = {125350},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.125350}
}
```

## Contact
If you have any questions, please contact: Zhixiong Huang: hzxcyanwind@mail.dlut.edu.cn
