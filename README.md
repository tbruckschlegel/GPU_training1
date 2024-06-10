# Pytorch GPU training test project

A Pytorch test project. 

- Based on https://github.com/ltkong218/IFRNet

- Changed to use https://github.com/lhao0301/pytorch-liteflownet3 instead of https://github.com/sniklaus/pytorch-liteflownet and to run on a single GPU, instead of a distributed set of nodes.

## Requirements

* Windows or Linux OS
* Python 3.x (https://www.python.org/)
* Pytorch with CUDA support (https://pytorch.org/get-started/locally/)
* CuPy (https://github.com/cupy/cupy)
* CUDA (https://developer.nvidia.com/cuda-downloads) 

## Running the script

**Clone the repo**

```$ git clone https://github.com/tbruckschlegel/example1.git```

**Execute the training**

Follow the steps from here https://github.com/ltkong218/IFRNet until you reach the training section, then run the single GPU training:

```python train_vimeo90k_single_gpu.py --model_name IFRNet --world_size 4 --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5 --training_data_location "d:\Downloads\vimeo_triplet"```

```python train_vimeo90k_single_gpu.py --model_name IFRNet_S --world_size 4 --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5 --training_data_location "d:\Downloads\vimeo_triplet"```

```python train_vimeo90k_single_gpu.py --model_name IFRNet_L --world_size 4 --epochs 300 --batch_size 6 --lr_start 1e-4 --lr_end 1e-5 --training_data_location "d:\Downloads\vimeo_triplet"```

**References**

```
[1] @InProceedings{Kong_2022_CVPR, 
  author = {Kong, Lingtong and Jiang, Boyuan and Luo, Donghao and Chu, Wenqing and Huang, Xiaoming and Tai, Ying and Wang, Chengjie and Yang, Jie}, 
  title = {IFRNet: Intermediate Feature Refine Network for Efficient Frame Interpolation}, 
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  year = {2022}
}
```

```
[2] @inproceedings{Hui_CVPR_2018,
  author = {Tak-Wai Hui and Xiaoou Tang and Chen Change Loy},
  title = {{LiteFlowNet}: A Lightweight Convolutional Neural Network for Optical Flow Estimation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
  year = {2018}
}
```

```
[3] @misc{pytorch-liteflownet,
  author = {Simon Niklaus},
  title = {A Reimplementation of {LiteFlowNet} Using {PyTorch}},
  year = {2019},
  howpublished = {\url{https://github.com/sniklaus/pytorch-liteflownet}}
}
```

```
[4] @inproceedings{hui2020liteflownet3,
  title={LiteFlowNet3: Resolving Correspondence Ambiguity for More Accurate Optical Flow Estimation},
  author={Hui, Tak-Wai and Loy, Chen Change},
  booktitle={European Conference on Computer Vision},
  pages={169--184},
  year={2020},
  organization={Springer}
}
```
