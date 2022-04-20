# An Implementation of HRNets for Image Classification

## Note
- Please refer to [https://github.com/HRNet/HRNet-Image-Classification/](https://github.com/HRNet/HRNet-Image-Classification/) for official implementation.

## ImageNet-1K Accuracy
| Model | Resolution | #Params | FLOPs |  #Epochs | Top-1 (%)| Download |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNet-W18 | 224x224 | 21.3M	| 3.99G | 300 | 78.6 | [hrnet18-6ca9d2049.pth]() |
| HRNet-W18 | 224x224 | 21.3M	| 3.99G | 600 | 79.4 | [hrnet18-699e7ab89.pth](https://drive.google.com/file/d/1Jpw1C9QCJuZTmczpEpW4rclc6JOSuNcU/view?usp=sharing) |
| HRNet-W32 | 224x224 | 41.2M	| 8.31G | 300 | 80.5 | [hrnet32-21df535e7.pth]() |
| HRNet-W32 | 224x224 | 41.2M	| 8.31G | 600 | 81.2 | [hrnet32-9f864d2d6.pth](https://drive.google.com/file/d/1lnTLueRkd1VixTSNhFY6CdBm8rOZlvD_/view?usp=sharing) |

## Training
```bash
# Note: https://github.com/rwightman/pytorch-image-models/blob/master/distributed_train.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /path/to/imagenet/ --model hrnet32 --amp
```

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal= {TPAMI}
  year={2019}
}
````
