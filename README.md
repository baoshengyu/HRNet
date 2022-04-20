# An Implementation of HRNets for ImageNet Classification


## ImageNet-1K Validation
| Model | Resolution | #Params | FLOPs |  #Epochs | Top-1 (%)| Download |
|:--:|:--:|:--:|:--:|:--:|
| HRNet-W18 | 224x224 | 21.3M	| 3.99G | 300 | 78.6 | [pretrained]() |
| HRNet-W18 | 224x224 | 21.3M	| 3.99G | 600 | 79.4 | [pretrained]() |
| HRNet-W32 | 224x224 | 41.2M	| 8.31G | 300 | 80.5 | [pretrained]() |
| HRNet-W32 | 224x224 | 41.2M	| 8.31G | 600 | 81.2 | [pretrained]() |

## Training
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /path/to/imagenet/ --model hrnet32
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
  journal   = {TPAMI}
  year={2019}
}
````
