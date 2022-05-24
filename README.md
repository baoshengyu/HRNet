# An Implementation of HRNets for Image Classification

## Note
- Please refer to [https://github.com/HRNet/HRNet-Image-Classification/](https://github.com/HRNet/HRNet-Image-Classification/) for the official implementation.
 

## ImageNet-1K Accuracy
| Model | Resolution | #Params | FLOPs | #Epochs | Top-1 Acc(%)| Pretrained Weights |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNet-W18 | 224x224 | 21.3M	| 3.99G | 300 | 78.6 | [hrnet18-6ca9d2049.pth](https://drive.google.com/file/d/16mWKwZTlV9ypctaTspL34zdxl4xn7tVf/view?usp=sharing) |
| HRNet-W18 | 224x224 | 21.3M	| 3.99G | 600 | 79.4 | [hrnet18-699e7ab89.pth](https://drive.google.com/file/d/1Jpw1C9QCJuZTmczpEpW4rclc6JOSuNcU/view?usp=sharing) |
| HRNet-W32 | 224x224 | 41.2M	| 8.31G | 300 | 80.5 | [hrnet32-21df535e7.pth](https://drive.google.com/file/d/1z-8vAaZhDRBie3za2Q7iqNOIHFNnejQG/view?usp=sharing) |
| HRNet-W32 | 224x224 | 41.2M	| 8.31G | 600 | 81.2 | [hrnet32-9f864d2d6.pth](https://drive.google.com/file/d/1lnTLueRkd1VixTSNhFY6CdBm8rOZlvD_/view?usp=sharing) |

## Training Example
- All models are trained using a similar strategy to [DeiT](https://github.com/facebookresearch/deit).
- Environment: pytorch==1.10.0, timm==0.5.0.
- Training codes are modified from [https://github.com/rwightman/pytorch-image-models/](https://github.com/rwightman/pytorch-image-models/).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /path/to/imagenet/ --model hrnet32 --amp
```

## Reference

````
@article{wang2020deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal= {TPAMI}
  year={2020}
}
````
