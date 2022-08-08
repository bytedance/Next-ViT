# Next-ViT
This repo is the official implementation of ["Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios"](https://arxiv.org/abs/2207.05501).  
<div style="text-align: center">
<img src="images/structure.png" title="Next-ViT-S" height="75%" width="75%">
</div>
Figure 1. The overall hierarchical architecture of Next-ViT.</center>

## Introduction
Due to the complex attention mechanisms and model design, most existing vision Transformers (ViTs) can not perform as efficiently as convolutional neural networks (CNNs) in realistic industrial deployment scenarios, e.g. TensorRT and CoreML. This poses a distinct challenge: Can a visual neural network be designed to infer as fast as CNNs and perform as powerful as ViTs? Recent works have tried to design CNN-Transformer hybrid architectures to address this issue, yet the overall performance of these works is far away from satisfactory. To end these, we propose a next generation vision Transformer for efficient deployment in realistic industrial scenarios, namely Next-ViT, which dominates both CNNs and ViTs from the perspective of latency/accuracy trade-off. In this work, the Next Convolution Block (NCB) and Next Transformer Block (NTB) are respectively developed to capture local and global information with deployment-friendly mechanisms. Then, Next Hybrid Strategy (NHS) is designed to stack NCB and NTB in an efficient hybrid paradigm, which boosts performance in various downstream tasks. Extensive experiments show that Next-ViT significantly outperforms existing CNNs, ViTs and CNN-Transformer hybrid architectures with respect to the latency/accuracy trade-off across various vision tasks. On TensorRT, Next-ViT surpasses ResNet by 5.5 mAP (from 40.4 to 45.9) on COCO detection and 7.7% mIoU (from 38.8% to 46.5%) on ADE20K segmentation under similar latency. Meanwhile, it achieves comparable performance with CSWin, while the inference speed is accelerated by 3.6×. On CoreML, Next-ViT surpasses EfficientFormer by 4.6 mAP (from 42.6 to 47.2) on COCO detection and 3.5% mIoU (from 45.1% to 48.6%) on ADE20K segmentation under similar latency.
![Next-ViT-R](images/result.png)
<center>Figure 2. Comparison among Next-ViT and efficient Networks, in terms of accuracy-latency trade-off.</center>


# Usage

First, clone the repository locally:
```
git clone https://github.com/bytedance/Next-ViT.git
```
Then, install `torch=1.10.0`, `mmcv-full==1.5.0`, `timm==0.4.9` and etc.

```
pip3 install -r requirements.txt
```
## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Image Classification
We provide a series of Next-ViT models pretrained on ILSVRC2012 ImageNet-1K dataset. More details can be seen in [[paper]](https://arxiv.org/abs/2207.05501).

| Model | Dataset | Resolution | FLOPs (G) | Params (M) | TensorRT <br/>Latency(ms) | CoreML <br/>Latency(ms) | Acc@1 | Pytorch<br/>Checkpoint | Log |
|:-----:|:-----:|:-----:|:-----:|:--------:|:-----------:|:-------------------------:|:-----------------------:|:----------------------:|:---:| 
| Next-ViT-S | ImageNet-1K | 224 |   5.8    |    31.7     |            7.7            |           3.5          | 82.5 |           -            |  -  |
| Next-ViT-B | ImageNet-1K | 224 |   8.3    |    44.8     |           10.5            |           4.5          | 83.2 |           -            |  -  |
| Next-ViT-L | ImageNet-1K | 224 |   10.8   |    57.8     |           13.0            |           5.5          | 83.6 |           -            |  -  |
| Next-ViT-S | ImageNet-1K | 384 |   17.3   |    31.7    |           21.6            |           8.9           |83.6 |           -            |  -  |
| Next-ViT-B | ImageNet-1K | 384 |   24.6   |    44.8     |           29.6            |           12.4         |84.3 |           -            |  -  |
| Next-ViT-L | ImageNet-1K | 384 |   32.0   |    57.8     |           36.0            |           15.2         |84.7  |           -            |  -  |

#### Training

To train Next-ViT-S on ImageNet  using 8 gpus for 300 epochs, run:

```shell
cd classification/
bash train.sh 8 --model nextvit_small --batch-size 256 --lr 5e-4 --warmup-epochs 20 --weight-decay 0.1 --data-path your_imagenet_path
```
Finetune Next-ViT-S with 384x384 input size for 30 epochs, run:
```shell
cd classification/
bash train.sh 8 --model nextvit_small --batch-size 128 --lr 5e-6 --warmup-epochs 0 --weight-decay 1e-8 --epochs 30 --sched step --decay-epochs 60 --input-size 384 --resume ../checkpoints/nextvit_small_in1k_224.pth --finetune --data-path your_imagenet_path 

```

#### Evaluation 

To evaluate the performance of Next-ViT-S on ImageNet using 8 gpus, run:
```shell
cd classification/
bash train.sh 8 --model nextvit_small --batch-size 256 --lr 5e-4 --warmup-epochs 20 --weight-decay 0.1 --data-path your_imagenet_path --resume ../checkpoints/nextvit_small_in1k_224.pth --eval
```
## Detection
Our code is based on  [mmdetection](https://github.com/open-mmlab/mmdetection), please install `mmdetection==2.23.0`. Next-ViT serve as the strong backbones for
Mask R-CNN. It's easy to apply Next-ViT in other detectors provided by mmdetection based on our examples. More details can be seen in [[paper]](https://arxiv.org/abs/2207.05501).
#### Mask R-CNN
| Backbone   | Lr Schd | Param.(M) | FLOPs(G) | bbox mAP | mask mAP | Pytorch<br/>Checkpoint | Log |
|------------|:-------:|:---------:|:--------:|:--------:|:--------:|:----------------------:|:---:|
| Next-ViT-S |   1x    |   51.8    |   290    |   45.9   |   41.8   |                        |     |
| Next-ViT-S |   3x    |   51.8    |   290    |   48.0   |   43.2   |                        |     |
| Next-ViT-B |   1x    |   64.9    |   340    |   47.2   |   42.8   |                        |     |
| Next-ViT-B |   3x    |   64.9    |   340    |   49.5   |   44.4   |                        |     |
| Next-ViT-L |   1x    |   77.9    |   391    |   48.0   |   43.2   |                        |     |
| Next-ViT-L |   3x    |   77.9    |   391    |   50.2   |   44.8   |                        |     |

#### Training
To train  Mask R-CNN with Next-ViT-S backbone using 8 gpus, run:
```shell
cd detection/
PORT=29501 bash dist_train.sh configs/mask_rcnn_nextvit_small_1x.py 8
```
#### Evaluation
To evaluate Mask R-CNN with Next-ViT-S backbone using 8 gpus, run:
```shell
cd detection/
PORT=29501 bash dist_test.sh configs/mask_rcnn_nextvit_small_1x.py ../checkpoints/mask_rcnn_1x_nextvit_small.pth 8 --eval bbox
```
## Semantic Segmentation
Our code is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), please install `mmsegmentation==0.23.0`. Next-ViT serve as the strong backbones for segmentation tasks on ADE20K dataset. It's easy to extend it to other datasets and segmentation methods. More details can be seen in [[paper]](https://arxiv.org/abs/2207.05501).

#### Semantic FPN 80k

| Backbone   | FLOPs(G) | Params (M) | TensorRT <br/>Latency(ms) | CoreML <br/>Latency(ms) | mIoU | Pytorch<br/>Checkpoint | Log |
|------------|:--------:|:----------:|:-------------------------:|:-----------------------:|:----:|------------------------|:---:|
| Next-ViT-S |   208    |    36.3    |           38.2            |          18.1           | 46.5 |                        |     |
| Next-ViT-B |   260    |    49.3    |           51.6            |          24.4           | 48.6 |                        |     |
| Next-ViT-L |   331    |    62.4    |           65.3            |          30.1           | 49.1 |                        |     |

#### UperNet 160k 

| Backbone   | FLOPs(G) | Params (M) | TensorRT <br/>Latency(ms) | CoreML <br/>Latency(ms) | mIoU(ss/ms) | Pytorch<br/>Checkpoint | Log |
|------------|:--------:|:----------:|:-------------------------:|:-----------------------:|:-----------:|------------------------|:---:|
| Next-ViT-S |   968    |    66.3    |           38.2            |          18.1           |  48.1/49.0  |                        |     |
| Next-ViT-B |   1020   |    79.3    |           51.6            |          24.4           |  50.4/51.1  |                        |     |
| Next-ViT-L |   1072   |    92.4    |           65.3            |          30.1           |  50.1/50.8  |                        |     |


#### Training
To train Semantic FPN 80k with Next-ViT-S backbone using 8 gpus, run:
```shell
cd segmentation/
PORT=29501 bash dist_train.sh configs/fpn_512_nextvit_small_80k.py 8
```
#### Evaluation
To evaluate Semantic FPN 80k(single scale) with Next-ViT-S backbone using 8 gpus, run:
```shell
cd segmentation/
PORT=29501 bash dist_test.sh configs/fpn_512_nextvit_small_80k.py ../checkpoints/fpn_80k_nextvit_small.pth 8 --eval mIoU
```

## Deployment and Latency Measurement
we provide [scripts]() to convert Next-ViT from pytorch model to [CoreML](https://developer.apple.com/documentation/coreml) model and [TensorRT](https://developer.nvidia.com/tensorrt) engine.
#### CoreML
Convert Next-ViT-S to CoreML model with `coremltools==5.2.0`, run:
```shell
cd deployment/
python3 export_coreml_model.py --model nextvit_small --batch-size 1 --image-size 224
```
| Model | Resolution | FLOPs (G) | CoreML <br/>Latency(ms) | CoreML Model |
|:-----:|:-----:|:-----:|:--------:|:-----------:| 
| Next-ViT-S | 224 |   5.8    |           3.5          |  |
| Next-ViT-B | 224 |   8.3    |           4.5          |  |
| Next-ViT-L | 224 |   10.8   |           5.5          |  |

We uniformly benchmark CoreML Latency on an iPhone12 Pro Max(iOS 16.0) with Xcode 14.0. The performance report of CoreML model can be generated with Xcode 14.0 directly([new feature](https://developer.apple.com/videos/play/wwdc2022/10027/) of Xcode 14.0).  
![Next-ViT-R](images/coreml_runtime.jpeg)
<center>Figure 3. CoreML latency of Next-ViT-S/B/L.</center>

#### TensorRT
Convert Next-ViT-S to TensorRT engine with `tensorrt==8.0.3.4`, run:
```shell
cd deployment/
python3 export_tensorrt_engine.py --model nextvit_small --batch-size 8  --image-size 224 --datatype fp16 --profile True --trtexec-path /usr/bin/trtexec
```
## Acknowledgement

We heavily borrow the code from [Twins](https://github.com/Meituan-AutoML/Twins).

## Citation
If you find this project useful in your research, please consider cite:

```
@article{li2022next,
  title={Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios},
  author={Li, Jiashi and Xia, Xin and Li, Wei and Li, Huixia and Wang, Xing and Xiao, Xuefeng and Wang, Rui and Zheng, Min and Pan, Xin},
  journal={arXiv preprint arXiv:2207.05501},
  year={2022}
}
```
