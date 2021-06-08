# Temporal Memory Attention for Video Semantic Segmentation, [arxiv](https://arxiv.org/abs/2102.08643)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-memory-attention-for-video-semantic/video-semantic-segmentation-on-camvid)](https://paperswithcode.com/sota/video-semantic-segmentation-on-camvid?p=temporal-memory-attention-for-video-semantic)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/temporal-memory-attention-for-video-semantic/video-semantic-segmentation-on-cityscapes-val)](https://paperswithcode.com/sota/video-semantic-segmentation-on-cityscapes-val?p=temporal-memory-attention-for-video-semantic)

## Introduction
We propose a Temporal Memory Attention Network (TMANet) to adaptively integrate the long-range temporal relations over 
the video sequence based on the self-attention mechanism without exhaustive optical flow prediction.
Our method achieves new state-of-the-art performances on two challenging video semantic segmentation datasets, 
particularly 80.3% mIoU on Cityscapes and 76.5% mIoU on CamVid with ResNet-50. (Accepted by ICIP2021)

![image](images/overview.jpg)

## Updates
2021/1: TMANet training and evaluation code released.

2021/6: Update README.md:
  * adding some Camvid dataset download links;
  * update 'camvid_video_process.py' script.
## Usage
* Install mmseg
  * Please refer to [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) to get installation guide. 
  * This repository is based on mmseg-0.7.0 and pytorch 1.6.0.
* Clone the repository.
  ```shell
  git clone https://github.com/wanghao9610/TMANet.git
  cd TMANet
  pip install -e .
  ```
* Prepare the datasets
  * Download [Cityscapes](https://www.cityscapes-dataset.com/) dataset and [Camvid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) dataset. 
  * For Camvid dataset, we need to extract frames from downloaded
    videos according to the following steps:
    * Download the raw video from [here](https://drive.google.com/drive/folders/19eAfQ7Of4LUe4C4Z-S60EcTFC8BCVTr0?usp=sharing), in which I provide a google drive link to download.
    * Put the downloaded raw video(e.g. 0016E5.MXF, 0006R0.MXF, 0005VD.MXF, 01TP_extract.avi) to ./data/camvid/raw .
    * Download the extracted images and labels from [here](https://drive.google.com/file/d/1FcVdteDSx0iJfQYX2bxov0w_j-6J7plz/view?usp=sharing) 
      and split.txt file from [here](https://drive.google.com/drive/folders/1a9I09fnI9s1mGBFRB7bW5dyzhs5MkTZ7?usp=sharing), untar the tar.gz file to ./data/camvid , 
      and we will get two subdirs "./data/camvid/images" (stores the images with annotations), and "./data/camvid/labels" (stores the ground 
      truth for semantic segmentation). Reference the following shell command: 
      ```shell
      cd TMANet
      cd ./data/camvid
      wget https://drive.google.com/file/d/1FcVdteDSx0iJfQYX2bxov0w_j-6J7plz/view?usp=sharing
      # or first download on your PC then upload to your server.
      tar -xf camvid.tar.gz 
      ```
    * Generate image_sequence dir frame by frame from the raw videos. Reference the following shell command:
      ```shell
      cd TMAnet
      python tools/convert_datasets/camvid_video_process.py
      ```
  * For Cityscapes dataset, we need to request the download link of 'leftImg8bit_sequence_trainvaltest.zip' from 
    [Cityscapes dataset official webpage](https://www.cityscapes-dataset.com/downloads/).
  * The converted/downloaded datasets store on ./data/camvid and ./data/cityscapes path.
    
    File structure of video semantic segmentation dataset is as followed.
    ```none
    ├── data                                              ├── data                              
    │   ├── cityscapes                                    │   ├── camvid                        
    │   │   ├── gtFine                                    │   │   ├── images                    
    │   │   │   ├── train                                 │   │   │   ├── xxx{img_suffix}       
    │   │   │   │   ├── xxx{img_suffix}                   │   │   │   ├── yyy{img_suffix}       
    │   │   │   │   ├── yyy{img_suffix}                   │   │   │   ├── zzz{img_suffix}       
    │   │   │   │   ├── zzz{img_suffix}                   │   │   ├── annotations               
    │   │   │   ├── val                                   │   │   │   ├── train.txt             
    │   │   ├── leftImg8bit                               │   │   │   ├── val.txt               
    │   │   │   ├── train                                 │   │   │   ├── test.txt              
    │   │   │   │   ├── xxx{seg_map_suffix}               │   │   ├── labels                    
    │   │   │   │   ├── yyy{seg_map_suffix}               │   │   │   ├── xxx{seg_map_suffix}   
    │   │   │   │   ├── zzz{seg_map_suffix}               │   │   │   ├── yyy{seg_map_suffix}   
    │   │   │   ├── val                                   │   │   │   ├── zzz{seg_map_suffix}   
    │   │   ├── leftImg8bit_sequence                      │   │   ├── image_sequence            
    │   │   │   ├── train                                 │   │   │   ├── xxx{sequence_suffix}  
    │   │   │   │   ├── xxx{sequence_suffix}              │   │   │   ├── yyy{sequence_suffix}  
    │   │   │   │   ├── yyy{sequence_suffix}              │   │   │   ├── zzz{sequence_suffix}  
    │   │   │   │   ├── zzz{sequence_suffix}              
    │   │   │   ├── val                                   
    ```

* Evaluation
  * Download the trained models for [Cityscapes](https://drive.google.com/drive/folders/16EMm46zRIKkGC-wIse4In5lV6zUZCIQB) and [Camvid](https://drive.google.com/drive/folders/1wiKyMZItme9cb1Kfidtm4ziDT7TdrQ22?usp=sharing). And put them on ./work_dirs/{config_file}  
  * Run the following command(on Cityscapes):
  ```shell
  sh eval.sh configs/video/cityscapes/tmanet_r50-d8_769x769_80k_cityscapes_video.py
  ```
* Training
  * Please download the pretrained [ResNet-50](https://drive.google.com/drive/folders/1IRkBsvJpZ1R1cS5La-7On03VoJErgvGX) model, and put it on ./init_models .
  * Run the following command(on Cityscapes):
  ```shell
  sh train.sh configs/video/cityscapes/tmanet_r50-d8_769x769_80k_cityscapes_video.py
  ```
  Note: the above evaluation and training shell commands execute on Cityscapes, if you want to execute evaluation or 
  training on Camvid, please replace the config file on the shell command with the config file of Camvid.
## Citation
  If you find TMANet is useful in your research, please consider citing:
  ```shell
  @misc{wang2021temporal,
      title={Temporal Memory Attention for Video Semantic Segmentation}, 
      author={Hao Wang and Weining Wang and Jing Liu},
      year={2021},
      eprint={2102.08643},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
  }
  ```
## Acknowledgement
Thanks [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) contribution to the community!
