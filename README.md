## This is a NAS-based search program for hyperspectral image classification.

## Installation

-   `Python 3.11`
-   `Pytorch 2.3.1`
-   `pip install -r requirements.txt`

## Datasets preparing

- You need to package the dataset and its ground truth into an h5 format file.
- The MUUFL dataset can be obtained at the URL: https://github.com/GatorSense/MUUFLGulfport.git

- The Hoouston2018 dataset can be obtained at the URL: 2018 IEEE GRSS Data Fusion Challenge – Fusion of Multispectral LiDAR and Hyperspectral Data – Machine Learning and Signal Processing Laboratory.

- The XiongAn dataset can be freely available at the URL: http://www.hrs-cas.com/a/share/shujuchanpin/2019/0501/1049.html

- If you would like to generate the training, test and val samples, run the samples_extraction.py script to assign them.
- •	python samples_extraction.py --data_root data_dir --dist_dir output_dir --dataset dataset_name --train_num number_training_samples --val_num number_val_samples

- Then, you should set the path of sample assignment files e,g("Houston2018_dist_per_train-20.0_val-10.0.h5") in the config files.


## Architucture Searching

-   `python search.py --config-file './configs/Houston2018/search.yaml' --device '0'`
-   `python search.py --config-file './configs/MUUFL/search.yaml' --device '0'`
-   `python search.py --config-file './configs/XiongAn/search.yaml' --device '0'`

## Model Training and Inference

-   `python train.py --config-file './configs/Houston2018/train.yaml' --device '0'`
-   `python train.py --config-file './configs/MUUFL/train.yaml' --device '0'`
-   `python train.py --config-file './configs/XiongAn/train.yaml' --device '0'`
