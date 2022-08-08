### Setup

#### Libraries

1. Clone this repository and enter it:

```shell
git clone https://github.com/vltanh/ganet-al
cd GANet
```

1. Create a conda virtual environment and activate it.

```shell
conda create -n ganet python=3.7 -y
conda activate ganet
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements/build.txt
python setup.py develop
```

#### Dataset

Setup so that the data directory is as follows

```
data
    |- jiqing
        |- images_train
            |- 0250
                |- 1.png
                |- ...
            |- ...
        |- txt_label
            |- 0250
                |- 1.txt
                |- ...
            |- ...
        |- txt_labels
            |- images_train
                |- 0250
                    |- 1.lines.txt
                    |- ...
                |- ...
```

#### Pretrained models

Download into `dino_models` the [dino_vitbase8_pretrain_full_checkpoint.pth](https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_full_checkpoint.pth)

### Full

#### Initial round

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/al/al_init_random.sh jiqing 0251 res101s4
```

where:
- `jiqing`: dataset index
- `0251`: video index
- `res101s4`: configuration file index for the main network

#### Main rounds

Must be run after completion of a run of the initial round.

##### Random

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/al/al_random.sh jiqing random 0251 res101s4
```

where:
- `jiqing`: dataset index
- `random`: initial round strategy
- `0251`: video index
- `res101s4`: configuration file index for the main network

##### Uncertainty

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/al/al_random.sh jiqing random 0251 res101s4 res18s4_kd res18s4
```

where:
- `jiqing`: dataset index
- `random`: initial round strategy
- `0251`: video index
- `res101s4`: configuration file index for the main network (also teacher)
- `res18s4_kd`: configuration file index for the student-kd network
- `res101s4`: configuration file index for the student (small) network

##### Diversity

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/al/al_random.sh jiqing random 0251 res101s4
```

where:
- `jiqing`: dataset index
- `random`: initial round strategy
- `0251`: video index
- `res101s4`: configuration file index for the main network

### Training

#### Train detector

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/train.sh jiqing 0251 res101s4
```

where:
- `jiqing`: dataset index
- `0251`: video index
- `res101s4`: configuration file index for the main network

#### Train student

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/train_student.sh jiqing 0251 res101s4 res18s4_kd
```

where:
- `jiqing`: dataset index
- `0251`: video index
- `res101s4`: configuration file index for the teacher network
- `res18s4_kd`: configuration file index for the student network

#### Train KD

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/train_kd.sh jiqing 0251 res101s4 res18s4_kd
```

where:
- `jiqing`: dataset index
- `0251`: video index
- `res101s4`: configuration file index for the teacher network
- `res18s4_kd`: configuration file index for the student network

### Infer

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/infer.sh jiqing checkpoints/jiqing/diversity/0251/0 res101s4
```

where:
- `jiqing`: dataset index
- `checkpoints/jiqing/diversity/0251/0`: path to output directory
- `res101s4`: configuration file index for the teacher network