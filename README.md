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