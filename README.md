### Full

### Training

#### Train detector

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/train.sh jiqing kd 0251 res101s4
```

#### Train student

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/train_student.sh jiqing kd 0251 res101s4 res18s4_kd
```

#### Train KD

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/train_kd.sh jiqing kd 0251 res101s4 res18s4_kd
```

### Infer

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/infer.sh jiqing kd 0251 res101s4
```