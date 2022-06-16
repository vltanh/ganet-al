### Full

### Training

#### Train detector

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/train.sh jiqing kd 0250 res101s4
```

#### Train student

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/train_student.sh jiqing kd 0250 res101s4 res18s8_kd
```

#### Train KD

```
CUDA_VISIBLE_DEVICES=4 bash -e scripts/train_kd.sh jiqing kd 0250 res101s4
```

### Infer