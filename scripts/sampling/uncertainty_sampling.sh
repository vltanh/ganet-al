python strategy/uncertainty_sampling.py \
    --inp_path data/jiqing/list_20fps/0251.txt \
    --out_path test.txt \
    --cfg configs/jiqing/res101s4.py \
    --teacher_cfg configs/jiqing/res101s4.py \
    --teacher_ckpt checkpoints/jiqing/kd/0251/0/res101s4/latest.pth \
    --studentkd_cfg configs/jiqing/res18s4.py \
    --studentkd_ckpt checkpoints/jiqing/kd/0251/0/res18s4_kd/latest.pth \
    --student_cfg configs/jiqing/res18s4.py \
    --student_ckpt checkpoints/jiqing/kd/0251/0/res18s4/latest.pth \
    --n_sample 100