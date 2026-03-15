# Run Commands

## GPU 할당

```bash
# 단일 GPU 지정 (0~7)
CUDA_VISIBLE_DEVICES=0 python run.py --benchmark_type libero_object

# 여러 GPU 지정
CUDA_VISIBLE_DEVICES=0,1 python run.py --benchmark_type libero_object

# 각 task를 서로 다른 GPU에서 동시 실행
CUDA_VISIBLE_DEVICES=0 python run.py --benchmark_type libero_object &
CUDA_VISIBLE_DEVICES=1 python run.py --benchmark_type libero_spatial &
CUDA_VISIBLE_DEVICES=2 python run.py --benchmark_type libero_goal &
CUDA_VISIBLE_DEVICES=3 python run.py --benchmark_type libero_90 &
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --benchmark_type libero_object
CUDA_VISIBLE_DEVICES=1 python run.py --benchmark_type libero_spatial
CUDA_VISIBLE_DEVICES=2 python run.py --benchmark_type libero_goal
CUDA_VISIBLE_DEVICES=3 python run.py --benchmark_type libero_90
```

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --benchmark_type libero_object --checkpoint_path runs/libero_object/<date>/<time>/final_model.pth
CUDA_VISIBLE_DEVICES=1 python run.py --benchmark_type libero_spatial --checkpoint_path runs/libero_spatial/<date>/<time>/final_model.pth
CUDA_VISIBLE_DEVICES=2 python run.py --benchmark_type libero_goal --checkpoint_path runs/libero_goal/<date>/<time>/final_model.pth
CUDA_VISIBLE_DEVICES=3 python run.py --benchmark_type libero_90 --checkpoint_path runs/libero_90/<date>/<time>/final_model.pth
```

## Resume (libero_90)

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --benchmark_type libero_90 --resume runs/libero_90/2026-03-13/06-42-03/resume_last.pth
```
