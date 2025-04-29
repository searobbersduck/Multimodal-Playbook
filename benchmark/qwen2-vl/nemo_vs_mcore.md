## 

```
mkdir -p /workspace/code/nemo
cd /workspace/code/nemo

git clone https://github.com/NVIDIA/NeMo.git
cd Nemo
git checkout cd55157cb05e0a60066caf31e3354ff0e690b086
```

```
环境
container: nvcr.io/nvidia/nemo:24.12.01
nemo code: https://gitlab-master.nvidia.com/xueh/nemo-qwenvl
megatron code: https://gitlab-master.nvidia.com/xueh/megatron-lm-qwen-2-vl/-/blob/update
container-mounts一下: /home/xueh/projects/Qwen2-vl/megatron-lm:/opt/megatron-lm,/home/xueh/projects/Qwen2-vl/nemo-qwenvl:/opt/NeMo
然后安装一下：
pip install qwen_vl_utils
pip install lightning
pip install --upgrade transformers
pip install --upgrade megatron-energon
```

```
srun -w h20-[5-6] -N 1 --gres=gpu:8 --container-image=/home/weidongz/docker_workspace/images/nemo-2504.sqsh --container-save=/home/weidongz/docker_workspace/images/nemo-2504.sqsh --container-remap-root --container-mounts=/home/weidongz/docker_workspace:/workspace --container-writable --pty bash
```

```
torchrun --nproc_per_node=2 /workspace/code/nemo/NeMo/scripts/vlm/qwen2vl_finetune.py  \
    --data_type=qwen2vl \
    --data_path=/workspace/data/mm/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /workspace/data/mm/data/LLaVA-Pretrain/images \
    --num_nodes 1 \
    --log_dir "/workspace/code/nemo/NeMo/experiments/qwen2vl_finetune" \
    --devices=2 \
    --tp_size 2 --pp_size 1 \
    --gbs 32 --mbs 1 \
    --wandb_project=qwen2vl_demo \
    --name=qwen2vl_finetune \
    --restore_path "/workspace/code/nemo/NeMo/experiments/qwen2vl_checkpoint"
```

```
WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="qwen2.5-vl-convergence"

# RESULTS="${WORK_DIR}/results_${NAME}"
# mkdir -p ${RESULTS}

wandb login ${WANDB}

torchrun --nproc_per_node=2 /workspace/code/nemo/NeMo/scripts/vlm/qwen2vl_finetune.py  \
    --data_type=energon \
    --data_path=/workspace/data/mm/data/LLaVA-Pretrain/wds1 \
    --image_folder /workspace/data/mm/data/LLaVA-Pretrain/images \
    --num_nodes 1 \
    --log_dir "/workspace/code/nemo/NeMo/experiments/qwen2vl_finetune" \
    --devices=2 \
    --tp_size 2 --pp_size 1 \
    --gbs 32 --mbs 1 \
    --wandb_project=qwen2vl_demo \
    --name=qwen2vl_finetune
```

```
WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="qwen2vl_demo"

# RESULTS="${WORK_DIR}/results_${NAME}"
# mkdir -p ${RESULTS}

wandb login ${WANDB}

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 /workspace/code/nemo/NeMo/scripts/vlm/qwen2vl_finetune.py  \
    --data_type=energon \
    --data_path=/workspace/data/mm/data/LLaVA-Pretrain/wds \
    --image_folder /workspace/data/mm/data/LLaVA-Pretrain/images \
    --num_nodes 1 \
    --log_dir "/workspace/code/nemo/NeMo/experiments/qwen2vl_finetune_gbs1mbs1" \
    --devices=1 \
    --tp_size 1 --pp_size 1 \
    --gbs 1 --mbs 1 \
    --wandb_project=qwen2vl_demo \
    --name=qwen2vl_finetune
```

```
python /workspace/code/nemo/NeMo/nemo/collections/vlm/qwen2vl/data/convert_to_qwen2vl_wds.py --dataset-root=/workspace/data/mm/data/LLaVA-Pretrain --json=/workspace/data/mm/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json --max-samples-per-tar=1 --mediate-path=images

```



[NeMo I 2025-04-27 17:03:58 utils:507] Setting up optimizer with config OptimizerConfig(optimizer='adam', lr=2e-06, min_lr=None, decoupled_lr=None, decoupled_min_lr=None, weight_decay=0.01, fp16=False, bf16=True, params_dtype=torch.float32, use_precision_aware_optimizer=False, main_grads_dtype=torch.float32, main_params_dtype=torch.float32, exp_avg_dtype=torch.float32, exp_avg_sq_dtype=torch.float32, loss_scale=None, initial_loss_scale=4294967296, min_loss_scale=1.0, loss_scale_window=1000, hysteresis=2, adam_beta1=0.9, adam_beta2=0.95, adam_eps=1e-08, sgd_momentum=0.9, use_distributed_optimizer=True, overlap_param_gather_with_optimizer_step=False, optimizer_cpu_offload=False, optimizer_offload_fraction=0.0, use_torch_optimizer_for_cpu_offload=False, overlap_cpu_optimizer_d2h_h2d=False, pin_cpu_grads=True, pin_cpu_params=True, clip_grad=1.0, log_num_zeros_in_grad=False, barrier_with_L1_time=False, timers=None, config_logger_dir='')


export CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 /workspace/code/nemo/NeMo/scripts/vlm/qwen2vl_finetune.py  \
    --data_type=energon \
    --data_path=/workspace/data/mm/data/LLaVA-Pretrain/wds1 \
    --image_folder /workspace/data/mm/data/LLaVA-Pretrain/images \
    --num_nodes 1 \
    --log_dir "/workspace/code/nemo/NeMo/experiments/qwen2vl_finetune_gbs1mbs1_exp2" \
    --devices=1 \
    --tp_size 1 --pp_size 1 \
    --gbs 1 --mbs 1 \
    --wandb_project=qwen2vl_demo \
    --name=qwen2vl_finetune
