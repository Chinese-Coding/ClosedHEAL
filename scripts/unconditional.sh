PYTHONPATH=/home/zfq/Desktop/OriginHEAL:$PYTHONPATH \
accelerate launch opencood/tools/train_unconditional.py \
  --output_dir="/home/zfq/Desktop/logs/diffusion_train" \
  --mixed_precision="bf16" \
  --train_batch_size=32 \
  --enable_xformers_memory_efficient_attention

