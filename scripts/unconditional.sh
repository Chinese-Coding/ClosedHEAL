PYTHONPATH=/home/zfq/Desktop/OriginHEAL:$PYTHONPATH \
accelerate launch \
  --num_machines=1 --dynamo_backend="auto" \
  opencood/tools/train_unconditional.py \
  --output_dir="/home/zfq/Desktop/logs/diffusion_train" \
  --root_dir="/dataset/OPV2V/train" \
  --mixed_precision="bf16" --train_batch_size=32 \
  --dataloader_num_workers=4 --num_epochs=100 \
  --enable_xformers_memory_efficient_attention


