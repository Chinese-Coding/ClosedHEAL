#export https_proxy=http://10.29.136.93:7890
PYTHONPATH=/home/zfq/Desktop/OriginHEAL:$PYTHONPATH \
    python opencood/tools/train_text_to_image.py \
        -y None \
        --model_dir /home/zfq/Desktop/logs/diffusion_train \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1 \
        --output_dir /home/zfq/Desktop/logs/diffusion_train \
