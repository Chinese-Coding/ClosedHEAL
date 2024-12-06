#export https_proxy=http://10.29.136.93:7890
PYTHONPATH=/home/zfq/Desktop/OriginHEAL:$PYTHONPATH \
    python opencood/tools/diffusion_train.py \
        -y None \
        --model_dir /home/zfq/Desktop/logs/diffusion_train \
