export PYTHONPATH=/export/App/zhaoziqiang/facenet/src
export CUDA_VISIBLE_DEVICES=1
#py=/export/software/anaconda/bin/python2.7
py=/export/software/anaconda-tf-1.0/bin/python2.7


$py src/facenet_train_classifier3.py \
    --logs_base_dir ./logs/trainv2_1/ \
    --models_base_dir ./models/trainv2_1/ \
    --data_dir ../big-black-all-train-160-subdir-same \
    --image_size 160 \
    --model_def models.inception_resnet_v2 \
    --optimizer RMSPROP \
    --gpu_memory_fraction 0.9 \
    --learning_rate -1 \
    --max_nrof_epochs 10000 \
    --keep_probability 0.8 \
    --learning_rate_schedule_file data/train1.txt \
    --weight_decay 5e-5 \
    --center_loss_factor 0.001 \
    --center_loss_alfa 0.9

