export PYTHONPATH=/export/App/zhaoziqiang/facenet/src
export CUDA_VISIBLE_DEVICES=1
#py=/export/software/anaconda/bin/python2.7
py=/export/software/anaconda-tf-1.0/bin/python2.7


#--data_dir ../big-black-all-train-299-subdir-m \
#--data_dir ../big-black-all-train-160-subdir-same \
$py src/facenet_train_classifier3.py \
    --logs_base_dir ./logs/trainv2_2_299_m/ \
    --models_base_dir ./models/trainv2_2_299_m/ \
    --data_dir ../big-black-all-train-299-subdir-m \
    --image_size 299 \
    --model_def models.inception_resnet_v1 \
    --optimizer RMSPROP \
    --gpu_memory_fraction 0.98 \
    --learning_rate -1 \
    --max_nrof_epochs 10000 \
    --keep_probability 0.6 \
    --learning_rate_schedule_file data/train2.txt \
    --weight_decay 5e-5 \
    --center_loss_factor 0.001 \
    --center_loss_alfa 0.9 \
    --batch_size 20
