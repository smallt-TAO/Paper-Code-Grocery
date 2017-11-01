py=/export/App/anaconda-tf-1.0/bin/python2.7

$py src/launcher.py \
	--phase=test \
        --visualize   \
	--data-path=dataset/test10/ann.txt  \
	--data-base-dir=./dataset/test10  \
	--log-path=log_01_16_test.log  \
	--attn-num-hidden 800 \
	--batch-size 64 \
	--model-dir=model_new_800 \
	--load-model \
	--num-epoch=3 \
	--gpu-id=1 \
	--output-dir=model_01_16/synth90 \
	# --use-gru \
        --target-embedding-size=20
