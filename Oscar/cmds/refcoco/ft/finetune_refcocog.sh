#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch \
	--master_port 10097 --nproc_per_node=4	 oscar/finetune/normal_finetune.py \
	--model_name_or_path pretrained_models/image_captioning/pretrained_base \
	--eval_model_dir pretrained_models/image_captioning/pretrained_base \
	--do_train \
	--max_img_seq_length 75 \
	--train_dir /data_local/zhangao/codes/prompt_feat/output/normal_finetune/inference/refcocog_train/ \
	--test_dir /data_local/zhangao/codes/prompt_feat/output/normal_finetune/inference/refcocog_val/ \
	--do_lower_case \
	--add_od_labels \
	--learning_rate 0.02 \
	--per_gpu_train_batch_size 8 \
	--per_gpu_eval_batch_size 32 \
	--num_train_epochs 60 \
	--tie_weights \
	--freeze_embedding \
        --label_smoothing 0.1 \
    	--drop_worst_ratio 0.2 \
	--drop_worst_after 20000 \
	--output_dir output/normal_finetune/refcocog