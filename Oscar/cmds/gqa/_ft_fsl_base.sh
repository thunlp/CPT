N_SHOT=$1
SEED=$2
USE_COLOR=0
EPOCH=200
METHOD_NAME="ft"

if [ "$N_SHOT" -ge "16" ];then
	EPOCH=$EPOCH
fi


RDIR=results/gqa/test_dev/$METHOD_NAME/$N_SHOT/$SEED

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
	--master_port 10093 --nproc_per_node=4 oscar/fewshot/gqa_ft.py -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
	--img_feat_file ../prompt_feat/output/gqa/gqa_img_feats/predictions.tsv \
	--testdev_color_img_feat_file ../prompt_feat/output/gqa/test-dev/inference/gqa_bal_qla_testdev/predictions.tsv \
	--train_color_img_feat_file ../prompt_feat/output/gqa/train/inference/gqa_all_qla_train_sub/predictions.tsv \
	--data_dir ../data/gqa \
	--label_file datasets/gqa/trainval_testdev_all_label2ans.pkl \
	--model_type bert  \
	--model_name_or_path pretrained_models/image_captioning/pretrained_base \
	--task_name gqa --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 16 \
	--per_gpu_train_batch_size 2 --learning_rate 6e-05 --num_train_epochs $EPOCH --output_dir output/gqa/fsl \
	--img_feature_type faster_r-cnn  \
	--loss_type xe --save_epoch 10000 --seed 8 --eval_epoch 1000 \
	--logging_steps 4000 --drop_out 0.3 --do_eval --do_train --weight_decay 0.05 --warmup_steps 0 \
	--n_sample $N_SHOT --random_seed $SEED --use_color $USE_COLOR --result_dir $RDIR
