GPU_N=$1
TRAIN_PATH=vcr_train$1
N_SHOT=$2
SEED=$3
TASK_NAME=$4
METHOD_NAME=$5
EPOCHS=5
TEST_PATH=pt_vcr_val

if [ "$N_SHOT" -ge "16" ];then
	EPOCHS=10
fi

if [ "$N_SHOT" -ge "128" ];then
        EPOCHS=20
fi



echo $EPOCHS

CUDA_VISIBLE_DEVICES=$GPU_N python -m torch.distributed.launch \
	--master_port 1009$GPU_N --nproc_per_node=1 oscar/fewshot/vcr_qar_nsp_cpt.py -j 4 --img_feature_dim 2054 --max_img_seq_length 45 \
	--im_feat_dir ../prompt_feat/output/vcr/$TEST_PATH/inference/vcr_val \
	--train_im_feat_dir ../prompt_feat/output/vcr/$TRAIN_PATH/inference/vcr_train/ \
	--data_dir ../data/vcr\
	--result_dir results/vcr/$TASK_NAME/$METHOD_NAME/$N_SHOT/$SEED \
	--model_type bert  \
	--model_name_or_path pretrained_models/image_captioning/pretrained_base \
	--task_name $TASK_NAME --do_lower_case --max_seq_length 165 --per_gpu_eval_batch_size 4 \
	--per_gpu_train_batch_size 1 --learning_rate 6e-05 --num_train_epochs $EPOCHS  --output_dir output/vcr/fsl_color \
	--img_feature_type faster_r-cnn  \
	--loss_type xe --save_epoch 500 --seed 8 \
	--logging_steps 4000 --drop_out 0.3 --do_train --do_val --weight_decay 0.05 --warmup_steps 0 --eval_epoch 500
