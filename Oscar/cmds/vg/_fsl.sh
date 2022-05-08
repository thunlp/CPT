GPU=$1
NSHOT=$2
SEED=$3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
	--master_port 1009$1 --nproc_per_node=4  oscar/fewshot/vg_cpt.py \
	--model_name_or_path pretrained_models/image_captioning/pretrained_base \
	--eval_model_dir pretrained_models/image_captioning/pretrained_base \
	--do_train \
	--vocab ../data/vg/vg.json \
	--annotation ../data/vg/vg_val.pk \
	--train_dir "../prompt_feat/output/vg/train/$GPU/inference/vg_train" \
	--test_dir ../prompt_feat/output/vg/val/inference/vg_val \
	--result_dir "results/vg/cpt/fsl/$NSHOT/$SEED" \
	--do_lower_case \
	--add_od_labels \
	--per_gpu_train_batch_size 10 \
	--per_gpu_eval_batch_size 16 \
	--num_train_epochs 200 \
	--tie_weights \
	--freeze_embedding \
  --label_smoothing 0.1 \
  --drop_worst_ratio 0.2 \
	--drop_worst_after 20000
