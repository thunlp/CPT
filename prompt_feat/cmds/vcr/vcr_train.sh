GPU=$1
OUTPATH=output/vcr/vcr_train$1
NSHOT=$2
SEED=$3

rm -rf $OUTPATH

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch \
	--master_port 1009$GPU --nproc_per_node=1 tools/test_vcr_net.py \
	--config-file sgg_configs/vgattr/vinvl_colorft.yaml \
	TEST.IMS_PER_BATCH 1 \
	MODEL.RPN.FORCE_BOXES True \
	MODEL.ROI_BOX_HEAD.FORCE_BOXES True \
	MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth \
	MODEL.ROI_HEADS.NMS_FILTER 2 \
	MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
	DATASETS.TEST \(\"VCRColorDataset\",\)   \
	DATA_DIR "data/vcr/yamls/train.yaml" \
	TEST.IGNORE_BOX_REGRESSION False \
	MODEL.CLS_AGNOSTIC_BBOX_REG False \
	MODEL.ATTRIBUTE_ON True \
	TEST.OUTPUT_FEATURE True \
	OUTPUT_DIR $OUTPATH  N_SHOT $NSHOT RAND_SEED $SEED COLOR_D $4
