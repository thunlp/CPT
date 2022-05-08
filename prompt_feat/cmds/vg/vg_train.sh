rm -rf output/vg/train/

GPU=$1
N_SHOT=$2
RAND_SEED=$3

CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch \
	--master_port 1009$GPU --nproc_per_node=1 tools/test_vg_net.py \
	--config-file sgg_configs/vgattr/vinvl_colorft.yaml \
	TEST.IMS_PER_BATCH 1 \
	MODEL.RPN.FORCE_BOXES True \
	MODEL.ROI_BOX_HEAD.FORCE_BOXES True \
	MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth \
	MODEL.ROI_HEADS.NMS_FILTER 2 \
	MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
	DATASETS.TEST \(\"VGDataset\",\)   \
	DATA_DIR "data/vg/yamls/train.yaml" \
	TEST.IGNORE_BOX_REGRESSION False \
	MODEL.CLS_AGNOSTIC_BBOX_REG False \
	MODEL.ATTRIBUTE_ON True \
	TEST.OUTPUT_FEATURE True \
	OUTPUT_DIR "output/vg/train/$GPU"  N_SHOT $N_SHOT RAND_SEED $RAND_SEED
