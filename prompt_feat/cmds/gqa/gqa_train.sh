rm -rf output/gqa/train/*
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9  python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
	--master_port 10093 --nproc_per_node=4 tools/test_vcr_net.py \
	--config-file sgg_configs/vgattr/vinvl_colorft.yaml \
	TEST.IMS_PER_BATCH 32 \
	MODEL.RPN.FORCE_BOXES True \
	MODEL.ROI_BOX_HEAD.FORCE_BOXES True \
	MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth \
	MODEL.ROI_HEADS.NMS_FILTER 2 \
	MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
	DATASETS.TEST \(\"GQAColorDataset\",\)   \
	DATA_DIR "data/gqa/yamls/train.yaml" \
	TEST.IGNORE_BOX_REGRESSION False \
	MODEL.CLS_AGNOSTIC_BBOX_REG False \
	MODEL.ATTRIBUTE_ON True \
	TEST.OUTPUT_FEATURE True \
	OUTPUT_DIR "./output/gqa/train" # N_SHOT $1 RAND_SEED $2
