rm output/refcoco/cpt/inference/refcoco_testB/*

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
	--master_port 10099 --nproc_per_node=4 tools/test_refcoco_net.py \
	--config-file sgg_configs/vgattr/vinvl_x152c4.yaml \
	TEST.IMS_PER_BATCH 8 \
	MODEL.RPN.FORCE_BOXES True \
	MODEL.ROI_BOX_HEAD.FORCE_BOXES True \
	MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth \
	MODEL.ROI_HEADS.NMS_FILTER 2 \
	MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
	DATASETS.TEST \(\"RefCoCoDataset\",\)   \
	DATA_DIR "data/refcoco/yamls/refcoco_testB.yaml" \
	TEST.IGNORE_BOX_REGRESSION False \
	MODEL.CLS_AGNOSTIC_BBOX_REG False \
	MODEL.ATTRIBUTE_ON True \
	TEST.OUTPUT_FEATURE True \
	OUTPUT_DIR "./output/refcoco/cpt"
