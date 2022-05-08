# if you modify the number of GPU, do not forget to modify the `TEST.IMG_PER_BATCH` and `--nproc_per_node`
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
  --master_port 10094 --nproc_per_node=4  tools/test_sg_net.py \
	--config-file sgg_configs/vgattr/vinvl_x152c4.yaml \
	TEST.IMS_PER_BATCH 16 \
	MODEL.WEIGHT models/vinvl/vinvl_vg_x152c4.pth \
	MODEL.ROI_HEADS.NMS_FILTER 1 \
	MODEL.ROI_HEADS.SCORE_THRESH 0.2 \
	DATASETS.TEST \(\"ImgDataset\",\)   \
	DATA_DIR "../data/gqa/images" \
	TEST.IGNORE_BOX_REGRESSION True \
	MODEL.ATTRIBUTE_ON True \
	TEST.OUTPUT_FEATURE True \
	OUTPUT_DIR "./output/gqa/gqa_img_feats/"


