cd Oscar/
#wget https://biglmdiag.blob.core.windows.net/vinvl/model_ckpts/image_captioning/pretrained_base.zip
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/pretrained_base.zip
unzip pretrained_base.zip
mkdir -p pretrained_models/image_captioning/pretrained_base/
mv pretrained_base/checkpoint-2000000/* pretrained_models/image_captioning/pretrained_base/
rm -rf pretrained_base
rm pretrained_base.zip
cd ..

cd prompt_feat/
mkdir -p models/vinvl
cd models/vinvl
#wget https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/vinvl_model_zoo/vinvl_vg_x152c4.pth
wget https://thunlp.oss-cn-qingdao.aliyuncs.com/vinvl_vg_x152c4.pth
cd ../../..


