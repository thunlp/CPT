# you can direcly run by 
# bash install.sh

# create a new environment
conda create --name cpt python=3.7
conda activate cpt

# install pytorch1.6
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

export INSTALL_DIR=$PWD

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
cd ..

# install requirements
pip install -r requirements.txt

# install prompt_feat
cd prompt_feat
python setup.py build develop
cd ..

# install oscar
cd Oscar
# install transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git reset --hard 067923d3267325f525f4e46f357360c191ba562e
cd ..
# install coco_caption
git clone https://github.com/LuoweiZhou/coco-caption.git
cd coco-caption
git reset --hard de6f385503ac9a4305a1dcdc39c02312f9fa13fc
# ./get_stanford_models.sh
cd ..
python setup.py build develop

unset INSTALL_DIR
