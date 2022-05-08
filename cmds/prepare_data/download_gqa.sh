
if [ ! -d "data" ]; then
  mkdir data
fi
cd data
# download annotations
if [ ! -d "gqa" ]; then
  wget https://thunlp.oss-cn-qingdao.aliyuncs.com/cpt-gqa.tar.gz
  tar xzvf cpt-gqa.tar.gz
  rm cpt-gqa.tar.gz
fi


# download images
cd gqa
if [ ! -d "images" ]; then
  wget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip
  unzip images.zip
  rm images.zip
fi

