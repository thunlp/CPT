if [ ! -d "data" ]; then
  mkdir data
fi
cd data

# download annotations
if [ ! -d "refcoco" ]; then
  wget https://thunlp.oss-cn-qingdao.aliyuncs.com/cpt-refcoco.tar.gz
  tar xzvf cpt-refcoco.tar.gz
  rm cpt-refcoco.tar.gz
fi


# download images
cd refcoco
if [ ! -d "train2014" ]; then
  wget https://thunlp.oss-cn-qingdao.aliyuncs.com/train2014.zip
  unzip train2014.zip
  rm train2014.zip
fi

