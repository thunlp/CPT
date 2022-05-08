if [ ! -d "data" ]; then
  mkdir data
fi
cd data

# download annotations
if [ ! -d "vcr" ]; then
  wget https://thunlp.oss-cn-qingdao.aliyuncs.com/cpt-vcr.tar.gz
  tar xzvf cpt-vcr.tar.gz
  rm cpt-vcr.tar.gz
fi


# download images
cd vcr
if [ ! -d "vcr1images" ]; then
  wget https://s3.us-west-2.amazonaws.com/ai2-rowanz/vcr1images.zip
  unzip vcr1images.zip
  rm vcr1images.zip
fi

