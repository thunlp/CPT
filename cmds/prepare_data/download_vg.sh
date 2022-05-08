if [ ! -d "data" ]; then
  mkdir data
fi
cd data

# download annotations
if [ ! -d "vg" ]; then
  wget https://thunlp.oss-cn-qingdao.aliyuncs.com/cpt-vg.tar.gz
  tar xzvf cpt-vg.tar.gz
  rm cpt-vg.tar.gz
fi


# download images
cd vg
if [ ! -d "VG_100K" ]; then
  wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
  wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
  unzip images.zip
  unzip images2.zip
  mv VG_100K_2/* VG_100K
  rm -rf VG_100K_2
  rm images.zip
  rm images2.zip
fi

