# PlenVDB: A Memory Efficient VDB-Based Radiance Fields for Fast Training and Rendering



## File tree

.PlenVDB  
├── openvdb  
├── plenvdb  
├── README.md  

## Requirements

- Ubuntu, python==3.7, g++>=6.3.1, gcc>=6.3.1, CMake>=3.18.0, Boost>=1.70, TBB>=2019.0, Blosc>=1.7.0
- pytorch and torch-scatter is dependent on CUDA, please install the correct version for your machine

First, let's compile OpenVDB, NanoVDB and the python module. We mainly focus on **PlenVDB/openvdb** directory, which is an old version of [OpenVDB library](https://github.com/AcademySoftwareFoundation/openvdb). We have tested on g++7.5.0, gcc7.5.0, make3.18.0,libboost1.74 tbb2019 and libblosc1.7.0. And you can go to **PlenVDB/openvdb/cmake/config/OpenVDBVersions.cmake** for detailed version requirements about the dependencies. If you have some trouble with the dependencies, we hope the commands in the **[Additional Commands](#Additional-Commands)** will help.

When all dependencies are ready, run

```bash
cd PlenVDB/openvdb
mkdir build
cd build
cmake -DOPENVDB_BUILD_NANOVDB=ON -DOPENVDB_BUILD_PYTHON_MODULE=ON -DUSE_NUMPY=ON ..
# if there is some thing wrong, try the following command
# cmake -DOPENVDB_BUILD_NANOVDB=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 -DOPENVDB_BUILD_PYTHON_MODULE=ON -DCMAKE_PREFIX_PATH=/usr/local/boost_1_74_0/ -DUSE_NUMPY=ON ..
sudo make -j4
sudo make install
```

Second, let's create an environment for running. Here we give the CUDA10.2 version.

```bash
# cd PlenVDB/plenvdb/
conda create -n plenvdb python=3.7
conda activate plenvdb
pip install -r requirements.txt
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/  pytest
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cu102.html
pip install imageio-ffmpeg
```

Third, let's compile the plenvdb.so, which is a VDB-based data structure.

```bash
cd PlenVDB/plenvdb/lib/vdb/
mkdir build
cd build
cmake ..
# cmake  -DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 ..
```

If you meet with some difficulties in compilation, I recommende you to read its [repository](https://github.com/AcademySoftwareFoundation/openvdb) or [document](https://www.openvdb.org/documentation/doxygen/build.html). Plus, hope the commands that I've used in the following will help.

## Datasets

Download [NeRF-Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), [NSVF](https://dl.fbaipublicfiles.com/nsvf/dataset/Synthetic_NSVF.zip), [BlendedMVS](https://dl.fbaipublicfiles.com/nsvf/dataset/BlendedMVS.zip), [DeepVoxels](https://drive.google.com/open?id=1ScsRlnzy9Bd_n-xw83SP-0t548v63mPH) and put them under PlenVDB/plenvdb/data/.

Take NeRF-Synthetic for example:

.data
├── download_data.sh
└── nerf_synthetic
    └── mic
        ├── test
        ├── train
        ├── transforms_test.json
        ├── transforms_train.json
        ├── transforms_val.json
        └── val

## Training

Run for mic

```bash
# cd PlenVDB/plenvdb/
python run.py --config configs/nerf/mic.py # --render_test
python vdb_compression.py --basedir logs/nerf_synthetic/ --scenes mic
python run.py --config configs/nerf/mic.py --render_test --render_only --cps
```



## Acknowledgement

This code is built upon the publicly available code [DVGO](https://github.com/sunset1995/DirectVoxGO) and [OpenVDB](https://github.com/AcademySoftwareFoundation/openvdb). Thanks the authors of DVGO and OpenVDB for making their excellent work and codes publicly available.



## Citation

Please cite the following paper if you use this repository in your reseach.





## Additional Commands

### Install g++>=6.3.1, cmake>=3.18

If the g++ version is not matching, refer to https://zhuanlan.zhihu.com/p/261001751.
```bash
wget https://cmake.org/files/v3.18/cmake-3.18.0-Linux-x86_64.tar.gz
cmake -DOPENVDB_BUILD_NANOVDB=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 ..
```

### Install Boost>=1.70
```bash
# First method:
# Try the following two commands, if such version not found, turn to the second method.
sudo apt-get install libboost-python1.74-dev
sudo apt-get install libboost-numpy1.74-dev

# Second method:
# refer to https://blog.csdn.net/chen411120086/article/details/122618226
# refer to https://zhuanlan.zhihu.com/p/418098249
wget -O boost_1_74_0.tar.gz https://sourceforge.net/projects/boost/files/boost/1.74.0/boost_1_74_0.tar.gz/download
tar xzvf boost_1_74_0.tar.gz
cd boost_1_74_0/
./bootstrap.sh --prefix=/usr/local/boost_1_74_0/ --with-libraries=all --with-python-version=3.7 --with-python=/usr/bin/python3.7 --with-python-root=/usr/lib/python3.7/
./b2 include=/usr/include/python3.7m
sudo ./b2 --prefix=/usr/local/boost_1_74_0/ install # remember to make sure that /usr/bin/python->/usr/bin/python3.7
vim ~/.bashrc
# Add:
#     CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/boost_1_74_0/include
#     LIBRARY_PATH=$LIBRARY_PATH:/usr/local/boost_1_74_0/lib
#     export LIBRARY_PATH CPLUS_INCLUDE_PATH
source ~/.bashrc
sudo ldconfig /usr/lcoal/boost_1_74_0/lib/
```

### tbb>=2019.0
```bash
wget https://github.com/oneapi-src/oneTBB/archive/refs/tags/2019_U9.tar.gz
tar -zxvf oneTBB-2019_U9.tar.gz
cd oneTBB-2019_U9/
make
cd build
chmod +x *.sh
sh generate_tbbvars.sh
sh tbbvars.sh
cd linux_intel64_gcc_cc7.5.0_libc2.23_kernel4.15.0_release/ # the filename might be different on yr machine
cp *.so /usr/lib/x86_64-linux-gnu/
cp *.so.2 /usr/lib/x86_64-linux-gnu/
ldconfig
cd ~/oneTBB-2019_U9/
cp -r include/* /usr/include
cd examples
make
```

### libblosc

```
apt-get install -y libblosc-dev
```