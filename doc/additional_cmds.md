# Additional Commands

## Install g++>=6.3.1, cmake>=3.18

If the g++ version is not matching, refer to https://zhuanlan.zhihu.com/p/261001751.
```bash
wget https://cmake.org/files/v3.18/cmake-3.18.0-Linux-x86_64.tar.gz
cmake -DOPENVDB_BUILD_NANOVDB=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 ..
```

## Install Boost>=1.70
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

## tbb>=2019.0
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

## libblosc

```
apt-get install -y libblosc-dev
```

## Compile OpenVDB, NanoVDB and PyOpenVDB

```bash
# if there is some thing wrong, try the following command
cmake -DOPENVDB_BUILD_NANOVDB=ON -DCMAKE_C_COMPILER=/usr/bin/gcc-7 -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 -DOPENVDB_BUILD_PYTHON_MODULE=ON -DCMAKE_PREFIX_PATH=/usr/local/boost_1_74_0/ -DUSE_NUMPY=ON ..
```