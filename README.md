### Env. Setting
See [this document](https://bytedance.feishu.cn/docx/doxcnaaiohROuPs8yNZRaFjHxSg)

### Running
You can download datasets from [DVGO website](https://sunset1995.github.io/dvgo/)

Then running code for training:

```python
python run.py --config configs/nerf/mic.py --render_test
```


### TO DO

- [x] complete coarse training fully based on OpenVDB
- [x] realize 'openvdb_helper.so' with CPU
- [x] realize 'openvdb_helper.so' with CUDA
- [ ] try improving performance
- [ ] use "torch.h"
- [x] complete fine training fully based on OpenVDB
- [x] realize model loader funtion
- [ ] explore Adaptive-Loss Learning
- [ ] explore Voxel-Oriented Sampling
- [ ] evaluate performance



### PlenVDB在Arnold Workspace中的环境配置
1 Workspace创建
镜像配置可参考如下截图：
[图片]
配置gitlab：
开发环境配置初始化 - MacOS/Linux 

2 OpenVDB/NanoVDB环境配置
launch -- bash
git clone git@code.byted.org:yanhan.hans/PlenVDB.git
cd PlenVDB
bash setDefaultPython3.sh # 该镜像中python默认是python2
bash prepare_vdb.sh # 出现TEST PASSED就OK
cd openvdb
mkdir build
cd build
sudo apt-get install libboost-python1.74-dev
sudo apt-get install libboost-numpy1.74-dev
cmake -DOPENVDB_BUILD_NANOVDB=ON -DUSE_NUMPY=ON -DOPENVDB_BUILD_PYTHON_MODULE=ON ..
sudo make -j4
sudo make install
3 数据准备
cd DirectVoxGO
cd data
# bash download_data.sh # 好像会下载中断

去下列网页下载nerf_synthetic
https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
如下结构将数据放在data/目录下

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
4 DirectVoxGO/PlenVDB环境配置和运行
cd DirectVoxGO
pip install -r requirements.txt
apt-get update && apt-get install libgl1
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

cd lib/vdb/
mkdir build
cd build
cmake ..
make -j4
cd DirectVoxGO
# 运行时好像要编译torch的一些cpp代码，会花较长时间，需要耐心等待
# 运行DVGO
python run.py --config configs/nerf/mic.py --render_test > trydvgo.txt

# 运行PlenVDB
1.将configs/default.py中的density_type和k0_type（如下图）都由'DenseGrid'改为'VDBGrid'
2.将logs/nerf_synthetic/dvgo_mic改名为dvgo_mic_ori
3.执行：

python run.py --config configs/nerf/mic.py --render_test > tryplenvdb.txt
[图片]

环境配置细节（可不阅读）
修改默认python（个人推荐）
cd /usr/bin
ls -l | grep python  # 发现python指向的是python2

sudo rm -rf python
sudo ln -s /usr/bin/python3 /usr/bin/python

python  # 进入python3.7表示成功修改
配置Openvdb
# cmake安装参考[1]，一般在Arnold workspace中应该都装好了
launch -- bash  # 推荐在worker中进行，不然有些编译过程会导致崩溃？
cd ~

# 安装libboost-iostreams
sudo apt-get install aptitude  # 会安装默认的libboost-iostreams1.67.0
aptitude search boost  # 需要有1.70.0及以上版本，没有的话参考[4]自己装
apt list --installed  # 列举已安装的libararies
sudo apt-get autoremove libboost-iostreams1.67.0  # 删除已安装的旧版本
sudo apt-get install libboost-iostreams1.74-dev
# find / -name libboost*

# 安装libtbb
wget https://github.com/oneapi-src/oneTBB/archive/refs/tags/2019_U9.tar.gz
tar -zxvf 2019_U9.tar.gz
cd oneTBB-2019_U9/
make  # 我没遇到过错误，如果有报错参考[5]
cd build  # 添加tbb环境变量
chmod +x *.sh
sh generate_tbbvars.sh
sh tbbvars.sh
cd linux_intel64_gcc_cc8.3.0_libc2.28_kernel5.4.143.bsk.6_release/
cp *.so /usr/lib/x86_64-linux-gnu/
cp *.so.2 /usr/lib/x86_64-linux-gnu/
ldconfig
cd ~/oneTBB-2019_U9/  # 回到解压缩目录下
cp -r include/* /usr/include
cd examples  # 测试
make # 经过一段时间后无报错，则成功安装

# 安装libblosc
apt-get install -y libblosc-dev

# 安装openvdb
cd ~
mkdir code
cd code
git clone https://github.com/AcademySoftwareFoundation/openvdb.git
cd openvdb
mkdir build
cd build
cmake ..
sudo make -j4
sudo make install
如果想要安装python和numpy版的，或nanovdb，参考如下命令
# sudo apt-get install libboost-python1.74-dev
# sudo apt-get install libboost-numpy1.74-dev

# cd build
# cmake -DOPENVDB_BUILD_PYTHON_MODULE=ON ..

# cd build
# cmake -DUSE_NUMPY=ON ..

cd build
cmake -DOPENVDB_BUILD_NANOVDB=ON ..
sudo make -j4
sudo make install
之后，会在/usr/local/include，/usr/local/lib中生成代码和静态动态库。
Examples
main.cpp
CMakeLists.txt
build/
// main.cpp
#include <openvdb/openvdb.h>
#include <iostream>
int main(){}
# CMakefile.txt
project(plenvdb)
set (CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
LINK_DIRECTORIES(${CMAKE_INSTALL_PREFIX}/lib /usr/lib/x86_64-linux-gnu)

set(PLENVDB_SRC_FILES main.cpp)
add_executable(plenvdb ${PLENVDB_SRC_FILES})

set(PLENVDB_DYNAMIC_LIBS
    libopenvdb.so
    libtbb.so)
target_link_libraries(plenvdb ${PLENVDB_DYNAMIC_LIBS})
DirectVoxGO
VDBNeRF将在DirectVoxGO基础上修改实现，配置DirectVoxGO环境需要安装如下库
# packages
scipy
tqdm
lpips
mmcv
imageio
imageio-ffmpeg
opencv-python
torch_efficient_distloss
ninja
einops
# nvcc -V command not found
vim ~/.bashrc
# 添加 export PATH=$PATH:/usr/local/cuda/bin
source ~/.bashrc

# 安装torch_scatter cuda版
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# 如果遇到如下错误，需要安装libgl1库
# ImportError: libGL.so.1: cannot open shared object file: No such file or directory
apt-get update && apt-get install libgl1
pip install pybind11
参考文档
[1] https://zhuanlan.zhihu.com/p/110793004
[2] https://blog.csdn.net/Romance5201314/article/details/81667778
[3] https://codeantenna.com/a/pE4u2N8TO1
[4] https://stackoverflow.com/questions/12578499/how-to-install-boost-on-ubuntu
[5] https://www.jianshu.com/p/57b67477ff53
[6] https://blog.csdn.net/hp_cpp/article/details/110404651
[7] https://www.cxymm.net/article/weixin_43046653/100019901