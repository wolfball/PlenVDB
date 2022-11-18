basedir=$PWD

# echo "=========================="
# echo "install libboost-iostreams"
# echo "=========================="
# sudo apt-get autoremove libboost-iostreams1.67.0  # 删除已安装的旧版本
# sudo apt-get install libboost-iostreams1.74-dev

# echo "=========================="
# echo "     install libtbb"
# echo "=========================="
# wget https://github.com/oneapi-src/oneTBB/archive/refs/tags/2019_U9.tar.gz
# tar -zxvf 2019_U9.tar.gz
cd oneTBB-2019_U9/
make
wait
cd build  # add tbb env variables
chmod +x *.sh
sh generate_tbbvars.sh
sh tbbvars.sh
wait
cd linux_intel64_gcc_cc8.3.0_libc2.28_kernel5.4.143.bsk.6_release/
cp *.so /usr/lib/x86_64-linux-gnu/
cp *.so.2 /usr/lib/x86_64-linux-gnu/
ldconfig
wait
cd $basedir/oneTBB-2019_U9/ 
cp -r include/* /usr/include
wait
cd examples  # test
make # it will be a long time...

echo "=========================="
echo "    install libblosc"
echo "=========================="
apt-get install -y libblosc-dev