basedir=$PWD
cd /usr/bin
# ls -l | grep python
sudo rm -rf python
sudo ln -s /usr/bin/python3 /usr/bin/python
cd $basedir