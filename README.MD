Tested on Jetson Orin Nano 8GB  
Jetpack 6.2.1 MAXN Super Power Mode  

# Install and enable jetson-stats 
sudo pip3 install -U jetson-stats  
sudo systemctl restart jetson-stats  

# Install CodeCarbon for tracking emission footprint
pip3 install codecarbon ultralytics pandas opencv-python  

chandra@jetson-chandra:~$ yolo checks  
Ultralytics 8.4.24 🚀 Python-3.10.12 torch-2.5.0a0+872d972e41.nv24.08 CUDA:0 (Orin, 7620MiB)  
Setup complete ✅ (6 CPUs, 7.4 GB RAM, 111.0/914.4 GB disk)  

OS                     Linux-5.15.148-tegra-aarch64-with-glibc2.35  
Environment            Linux  
Python                 3.10.12  
Install                pip  
Path                   /home/chandra/.local/lib/python3.10/site-packages/ultralytics  
RAM                    7.44 GB  
Disk                   111.0/914.4 GB  
CPU                    ARMv8 Processor rev 1 (v8l)  
CPU count              6  
GPU                    Orin, 7620MiB  
GPU count              1  
CUDA                   12.6  
  
numpy                  ✅ 1.26.4>=1.23.0  
matplotlib             ✅ 3.5.1>=3.3.0  
opencv-python          ✅ 4.11.0.86>=4.6.0  
pillow                 ✅ 9.0.1>=7.1.2  
pyyaml                 ✅ 5.4.1>=5.3.1  
requests               ✅ 2.25.1>=2.23.0  
scipy                  ✅ 1.8.0>=1.4.1  
torch                  ✅ 2.5.0a0+872d972e41.nv24.8>=1.8.0  
torch                  ✅ 2.5.0a0+872d972e41.nv24.8!=2.4.0,>=1.8.0; sys_platform == "win32"  
torchvision            ✅ 0.20.0a0+afc54f7>=0.9.0  
psutil                 ✅ 7.2.2>=5.8.0  
polars                 ✅ 1.39.2>=0.20.0  
ultralytics-thop       ✅ 2.0.18>=2.0.18  

