# User Profiling On-Road-V2x Usecase-QCS610
## Introduction
This project is intended to build and deploy a Use Profiling on Road using Tiny-YoloV3 model on Thundercomm TurboX QCS610 Open Kit, a smart camera from Qualcomm, which is deployed on roadside light poles which will be predicting Human activities whether standing or walking on road. The inference using SNPE on QCS610 with DSP Hardware accelerator, it achieves the performance of 56.67 FPS. 

## Prerequisites 
1. Download the proprietary software from Qualcomm Chip Code. Follow the link provided (https://chipcode.qti.qualcomm.com/qualcomm/qcs610-le-1-0_ap_standard_oem/tree/r00071.1). 

2. Install Android Platform tools (ADB, Fastboot) on the host system. 

3. Flash the firmware image onto the board, refer the link below, (https://developer.qualcomm.com/qualcomm-robotics-kit/learning-resources/flashing-new-complete-thundercomm-image-board). 

4. Setup the Wi-Fi,refer the link given below, (https://developer.qualcomm.com/qualcomm-robotics-kit/learning-resources/setting-up-wifi) 

5. Download and Install the Application Tool chain SDK. (Setup instructions can be found in Application SDK User Manual). 

6. Setup the SNPE SDK (https://developer.qualcomm.com/sites/default/files/docs/snpe/setup.html) in the host system. You can download SNPE SDK here. (https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk) 


## Steps to build and deploy User Profiling On-Road Application 

1. Create the folder on host for keeping the dependencies. 
```sh
   $ mkdir User_Profiling
```
2. Clone the project on the host system.
```sh
   $ git clone <source repository>  
   $ cd User_Profiling/ 
```
3. Follow the steps below for creating the dependencies folders. 
```sh
   $ mkdir opencv  
   $ mkdir pybind11 
```  
### 1. Steps to build OpenCV Library: 

This allows us to cross build the OpenCV for the target platform, using the Yocto Environment & Bitbake file of OpenCV 
   1. Yocto environment setup can be done after completing the pre-requisite – 1. 
   2. Bitbake of OpenCV can be found in “qcs610-le-1-0_ap_standard_oem/apps_proc/poky/meta-openembedded/meta-oe/recipes-support/OpenCV/ opencv_3.4.5.bb”. 

1.Source the Yocto environment on the host system.
```sh
 <working directory>$ export SHELL=/bin/bash 
 <working directory>$ source poky/qti-conf/set_bb_env.sh 
```
While sourcing, select the option given in menu, 
   1. Machine Menu - Select “qcs610-odk meta-qti-bsp” 
   2. Distribution Menu – Select “qti-distro-fullstack-perf meta-qti-bsp” 

Above step, generates the Yocto build directory and makes it as present working directory in shell. 
PATH: <working_directory>/build-qti-distro-fullstack-perf$  
The path mentioned above is referred as <build_directory> in next steps. 

2.Run the command below, to build the OpenCV: 
```sh
<build_directory>$ bitbake python3-opencv 
```
It utilizes the OpenCV bitbake file for building the same. 

3.After completion of build, shared libraries & header files of OpenCV will be available on the path given below, 
```sh
“./tmp-glibc/sysroots-components/armv7ahf-neon/python3-opencv/usr” 
```
4.Copy these files to the OpenCV directory in User_Profiling folder on host system. 
```sh
$ cp ./tmp-glibc/sysroots-components/armv7ahf-neon/python3-opencv/usr/lib  ~/User_Profiling/opencv 
```
`Note: For more information, refer to the “QCS610/QCS410 Linux Platform Development Kit Quick Start Guide document”. Also make sure that all the dependencies from the Yocto build has installed on the system (ex: libgphoto2, libv4l-utils). Bitbake recipes of above dependent libraries are available inside meta-oe layer, can be built directly using bitbake command.`
        
### 2. Steps to Build Pybind11 Library: 
For building Pybind11 package, Follow the same steps used to build OpenCV library up to step 1. Next steps should be followed has given below. 
1. Run the command below, to build the Pybind11 
```sh
   <build_directory>$ bitbake python3-pybind11 
```
It utilizes the Pybind11 bitbake file for building the same.

2. After completion of build, shared libraries & header files of Pybind11 will be available on the path given below,
```sh
   “./tmp-glibc/sysroots-components/armv7ahf-neon/python3-pybind11/usr”  
```  
3. Copy these files to the Pybind11 directory in User_Profiling folder on host   system. 
```sh
   $ cp ./tmp-glibc/sysroots-components/armv7ahf-neon/python3-pybind11/usr/lib  ~/User_Profiling/pybind11
```
### 3. Steps to Build Main Application: 

1. Move to the project directory in the host system by following the below steps. 
```sh
   $ cd ~/User_Profiling/snpe/Makefile.arm-oe-linux-gcc8.2hf 
```
2. In Makefile.arm-oe-linux-gcc8.2hf Change <PATH_TO> to proper required path.

3. Run the command below to build SNPE Wrapper. 
```sh
   $ make -f Makefile.arm-oe-linux-gcc8.2hf 
```
4. Above command will build qcsnpe.so file in <OBJ_DIR> path specified in makefile. 

5. Now push this project directory containing executable and other necessary files from host sysem to the target board. 
```sh
   $ adb push User_Profiling/ /data/User_Profiling/ 
```
`Note: Check for SNPE_ROOT, by executing following command. If not, setup SNPE as specified in prerequisite.`
```sh
$ export SNPE_ROOT= $SNPE_ROOT 
```
  `Make sure that have given proper path of build libraries (OpenCV and Pybind11) and include files in makefile.`

### 4. Steps to Run User Profiling On-Road on kit: 
Before running the main application set the board environment by following below steps: 

1. In the host system, enter following adb commands 
```sh
   $ adb root 
   $ adb remount 
   $ adb shell mount -o remount
``` 
2. Enable Wifi connectivity on the board by executing following commands 
```sh
   $ adb shell 
   /# wpa_supplicant -Dnl80211 -iwlan0 -c /etc/misc/wifi/wpa_supplicant.conf -ddddt & 
   /# dhcpcd wlan0 
```
3. Set the latest UTC time and date in the following format. 
```sh
   /# date -s '2022-08-22 10:07:00' 
```
4. Copy the Library files of snpe-1.68.0.3230 on to the kit from host system. 
```sh
   $ adb push <SNPE_ROOT>/lib/aarch64-oe-linux-gcc8.2/ /usr/lib/ 
```
5. Copy the DSP files of snpe-1.68.0.3230 on to the kit from host system. 
```sh
   $ adb push <SNPE_ROOT>/lib/dsp/ /usr/lib/rfsa/adsp/ 
```
6. Copy the bin files of snpe snpe-1.68.0.3230 on to the kit from host system. 
```sh
   $ adb push <SNPE_ROOT>/bin/aarch64-oe-linux-gcc8.2/ /usr/bin/ 
```
7. Export the shared library to the LD_LIBRARY_PATH 
```sh
   a.  /#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/User_Profiling/opencv/

   b.  /#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/User_Profiling/lib/ 
```
Steps for running the main application are as follows: 

1. Access the board through adb 
```sh
   $ adb shell 
```
2. Move to User_Profiling directory 
```sh
   /# cd /data/User_Profiling 
```
3. Run the application  
```sh
   /# python3 inference.py --img_folder/vid path_to_image_folder/video
```
4. Pull the saved output video from board to host system 
```sh
   $ adb pull /data/User_Profiling/reference_video.mp4
``` 
