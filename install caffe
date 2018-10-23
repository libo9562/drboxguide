The following guide includes the how-to instructions for the installation of BVLC/Caffe on Ubuntu 16.04 with Cuda Toolkit 8.0, CUDNN 5.1 library and OpenCV version 2 or 3. (A small record remains from the previous tutorial for Ubuntu 15.10 with the Cuda Toolkit 7.5, but that part will not be updated any further.) This guide also covers the KUbuntu distribution and the related distributions.

Execute these commands first:
```
sudo apt-get update

sudo apt-get upgrade

sudo apt-get install -y build-essential cmake git pkg-config

sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler

sudo apt-get install -y libatlas-base-dev 

sudo apt-get install -y --no-install-recommends libboost-all-dev

sudo apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev
```

# (Python general)
```
sudo apt-get install -y python-pip
```

# (Python 2.7 development files)
```
sudo apt-get install -y python-dev
sudo apt-get install -y python-numpy python-scipy
```

# (or, Python 3.5 development files)
```
sudo apt-get install -y python3-dev
sudo apt-get install -y python3-numpy python3-scipy
```
 
# (OpenCV 2.4)
```
sudo apt-get install -y python-opencv
```

(or, OpenCV 3.2 - see the instructions below)
If you own an NVIDIA graphics card, see the instructions for the installation of NVIDIA Graphics Driver, Cuda Toolkit and CUDNN library at the end of this document, or by clicking https://github.com/BVLC/caffe/wiki/Ubuntu-16.04-or-15.10-Installation-Guide#the-gpu-support-prerequisites.

For the instructions on how to use the OpenCV version 3.2, please see https://github.com/BVLC/caffe/wiki/OpenCV-3.2-Installation-Guide-on-Ubuntu-16.04

The configuration settings will differ for OpenCV 3.2 as you can see above, but here we continue with the basic settings that imply the version 2.x.

Go to the https://github.com/BVLC/caffe and download the zip archive. Unpack it to `~/bin/` or any other location. Enter the caffe-master directory in the terminal window. Note that only the version 1.0RC5 compiles well at this moment, so download the 1.0RC5 version from https://github.com/BVLC/caffe/archive/rc5.zip. If you download the 1.0 version from https://github.com/BVLC/caffe/archive/1.0.zip, you need to edit the file `/src/caffe/util/blocking_queue.cpp`. After the line 89, add the new line that contains the following:

```
template class BlockingQueue<Datum*>;
```

Copy the Makefile.config.example to Makefile.config like this:

```
cp Makefile.config.example Makefile.config
```
and open it for editing (with a text editor). I use the kate editor for this purpose, so the command that I execute goes as follows. You first need to install the kate editor with:

```
sudo apt-get install kate
```
and then you can edit the configuration file with:

```
kate ./Makefile.config &
```
The following line in the configuration file tells the program to use CPU only for the computations.

```
CPU_ONLY := 1
```
CPU_ONLY option is enabled for a computer without any NVIDIA graphics card and it is typical for the installation of Caffe inside the typical virtual machine. (Notice that there is a special type of virtual machine inside the Ubuntu host machine that can access the physical NVIDIA graphics card directly. See https://github.com/NVIDIA/nvidia-docker)

The default option value is to use GPU and CPU computation. Change the line if needed, by commenting it out (# CPU_ONLY := 1) if you have an NVIDIA graphics card with the proprietary driver, CUDA toolkit and CUDNN installed. Jump to the end of this guide to read about how to install the GPU support prerequisites.

The Makefile.config should contain the following lines, so find them and fill them in.

```
PYTHON_INCLUDE := /usr/include/python2.7 /usr/lib/python2.7/dist-packages/numpy/core/include
```
(for some Ubuntu 16.04 users, the path may be different)

```
PYTHON_INCLUDE := /usr/include/python2.7 /usr/local/lib/python2.7/dist-packages/numpy/core/include  
WITH_PYTHON_LAYER := 1  
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial  
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial  
```
(For ways to create an isolated Python environment, explore the topic of virtual environments here: http://docs.python-guide.org/en/latest/dev/virtualenvs/)

If your CUDA is 8.0, find this in Makefile.config..

```
CUDA_DIR := /usr/local/cuda
```
And replace it with this..

```
CUDA_DIR := /usr/local/cuda-8.0
```
Now lets continue with the instructions for the Ubuntu 15.10 first, followed by the instructions for Ubuntu 16.04 users.

Execute the additional commands:

```
find . -type f -exec sed -i -e 's^"hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;
```

```
cd /usr/lib/x86_64-linux-gnu

sudo ln -s libhdf5_serial.so.8.0.2 libhdf5.so

sudo ln -s libhdf5_serial_hl.so.8.0.2 libhdf5_hl.so
```
The above commands are no longer needed on Ubuntu 16.04 after a certain system update. If your Ubuntu 16.04 still needs this step to be performed, take the following considerations.

The file versions for libhdf5_serial.so and libhdf5_serial_hl.so are different and so the last two lines will need to be altered. Visit `/usr/lib/x86_64-linux-gnu/` and list the relevant contents of that directory by using the command such as `ls -l | grep hdf5`. The versions of libhdf5 that need to be linked to are 10.1.0 and 10.0.2 respectively:

```
find . -type f -exec sed -i -e 's^"hdf5.h"^"hdf5/serial/hdf5.h"^g' -e 's^"hdf5_hl.h"^"hdf5/serial/hdf5_hl.h"^g' '{}' \;
```

```
cd /usr/lib/x86_64-linux-gnu

sudo ln -s libhdf5_serial.so.10.1.0 libhdf5.so

sudo ln -s libhdf5_serial_hl.so.10.0.2 libhdf5_hl.so 
```

Now for both platforms lets return to the unpacked Caffe directory caffe-master and enter these commands:

```
cd python
```

```
for req in $(cat requirements.txt); do pip install $req; done
```
NOTE: If the Ubuntu operating system was updated, perhaps the Python layer needs to be updated and recompiled, because the Python module no longer works. Perform this step again in that case.

```
for req in $(cat requirements.txt); do pip install $req; done  
```
In case of any problems, try:

```
for req in $(cat requirements.txt); do sudo -H pip install $req --upgrade; done
```
The default Python version is 2. You can edit the Makefile.conf to enable the Python 3, but this will fail during the linking phase: boost_python3 cannot be found on Ubuntu 16.04. Instead, this file should be `/usr/lib/x86_64-linux-gnu/libboost_python-py35.so.1.58.0`. This requires further testing.

The next step is to build Caffe:

```
cd ..
```
(now you are in caffe-master directory)

The build process will fail in Ubuntu 16.04. Edit the Makefile with an editor such as

```
kate ./Makefile
```
and replace this line:

```
NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
```
with the following line

```
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
```
Also, open the file CMakeLists.txt and add the following line:

```
# ---[ Includes
set(${CMAKE_CXX_FLAGS} "-D_FORCE_INLINES ${CMAKE_CXX_FLAGS}")
```
(See the discussion at: https://github.com/BVLC/caffe/issues/4046)

When compiling with OpenCV 3.0 or errors show imread,imencode,imdecode or VideoCapture open your Makefile with some text editor, add opencv_imgcodecs behind.

```
 LIBRARIES += glog gflags protobuf leveldb snappy \
  lmdb boost_system boost_filesystem hdf5_hl hdf5 m \
  opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_videoio
```
(See the discussion at: https://github.com/BVLC/caffe/issues/1276)

Then

```
make all
make test
make runtest
make pycaffe      -should be finished already, so you can omit this one
make distribute
```
Note that the build process can be sped up by appending -j $(($(nproc) + 1)) to the above commands, which distributes the build across the available processors on your system. For example:

```
make all
```

can become

```
make all -j $(($(nproc) + 1))
```

In order to make the Python work with Caffe, open the file `~/.bashrc` for editing in your favorite text editor. There, add the following line at the end of file:

```
export PYTHONPATH=/path/to/caffe-master/python:$PYTHONPATH
```
You can also execute that same line immediately in the terminal as a command for immediate effects, or in general execute:

```
source ~/.bashrc 
```
In order to use the Caffe binaries, libraries, or include files, they need to be reachable through the search path, so one solution is to copy them into their respective directories: from the distribute directory to the `/usr/bin` or `/usr/lib or /usr/include`.

The binary models can be download with the following script. In caffe-master directory,

```
cd scripts
./download_model_binary.py ../models/bvlc_alexnet/
./download_model_binary.py ../models/bvlc_googlenet/
./download_model_binary.py ../models/bvlc_reference_caffenet/
./download_model_binary.py ../models/bvlc_reference_rcnn_ilsvrc13/
./download_model_binary.py ../models/finetune_flickr_style/
```
For more models, see https://github.com/BVLC/caffe/wiki/Model-Zoo *
For most Linux programs compiled from source, you can attempt to build a package that can be installed and uninstalled with a single click.

```
sudo apt-get install checkinstall
```
Now, when you execute the:

```
sudo checkinstall
```
and fill out a form with some easy questions, you will have the package made automatically. However, this uses the command "make install" in the background, which will fail, because the Caffe project does not have the target "install" configured in the Makefile.

The GPU support prerequisites

In Ubuntu desktop, enable the use of proprietary drivers in the Software & Updates Center for your desktop and install the NVIDIA graphics driver from the main Ubuntu package repository. See https://help.ubuntu.com/community/BinaryDriverHowto/Nvidia

Discover which driver number you need with:

```
sudo ubuntu-drivers devices
```
The LATEST version of Cuda Toolkit 8.0 is available from the NVIDIA website. Download the Cuda Toolkit 8.0 network installer and the CUDNN 5.1 package from the NVIDIA site, after registering and filling out the forms. https://developer.nvidia.com/cuda-downloads

Install the Cuda Toolkit 8.0 package manually in the terminal as instructed at the website. For example:

```
sudo apt-get update && sudo apt-get install wget -y --no-install-recommends
wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb"
sudo dpkg -i cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```
Download and install the archive that contains the CUDNN library from https://developer.nvidia.com/cudnn.

Put all the downloaded and unpacked CUDNN files manually starting with the search path directory where the CUDA toolkit is, each file in its own respective directory. That directory could be `/usr/local/cuda` or `/usr` if you installed Cuda 7.5 from the Ubuntu repository.

Example: the unpacked directory content should be copied from `/lib64` directory in the downloaded archive to `/usr/lib/x86_64-linux-gnu/` and from `/include` directory to `/usr/include/`.

```
CUDNN_URL="http://developer.download.nvidia.com/compute/redist/cudnn/v5.1/cudnn-8.0-linux-x64-v5.1.tgz"
wget ${CUDNN_URL}
sudo tar -xzf cudnn-8.0-linux-x64-v5.1.tgz -C /usr/local
rm cudnn-8.0-linux-x64-v5.1.tgz && sudo ldconfig
```
In Kubuntu 16.04, you also need to enable the use of proprietary drivers (in System Settings -> Driver Manager). The previous best option was to use the Muon package manager for all software installations. There, you can find the NVIDIA driver and the Cuda Toolkit 7.5 package. (Thus far, Cuda Toolkit 8.0 has not appeared in this package manager.) The packages are given through the standard Canonical and Canonical Partners software sources (repositories). You need to install the Muon manager itself with the following command:

```
sudo apt-get install muon
```
You can check your Ubuntu environment variables after the reboot, by executing the command:

```
export
```
Edit the Makefile.config in Caffe directory accordingly (as described in the config file itself) and recompile the Caffe to support the GPU computation. To recompile, first execute make clean.

Test the Caffe framework with DeepDream project

A fun way to try the Caffe framework for python is to use it in the DeepDream project. Visit the following site for specific source code: https://github.com/google/deepdream One of the trained neural networks used in this procedure, can be found here: http://places.csail.mit.edu/model/googlenet_places205.tar.gz Before you begin, you need to prepare the Ubuntu 16.04 system for the use of IPython and Jupyter. For these instructions, see: https://www.digitalocean.com/community/tutorials/how-to-set-up-a-jupyter-notebook-to-run-ipython-on-ubuntu-16-04

Then, you can add Python 2 and Python 3 kernels for the execution of IPython Notebooks (usually with the command ipython notebook filename).

```
python2 -m pip install ipykernel
python2 -m ipykernel install --user
```

```
python3 -m pip install ipykernel
python3 -m ipykernel install --user```
```
