# "Hello world" in Caffe2 (that is MNIST)

This repository is nothing more than a slightly modified MNIST tutorial (in python) which
you can run in order to produce network that can be loaded and used in C++ and a simple
program (using Qt for GUI) that attempts to do that.

## Install Caffe2
For me (Ubuntu) this is more or less following procedure:
  - clone caffe2 repo and cd into it
  - follow official installation tutorial or below steps (somehow when I follow official
    steps caffe is built twice first on make then on make install and besides by using this
    procedure I have better control over what is built in and what is not)
  - `mkdir build`
  - `cd build`
  - `cmake-gui .. $(python ../scripts/get_python_cmake_flags.py)`
  - in CMake gui specify "src" dir above build
  - repeat in loop until you are satisfied
    - "configure"
    - remove not used/needed components (e.g. CUDA - if you have problems with it)

  - set prefix to something (e.g. /usr/local/stow/caffe2 if you use stow)
  - generate & quit CMake gui
  - `make install`
  - if you use stow
    - `cd /usr/local/stow`
    - `sudo stow caffe2`

## Prepare code and train/test databases
- clone this repo
- download and unpack MNIST files (I've noticed that in newer version of MNIST
  tutorial they provide already built LMDB - if you decide to use it then update
  code accordingly since it still uses LevelDB)
- convert them to LevelDB using make_mnist_db binary from caffe2

## Train and run
- `python mnist.py`
- `cd QtMnist`
- `qmake QtMnist.pro`
- `make`
- `cd ..`
- `./QtMnistGUI`

Then on the left image you can scribble with your mouse, click "Convert" in order to downscale it
to 28x28 image which is input to the net.  This input is shown (upscaled) in the right image.
When you click "Test" the net will be fed with this image and the results will be shown in the form
of simple bar graph.
