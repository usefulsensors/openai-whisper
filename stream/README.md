# Run stream using whisper model on TFLite framework
TensorFlow Lite C++ stream example to run inference using [whisper.tflite](https://github.com/usefulsensors/openai-whisper/blob/main/models/whisper.tflite)(~40 MB hybrid model weights are in int8 and activations are in float32)

This example shows how you can build a stream TensorFlow Lite application using Whisper hybrid model.

#### Step 1. Clone Usefulsensors/openai-whisper repository

It requires 'git lfs install' as our [whisper.tflite](https://github.com/usefulsensors/openai-whisper/blob/main/models/whisper.tflite) model uses Git Large File Storage (LFS).

you can follow
[git lfs installation guide](https://git-lfs.github.com/)

```sh
git clone https://github.com/usefulsensors/openai-whisper.git
cd openai-whisper
```
#### Step 2. Install CMake tool

It requires CMake 3.16 or higher. On Ubuntu, you can simply run the following
command.

```sh
sudo apt-get install cmake
```
Or you can follow
[the official cmake installation guide](https://cmake.org/install/)

#### Step 3.Stream has dependency on SDL2 lib to capture audio

```sh
# Install SDL2 on Linux
sudo apt-get install libsdl2-dev
```

#### Step 4. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

#### Step 5. Copy required files to tensorflow_src/tensorflow/lite/examples/stream

```sh
cp -fr stream tensorflow_src/tensorflow/lite/examples/
```

#### Step 6. Create CMake build directory and run CMake tool

```sh
mkdir stream_build
cd stream_build
cmake ../tensorflow_src/tensorflow/lite/examples/stream
```

#### Step 7. Build TensorFlow Lite

In the stream directory,# build the stream example

```sh
cmake --build . -j
```

If the `cmake` build fails try using `make` instead,

```sh
# build the minimal example
make
```

#### Step 8. Run the stream application with whisper.tflite model
```sh
./stream ../models/whisper.tflite
```

