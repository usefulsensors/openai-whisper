# Use the Whisper Hybrid TFLite Model to compile and run the standalone streaming example using the TFLite Framework.
TensorFlow Lite C++ streaming example to run inference using [whisper.tflite](https://github.com/usefulsensors/openai-whisper/blob/main/models/whisper.tflite)(~40 MB hybrid model weights are in int8 and activations are in float32)

This example shows how you can build a streaming TensorFlow Lite example using Whisper hybrid model.

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

#### Step 4. Build and run stream example 

```sh
mkdir stream_standalone
cd stream_standalone
cmake ./
cmake --build . -j
```

#### Step 5. Run the stream example with whisper.tflite model
```sh
./stream_standalone ../models/whisper.tflite
```

