For more information see the OpenAI whisper [paper](https://cdn.openai.com/papers/whisper.pdf).

# To run whisper inference on TFLite framework
TensorFlow Lite C++ minimal example to run inference on [whisper.tflite](https://github.com/usefulsensors/openai-whisper/blob/main/models/whisper.tflite)(~40 MB hybrid model weights are in int8 and activations are in float32)

This example shows how you can build a simple TensorFlow Lite application.

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

#### Step 3. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```

#### Step 4. Copy required files to tensorflow_src/tensorflow/lite/examples/minimal

```sh
cp tflite_minimal/minimal.cc  tensorflow_src/tensorflow/lite/examples/minimal/
cp tflite_minimal/*.h  tensorflow_src/tensorflow/lite/examples/minimal/
```

#### Step 5. Create CMake build directory and run CMake tool

```sh
mkdir minimal_build
cd minimal_build
cmake ../tensorflow_src/tensorflow/lite/examples/minimal
```

#### Step 6. Build TensorFlow Lite

In the minimal_build directory,

```sh
cmake --build . -j
```

#### Step 7. Copy ~/tflite_minimal/vocab_gen.bin to minimal_build

```sh
cp ../tflite_minimal/vocab_gen.bin ./
```

#### Step 8. Run the whisper.tflite with pre generated input_features or 16Khz 16bit Mono Audio file
```sh
./minimal ../models/whisper.tflite
./minimal ../models/whisper.tflite ../samples/jfk.wav
./minimal ../models/whisper.tflite ../samples/test.wav
./minimal ../models/whisper.tflite ../samples/test_1.wav
