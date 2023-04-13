For more information see the OpenAI whisper [paper](https://cdn.openai.com/papers/whisper.pdf).

Supported platforms:

- [x] [Linux](#run-whisper-inference-on-tflite-framework) 
- [x] Mac OS (Intel)
- [x] [Android OS](#android-os)
- [x] [Apple iOS](#apple-ios) 


# Run whisper inference on TFLite framework
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
# build the minimal example
cmake --build . -j
```

If `cmake` build fails try specifying number of cores with -j flag,
```sh
cmake --build . -j 8
```

#### Step 7. Copy ~/tflite_minimal/filters_vocab_gen.bin to minimal_build

to run whisper.tflite
```sh
cp ../tflite_minimal/filters_vocab_gen.bin ./
```

#### Step 8. Run the whisper.tflite with pre generated input_features or 16Khz 16bit Mono Audio file
```sh
./minimal ../models/whisper.tflite
# transcribe an audio file
./minimal ../models/whisper.tflite ../samples/jfk.wav
./minimal ../models/whisper.tflite ../samples/test.wav
./minimal ../models/whisper.tflite ../samples/test_1.wav
```

to run whisper-small.tflite
```sh
cp ../models/filters_vocab_multilingual.bin ./filters_vocab_gen.bin
./minimal ../models/whisper-small.tflite ../samples/jfk.wav
```

to run whisper-medium.tflite
```sh
cp ../models/filters_vocab_multilingual.bin ./filters_vocab_gen.bin
./minimal ../models/whisper-medium.tflite ../samples/jfk.wav
```

Note: Use the arecord application to record test audio on a Linux computer.
```sh
arecord -r 16000 -c 1 -d 30 -f S16_LE test.wav

```

# Android OS
Feel free to download the openai/whisper-tiny tflite-based Android Whisper ASR APP from [Google App Store](https://play.google.com/store/apps/details?id=com.whisper.android.tflitecpp).

# Apple iOS
Feel free to download the openai/whisper-tiny tflite-based Apple Whisper ASR APP from [Apple App Store](https://apps.apple.com/in/app/whisper-asr/id6444556326).
