For more information see the OpenAI whisper [paper](https://cdn.openai.com/papers/whisper.pdf).

To run inference on TFLite framework,build the TFLite C++ minimal example on Ubuntu Linux machine
# TensorFlow Lite C++ minimal example to run inference on whisper.tflite(~40 MB hybrid model weights are in int8 and activations are in float32)

This example shows how you can build a simple TensorFlow Lite application.

#### Step 1. Install CMake tool

It requires CMake 3.16 or higher. On Ubuntu, you can simply run the following
command.

```sh
sudo apt-get install cmake
```

Or you can follow
[the official cmake installation guide](https://cmake.org/install/)

#### Step 2. Clone TensorFlow repository

```sh
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```
#### Step 3. Copy minimal.cc and input_features.h from [here](https://github.com/usefulsensors/openai-whisper) to tensorflow_src/tensorflow/lite/examples/

#### Step 4. Create CMake build directory and run CMake tool

```sh
mkdir minimal_build
cd minimal_build
cmake ../tensorflow_src/tensorflow/lite/examples/minimal
```

#### Step 5. Build TensorFlow Lite

In the minimal_build directory,

```sh
cmake --build . -j
```

#### Step 6. Run the whisper.tflite 
```sh
./minimal ~/openai-whisper/models/whisper.tflite ~/openai-whisper/test.wav
```

## Convert openai-whisper ASR from pytorch to tflite(int8) model
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/usefulsensors/openai-whisper/blob/main/tinynn_pytorch_to_tflite_int8.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Google Colaboratory</a>
  </td>
</table>
*Estimated Run Time: ~1 Mins.*

##

## Convert Huggingface-openai-whisper ASR tf saved to tflite model
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/usefulsensors/openai-whisper/blob/main/tflite_from_huggingface_whisper.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Google Colaboratory</a>
  </td>
</table>
*Estimated Run Time: ~5 Mins.*

##

## Run openai-whisper ASR model to generate closed captions for youtube videos
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/usefulsensors/openai-whisper/blob/main/youtube_to_subtitles.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Google Colaboratory</a>
  </td>
</table>
*Estimated Run Time: ~1 Mins.*

##

## Run  openai-whisper ASR model
<table class="tfo-notebook-buttons" align="left">
  <td>
    <a target="_blank" href="https://colab.research.google.com/github/usefulsensors/openai-whisper/blob/main/openai_whisper_ASR.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Google Colaboratory</a>
  </td>
</table>
*Estimated Run Time: ~1 Mins.*

##  
