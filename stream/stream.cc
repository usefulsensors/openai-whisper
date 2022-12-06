/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "whisper.h"
#include <SDL.h>
#include <SDL_audio.h>
#include "filters_vocab_gen.h"
using namespace std;
//#include "whisper_hybrid_tflite_model.h"

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>


#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

whisper_filters filters;
whisper_mel mel;

string convert_to_string(char* array, int size)
{
    int i;
    string str = "";
    for (i = 0; i < size; i++) {
        str = str + array[i];
    }
    return str;
}

//
//Load Mel and Vocabs from pre generated binary
//
int load_mel_vocab_from_pre_gen_array(void)
{
  int32_t n_vocab = 0;
  int index=0;
  std::string word;
  unsigned char *temp = filters_vocab_gen_bin;
  {
    uint32_t magic =  *(uint32_t*)temp;
    index = index + sizeof(uint32_t);
    //@magic:USEN
    if (magic != 0x5553454e) {
        printf("%s: invalid vocab file '%s' (bad magic)\n", __func__, "filters_vocab_gen_bin");
        return -1;
    }
  }

  // load mel filters
  {
      filters.n_mel = *(int32_t*)(temp+index);
      index = index+sizeof(int32_t);
      filters.n_fft = *(int32_t*)(temp+index);
      index = index+sizeof(int32_t);
      filters.data.resize(filters.n_mel * filters.n_fft);

      for (auto it =   filters.data.begin(); it !=   filters.data.end(); it++)
      {
          *it = *(float*)(temp+index);
          index = index+sizeof(float);
      }

  }

  // load vocab
  {
    n_vocab = *(int*)(temp+index);
    index = index+sizeof(n_vocab);
    g_vocab.n_vocab = n_vocab;
    printf("\nn_vocab:%d\n",(int)n_vocab);
    char vocab_word[1000];//
    for (int i = 0; i < n_vocab; i++) {
      uint32_t len = *(uint32_t*)(temp+index);
      index = index+sizeof(len);

      word.resize(len);

      for (int j =  0; j < len; j++)
      {
          vocab_word[j] = *(char*)(temp+index);
          index = index+sizeof(char);
      }
      word = convert_to_string(vocab_word, len);
      g_vocab.id_to_token[i] = word;
    //  printf("len:%d",(int)len);
    //  printf("'%s'\n", g_vocab.id_to_token[i].c_str());
    }

    g_vocab.n_vocab = 51864;//add additional vocab ids
    if (g_vocab.is_multilingual()) {
        g_vocab.token_eot++;
        g_vocab.token_sot++;
        g_vocab.token_prev++;
        g_vocab.token_solm++;
        g_vocab.token_not++;
        g_vocab.token_beg++;
    }
    for (int i = n_vocab; i < g_vocab.n_vocab; i++) {
        if (i > g_vocab.token_beg) {
            word = "[_TT_" + std::to_string(i - g_vocab.token_beg) + "]";
        } else if (i == g_vocab.token_eot) {
            word = "[_EOT_]";
        } else if (i == g_vocab.token_sot) {
            word = "[_SOT_]";
        } else if (i == g_vocab.token_prev) {
            word = "[_PREV_]";
        } else if (i == g_vocab.token_not) {
            word = "[_NOT_]";
        } else if (i == g_vocab.token_beg) {
            word = "[_BEG_]";
        } else {
            word = "[_extra_token_" + std::to_string(i) + "]";
        }
        g_vocab.id_to_token[i] = word;
        // printf("%s: g_vocab[%d] = '%s'\n", __func__, i, word.c_str());
    }
  }

  return 0;
}

//
// SDL Audio capture
//

SDL_AudioDeviceID g_dev_id_in = 0;

bool audio_sdl_init(const int capture_id) {
    if (g_dev_id_in) {
        fprintf(stderr, "%s: already initialized\n", __func__);
        return false;
    }

    SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s\n", SDL_GetError());
        return (1);
    }

    SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);

    {
        int nDevices = SDL_GetNumAudioDevices(SDL_TRUE);
        fprintf(stderr, "%s: found %d capture devices:\n", __func__, nDevices);
        for (int i = 0; i < nDevices; i++) {
            fprintf(stderr, "%s:    - Capture device #%d: '%s'\n", __func__, i, SDL_GetAudioDeviceName(i, SDL_TRUE));
        }
    }

    SDL_AudioSpec capture_spec_requested;
    SDL_AudioSpec capture_spec_obtained;

    SDL_zero(capture_spec_requested);
    SDL_zero(capture_spec_obtained);

    capture_spec_requested.freq     = WHISPER_SAMPLE_RATE;
    capture_spec_requested.format   = AUDIO_F32;
    capture_spec_requested.channels = 1;
    capture_spec_requested.samples  = 1024;

    if (capture_id >= 0) {
        fprintf(stderr, "%s: attempt to open capture device %d : '%s' ...\n", __func__, capture_id, SDL_GetAudioDeviceName(capture_id, SDL_TRUE));
        g_dev_id_in = SDL_OpenAudioDevice(SDL_GetAudioDeviceName(capture_id, SDL_TRUE), SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
    } else {
        fprintf(stderr, "%s: attempt to open default capture device ...\n", __func__);
        g_dev_id_in = SDL_OpenAudioDevice(nullptr, SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
    }
    if (!g_dev_id_in) {
        fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n", __func__, SDL_GetError());
        g_dev_id_in = 0;
    } else {
        fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n", __func__, g_dev_id_in);
        fprintf(stderr, "%s:     - sample rate:       %d\n", __func__, capture_spec_obtained.freq);
        fprintf(stderr, "%s:     - format:            %d (required: %d)\n", __func__, capture_spec_obtained.format, capture_spec_requested.format);
        fprintf(stderr, "%s:     - channels:          %d (required: %d)\n", __func__, capture_spec_obtained.channels, capture_spec_requested.channels);
        fprintf(stderr, "%s:     - samples per frame: %d\n", __func__, capture_spec_obtained.samples);
    }

    return true;
}


//
//Initialize interpreter
//
int g_tflite_intrepreter_init = 0;
std::unique_ptr<tflite::Interpreter> interpreter;
std::unique_ptr<tflite::FlatBufferModel> model;

bool tf_lite_interepreter_init(char* filename) {
    if (g_tflite_intrepreter_init) {
        fprintf(stderr, "%s: already initialized\n", __func__);
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    // Load tflite model
    model = tflite::FlatBufferModel::BuildFromFile(filename);

    TFLITE_MINIMAL_CHECK(model != nullptr);

    // Build the interpreter with the InterpreterBuilder.
    // Note: all Interpreters should be built with the InterpreterBuilder,
    // which allocates memory for the Interpreter and does various set up
    // tasks so that the Interpreter can read the provided model.

    tflite::InterpreterBuilder builder(*model, resolver);

    builder(&interpreter);
    TFLITE_MINIMAL_CHECK(interpreter != nullptr);

    // Allocate tensor buffers.
    TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);

    g_tflite_intrepreter_init = 1;

    return true;

}


///////////////////////////
int main(int argc, char* argv[]) {
  char* filename = argv[1];

  struct timeval start_time,end_time;

  const int n_samples = (WHISPER_STEP_MS/1000.0)*WHISPER_SAMPLE_RATE;
  const int n_samples_len = (WHISPER_LENGTH_MS/1000.0)*WHISPER_SAMPLE_RATE;
  const int n_samples_30s = 30*WHISPER_SAMPLE_RATE;
  const int n_samples_keep = 0.2*WHISPER_SAMPLE_RATE;
  std::vector<float> pcmf32(n_samples_30s, 0.0f);
  std::vector<float> pcmf32_old;
  const int n_new_line = WHISPER_LENGTH_MS / WHISPER_STEP_MS - 1;
  bool is_running = true;
  int n_iter = 0;


  //Load Mel and Vocab from pre gen binary
  load_mel_vocab_from_pre_gen_array();

  // init audio
  if (!audio_sdl_init(1)) {
      fprintf(stderr, "%s: audio_sdl_init() failed!\n", __func__);
      return 1;
  }
  SDL_PauseAudioDevice(g_dev_id_in, 0);

  // main audio loop
  while (is_running) {
      // handle Ctrl + C
      {
          SDL_Event event;
          while (SDL_PollEvent(&event)) {
              switch (event.type) {
                  case SDL_QUIT:
                      {
                          is_running = false;
                      } break;
                  default:
                      break;
              }
          }

          if (!is_running) {
              break;
          }
      } //end of CTRl+C

      if (!is_running) {
        break;
      }

      // process new audio
      if (n_iter > 0 && SDL_GetQueuedAudioSize(g_dev_id_in) > 2*n_samples*sizeof(float)) {
          fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
          SDL_ClearQueuedAudio(g_dev_id_in);
      }

      while (SDL_GetQueuedAudioSize(g_dev_id_in) < n_samples*sizeof(float)) {
          SDL_Delay(1);
      }

      const int n_samples_new = SDL_GetQueuedAudioSize(g_dev_id_in)/sizeof(float);

      // take one second from previous iteration
      //const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_30s/30 - n_samples_new));

      // take up to params.length_ms audio from previous iteration
      const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

      //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

      pcmf32.resize(n_samples_new + n_samples_take);

      for (int i = 0; i < n_samples_take; i++) {
          pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
      }

      SDL_DequeueAudio(g_dev_id_in, pcmf32.data() + n_samples_take, n_samples_new*sizeof(float));

      pcmf32_old = pcmf32;
      //Generate spectrograms
      pcmf32.resize((WHISPER_SAMPLE_RATE*WHISPER_CHUNK_SIZE),0);
      const auto processor_count = std::thread::hardware_concurrency();
      if (!log_mel_spectrogram(pcmf32.data(), pcmf32.size(), WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL, processor_count,filters, mel)) {
        fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
        return -1;
      }
      if(!g_tflite_intrepreter_init)
        tf_lite_interepreter_init(filename);

      // Get information about the memory area to use for the model's input.
      float* input = interpreter->typed_input_tensor<float>(0);

      memcpy(input, mel.data.data(), mel.n_mel*mel.n_len*sizeof(float));

      // Run inference
      gettimeofday(&start_time, NULL);
      TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
      gettimeofday(&end_time, NULL);

    //  printf("Inference time %ld seconds \n",(end_time.tv_sec-start_time.tv_sec));

      int output = interpreter->outputs()[0];
      TfLiteTensor *output_tensor = interpreter->tensor(output);
      TfLiteIntArray *output_dims = output_tensor->dims;
      // assume output dims to be something like (1, 1, ... ,size)
      auto output_size = output_dims->data[output_dims->size - 1];
    //  printf("\noutput_size:%d",output_size);
      //printf("output size:%d\n",output_size );
      int *output_int = interpreter->typed_output_tensor<int>(0);
      std::string text = "";

      // print result;
      for (int i = 0; i < output_size; i++) {
        //printf("%d\t",output_int[i]);
        if(output_int[i] == g_vocab.token_eot){
          break;
        }
        if((output_int[i] !=50257) && (output_int[i] !=50362))
            text += whisper_token_to_str(output_int[i]);
      }
      printf("%s", text.c_str());
      fflush(stdout);
      ++n_iter;

      if ((n_iter % n_new_line) == 0) {
        printf("\n");
        // keep part of the audio for next iteration to try to mitigate word boundary issues
        pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());
      }
    } //end of while

    if (g_dev_id_in >= 0) {
      SDL_CloseAudioDevice(g_dev_id_in);
    }
  return 0;
}
