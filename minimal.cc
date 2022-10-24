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
#include <cstdio>
#include <vector>
#include <iostream>
#include "tensorflow/lite/core/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "./input_features.h"
#include "./input_features_filters.h"
#include <cmath>
#include<iostream>
#include <sys/time.h>

#include <thread>
#include <vector>

//Added audio front end processing from https://github.com/ggerganov/whisper.cpp
// third-party utilities
// use your favorite implementations
#define DR_WAV_IMPLEMENTATION
#include "./dr_wav.h"
#include <fstream>
#include <cstdio>
#include <string>


#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_N_MEL       80
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

struct whisper_filters {
    int32_t n_mel;
    int32_t n_fft;

    std::vector<float> data;
};

struct whisper_mel {
    int n_len;
    int n_mel;

    std::vector<float> data;
};

void print(std::vector <float> const &a) {
   std::cout << "The vector elements are : ";

   for(int i=0; i < a.size(); i++)
   std::cout << a.at(i) << ' ';
}

// naive Discrete Fourier Transform
// input is real-valued
// output is complex-valued
void dft(const std::vector<float> & in, std::vector<float> & out) {
    int N = in.size();

    out.resize(N*2);

    for (int k = 0; k < N; k++) {
        float re = 0;
        float im = 0;

        for (int n = 0; n < N; n++) {
            float angle = 2*M_PI*k*n/N;
            re += in[n]*cos(angle);
            im -= in[n]*sin(angle);
        }

        out[k*2 + 0] = re;
        out[k*2 + 1] = im;
    }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
void fft(const std::vector<float> & in, std::vector<float> & out) {
    out.resize(in.size()*2);

    int N = in.size();

    if (N == 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }

    if (N%2 == 1) {
        dft(in, out);
        return;
    }

    std::vector<float> even;
    std::vector<float> odd;

    for (int i = 0; i < N; i++) {
        if (i % 2 == 0) {
            even.push_back(in[i]);
        } else {
            odd.push_back(in[i]);
        }
    }

    std::vector<float> even_fft;
    std::vector<float> odd_fft;

    fft(even, even_fft);
    fft(odd, odd_fft);

    for (int k = 0; k < N/2; k++) {
        float theta = 2*M_PI*k/N;

        float re = cos(theta);
        float im = -sin(theta);

        float re_odd = odd_fft[2*k + 0];
        float im_odd = odd_fft[2*k + 1];

        out[2*k + 0] = even_fft[2*k + 0] + re*re_odd - im*im_odd;
        out[2*k + 1] = even_fft[2*k + 1] + re*im_odd + im*re_odd;

        out[2*(k + N/2) + 0] = even_fft[2*k + 0] - re*re_odd + im*im_odd;
        out[2*(k + N/2) + 1] = even_fft[2*k + 1] - re*im_odd - im*re_odd;
    }
}

// ref: https://github.com/openai/whisper/blob/main/whisper/audio.py#L92-L124
bool log_mel_spectrogram(
    const float * samples,
    const int n_samples,
    const int sample_rate,
    const int fft_size,
    const int fft_step,
    const int n_mel,
    const int n_threads,
    const whisper_filters & filters,
    whisper_mel & mel) {

    // Hanning window
    std::vector<float> hann;
    hann.resize(fft_size);
    for (int i = 0; i < fft_size; i++) {
        hann[i] = 0.5*(1.0 - cos((2.0*M_PI*i)/(fft_size)));
    }

    mel.n_mel = n_mel;
    mel.n_len = (n_samples)/fft_step;
    mel.data.resize(mel.n_mel*mel.n_len);

    const int n_fft = 1 + fft_size/2;

    //printf("%s: n_samples = %d, n_len = %d\n", __func__, n_samples, mel.n_len);
    //printf("%s: recording length: %f s\n", __func__, (float) n_samples/sample_rate);

    std::vector<std::thread> workers(n_threads);
    for (int iw = 0; iw < n_threads; ++iw) {
        workers[iw] = std::thread([&](int ith) {
            std::vector<float> fft_in;
            fft_in.resize(fft_size);
            for (int i = 0; i < fft_size; i++) {
                fft_in[i] = 0.0;
            }

            std::vector<float> fft_out;
            fft_out.resize(2*fft_size);

            for (int i = ith; i < mel.n_len; i += n_threads) {
                const int offset = i*fft_step;

                // apply Hanning window
                for (int j = 0; j < fft_size; j++) {
                    if (offset + j < n_samples) {
                        fft_in[j] = hann[j]*samples[offset + j];
                    } else {
                        fft_in[j] = 0.0;
                    }
                }

                // FFT -> mag^2
                fft(fft_in, fft_out);

                for (int j = 0; j < fft_size; j++) {
                    fft_out[j] = (fft_out[2*j + 0]*fft_out[2*j + 0] + fft_out[2*j + 1]*fft_out[2*j + 1]);
                }
                for (int j = 1; j < fft_size/2; j++) {
                    //if (i == 0) {
                    //    printf("%d: %f %f\n", j, fft_out[j], fft_out[fft_size - j]);
                    //}
                    fft_out[j] += fft_out[fft_size - j];
                }
                if (i == 0) {
                    //for (int j = 0; j < fft_size; j++) {
                    //    printf("%d: %e\n", j, fft_out[j]);
                    //}
                }

                // mel spectrogram
                for (int j = 0; j < mel.n_mel; j++) {
                    double sum = 0.0;

                    for (int k = 0; k < n_fft; k++) {
                        sum += fft_out[k]*filters.data[j*n_fft + k];
                    }
                    if (sum < 1e-10) {
                        sum = 1e-10;
                    }

                    sum = log10(sum);

                    mel.data[j*mel.n_len + i] = sum;
                }
            }
        }, iw);
    }

    for (int iw = 0; iw < n_threads; ++iw) {
        workers[iw].join();
    }

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] > mmax) {
            mmax = mel.data[i];
        }
    }
    //printf("%s: max = %f\n", __func__, mmax);

    mmax -= 8.0;

    for (int i = 0; i < mel.n_mel*mel.n_len; i++) {
        if (mel.data[i] < mmax) {
            mel.data[i] = mmax;
        }

        mel.data[i] = (mel.data[i] + 4.0)/4.0;
    }

    return true;
}

// This is an example that is minimal to read a model
// from disk and perform inference. There is no data being loaded
// that is up to you to add as a user.
//
// NOTE: Do not add any dependencies to this that cannot be built with
// the minimal makefile. This example must remain trivial to build with
// the minimal build tool.
//
// Usage: minimal <tflite model>
int golden_generated_ids[21] = {50257,50362,1770,13,2264,346,353,318,262,46329,286,262,3504,6097,11,290,356,389,9675,284,7062};

#define TFLITE_MINIMAL_CHECK(x)                              \
  if (!(x)) {                                                \
    fprintf(stderr, "Error at %s:%d\n", __FILE__, __LINE__); \
    exit(1);                                                 \
  }

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "minimal <tflite model> <pcm_file name>\n");
    return 1;
  }
  const char* filename = argv[1];
  const char* pcmfilename = argv[2];
  // WAV input
  std::vector<float> pcmf32;
  {
      drwav wav;
      if (!drwav_init_file(&wav, pcmfilename, NULL)) {
          fprintf(stderr, "%s: failed to open WAV file '%s' - check your input\n", argv[0],pcmfilename);
        //  whisper_print_usage(argc, argv, {});
          return 3;
      }

      if (wav.channels != 1 && wav.channels != 2) {
          fprintf(stderr, "%s: WAV file '%s' must be mono or stereo\n", argv[0], pcmfilename);
          return 4;
      }

      if (wav.sampleRate != WHISPER_SAMPLE_RATE) {
          fprintf(stderr, "%s: WAV file '%s' must be 16 kHz\n", argv[0], pcmfilename);
          return 5;
      }

      if (wav.bitsPerSample != 16) {
          fprintf(stderr, "%s: WAV file '%s' must be 16-bit\n", argv[0], pcmfilename);
          return 6;
      }

      int n = wav.totalPCMFrameCount;

      std::vector<int16_t> pcm16;
      pcm16.resize(n*wav.channels);
      drwav_read_pcm_frames_s16(&wav, n, pcm16.data());
      drwav_uninit(&wav);

      // convert to mono, float
      pcmf32.resize(n);
      if (wav.channels == 1) {
          for (int i = 0; i < n; i++) {
              pcmf32[i] = float(pcm16[i])/32768.0f;
          }
      } else {
          for (int i = 0; i < n; i++) {
              pcmf32[i] = float(pcm16[2*i] + pcm16[2*i + 1])/65536.0f;
          }
      }
  }
  struct timeval start_time,end_time;
  gettimeofday(&start_time, NULL);
  //   Loaded - pre-computed mel filters from https://github.com/ggerganov/whisper.cpp
  whisper_filters filters;
  whisper_mel mel;
  filters.n_mel = 80;
  filters.n_mel = 201;
  filters.data.resize(filters.n_mel * filters.n_fft);
  memcpy((char *)filters.data.data(),input_features_filters_bin,16080*sizeof(float));



  if (!log_mel_spectrogram(pcmf32.data(), pcmf32.size(), WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, WHISPER_N_MEL, 1,filters, mel)) {
      fprintf(stderr, "%s: failed to compute mel spectrogram\n", __func__);
      return -1;
  }

  gettimeofday(&end_time, NULL);
  printf("\nAudio front end processing time %ld micro seconds \n",(end_time.tv_usec-start_time.tv_usec));
  printf("\nmel.n_len%d\n",mel.n_len);
  printf("\nmel.n_mel:%d\n",mel.n_mel);
  //print(mel.data);
  // Load model
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(filename);
  TFLITE_MINIMAL_CHECK(model != nullptr);

  // Build the interpreter with the InterpreterBuilder.
  // Note: all Interpreters should be built with the InterpreterBuilder,
  // which allocates memory for the Interpreter and does various set up
  // tasks so that the Interpreter can read the provided model.
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model, resolver);
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  //printf("=== Pre-invoke Interpreter State ===\n");
  // tflite::PrintInterpreterState(interpreter.get());
  // Get information about the memory area to use for the model's input.
  float* input = interpreter->typed_input_tensor<float>(0);
//  memcpy(input,_content_input_features_bin,80*3000*sizeof(float)); //to load pre generated input_features
  memcpy(input,mel.data.data(),mel.n_mel*mel.n_len*sizeof(float));

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  gettimeofday(&start_time, NULL);

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  gettimeofday(&end_time, NULL);
  printf("Inference time %ld seconds \n",(end_time.tv_sec-start_time.tv_sec));
  int output = interpreter->outputs()[0];
  TfLiteTensor *output_tensor = interpreter->tensor(output);
  TfLiteIntArray *output_dims = output_tensor->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  printf("output size:%d\n",output_size );
  //FILE *fp = fopen("generated_ids.bin", "w");
  int *output_int = interpreter->typed_output_tensor<int>(0);
  for (int i = 0; i < output_size; i++) {
//    if(golden_generated_ids[i]!=output_int[i]) {
  //    printf("\nMismatch with generated ids of golden output\n");
//    }
    printf("%d\t",output_int[i]);
  //  fwrite(output_int[i],sizeof(int),1,fp);
  }
  printf("\n");
  //fclose(fp);

  //printf("\n\n=== Post-invoke Interpreter State ===\n");
  ////  tflite::PrintInterpreterState(interpreter.get());
  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
  return 0;
}
