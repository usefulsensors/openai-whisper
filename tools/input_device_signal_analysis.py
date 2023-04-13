"""
Input device signal analysis with plotting.

This blocks the audio device.
TODO(guy): make non-blocking version.

Uses pyaudio: https://pypi.org/project/PyAudio/ 
https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

from scipy import signal
from sys import platform


def get_input_device_id(p):
  """Returns user selection of available input devices, using PyAudio."""
  info = p.get_host_api_info_by_index(0)
  num_devices = info.get('deviceCount')
  print('Available input devices:')
  for i in range(num_devices):
    if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
      device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
      print(f'{i}: {device_name}')
  # Prompt user to enter input device ID
  device_id = input('Enter input device ID: ')
  return int(device_id)


def count_zero_crossings(audio):
  """Returns count of number of zero crossings found in audio array."""
  num_zero_crossings = np.count_nonzero(
    np.diff(np.sign(audio))
    )
  return num_zero_crossings


def peak(audio):
  """Returns peak value of the audio array, dBFS."""
  return 20.0 * np.log10(np.max(np.abs(audio)))


def root_mean_square(audio):
  """Returns root mean square of the audio array, dBFS."""
  return 20.0 * np.log10((np.mean(audio ** 2) ** 0.5))


def add_color_escape_code(text_string, color='black'):
  reset = '\x1b[0m'
  if color == 'black':
    return '\x1b[30m' + text_string + reset
  elif color == 'red':
    return '\x1b[31m' + text_string + reset
  elif color == 'green':
    return '\x1b[32m' + text_string + reset
  elif color == 'yellow':
    return '\x1b[33m' + text_string + reset
  elif color == 'blue':
    return '\x1b[34m' + text_string + reset
  elif color == 'magenta':
    return '\x1b[35m' + text_string + reset
  elif color == 'cyan':
    return '\x1b[36m' + text_string + reset
  elif color == 'white':
    return '\x1b[37m' + text_string + reset
  else:
    return reset + text_string + reset


def main():
  parser = argparse.ArgumentParser(
    description='Time-domain signal processing on microphone audio stream.')
  parser.add_argument('-b', '--buffer_time', type=float, default=1.0,
                      help='Signal buffer time in seconds, S.')
  parser.add_argument('-c', '--chunk_size', type=int, default=1600,
                      help='Audio stream chunk size.')
  parser.add_argument('--crest_factor_threshold', type=float, default=9.0,
                      help=('Threshold value for the buffer crest factor, '
                            'e.g.: 9.'))
  parser.add_argument('-d', '--device', type=int, default=None,
                      help='Input device ID, script will prompt.')
  parser.add_argument('--filter_highpass', default=True,
                      action=argparse.BooleanOptionalAction,
                      help='Remove low frequency from the audio.')
  parser.add_argument('-p', '--plot', default=False,
                      action=argparse.BooleanOptionalAction,
                      help=('Show plot on Mac (warning: plotting fails on '
                            'Ubuntu 22.04 LTS with Python 3.10.9).'))
  parser.add_argument('-r', '--rms_threshold', type=float, default=-50.,
                      help=('Threshold value for the buffer rms level in dB, '
                            'e.g.: -50'))
  parser.add_argument('-s', '--sampling_frequency', type=int, default=16000,
                      help=('Audio stream sampling frequency in hertz, Hz '
                            '(warning: values other than 48000 fail on '
                            'Ubuntu 22.04 LTS with Python 3.10.9).'))
  parser.add_argument('-z', '--zero_crossings', default=False,
                      action=argparse.BooleanOptionalAction,
                      help=('Include zero crossings when analyzing energy '
                            '(this can catch non-speech noises such as '
                            'high frequency "hissing", or steady low frequency '
                            'tones).'))
  args = parser.parse_args()

  assert args.rms_threshold <= 0.
  assert args.crest_factor_threshold > 0.

  # Parameters for audio capture
  FORMAT = pyaudio.paFloat32
  CHANNELS = 1

  if args.device is None:
    # Get input device ID from user if not provided as argument
    p = pyaudio.PyAudio()
    device_id = get_input_device_id(p)
    p.terminate()
  else:
    device_id = args.device

  if args.filter_highpass:
    # Design coefficients for a high pass filter.
    b, a  = signal.butter(4, 60, 'highpass', fs=args.sampling_frequency)

  # Create PyAudio object and select input device
  p = pyaudio.PyAudio()
  device_info = p.get_device_info_by_index(device_id)

  if args.plot:
    # Initialize plot figure
    plt.figure()
    plt.pause(0.001)

  # Open microphone stream
  stream = p.open(format=FORMAT, 
                  channels=CHANNELS, rate=args.sampling_frequency,
                  input=True, input_device_index=device_info['index'], 
                  frames_per_buffer=args.chunk_size)

  # Initialize buffer variables
  signal_buffer = np.zeros(int(args.sampling_frequency * args.buffer_time))
  plot_index = 0

  if platform == 'linux':
    if args.plot: 
      print('WARNING: plotting on Linux may cause crash, try --no-plot')
    if args.sampling_frequency != 48000:
      print(f'WARNING: sampling_frequency {args.sampling_frequency} on Linux '
             'may cause crash, try -s 48000')

  print(
    'Basic analysis method attempts to identify speech-like signals '
    '(blue color).\nPress Ctrl-c to quit...')

  # Loop to continuously capture and process audio
  try:
    while True:
      # Read audio data from microphone stream
      audio_data = stream.read(args.chunk_size)

      # Convert audio data to numpy array
      audio_buffer = np.frombuffer(audio_data, dtype=np.float32)

      # Add new chunk to the buffer.
      signal_buffer[:-args.chunk_size] = signal_buffer[args.chunk_size:]
      signal_buffer[-args.chunk_size:] = audio_buffer
      plot_index += args.chunk_size

      # Display text and plot.
      if plot_index >= int(args.sampling_frequency * args.buffer_time):

        # Filter the signal buffer to remove low frequencies.
        audio = signal_buffer
        if args.filter_highpass:
          audio = signal.lfilter(b, a, signal_buffer)

        # Simple analysis of signal.
        rms = root_mean_square(audio)
        pk = peak(audio)
        crest_factor = pk - rms
        zero_crossings = count_zero_crossings(audio)
        normalized_zero_crossings = int(zero_crossings / args.buffer_time)

        text_str = (f'RMS:{rms:>+2.1f}dB  '
                    f'Peak:{pk:>+2.1f}dB  '
                    f'C.F.:{crest_factor:>4.1f}  '
                    f'Z.C.:{normalized_zero_crossings:>5d}')
        
        # Apply heuristic analysis to find signals with energy.
        # These choices selected on 'MacBook Pro Microphone' with
        # Settings > Sound > Input level set at default (0.5).
        colour = 'black'
        if (rms > args.rms_threshold and 
            crest_factor > args.crest_factor_threshold and
            (not args.zero_crossings or 
             (normalized_zero_crossings > 400 and 
              normalized_zero_crossings < 5000))
            ):
          colour = 'blue'  
        
        if args.plot:
          plt.clf()
          plt.plot(audio, colour)
          plt.title(text_str)
          plt.ylim([-1., 1.])
          plt.grid()
          plt.pause(0.001)

        if colour == 'black':
          # Use the default color, might be white on black, or reverse. 
          print(text_str)
        else:
          print(add_color_escape_code(text_str, colour))

        plot_index = 0

  except KeyboardInterrupt:
    print('\nCtrl-C exception, cleaning up resources.')

  # Clean up resources
  stream.stop_stream()
  stream.close()
  p.terminate()

if __name__ == '__main__':
    main()
