"""
Input device signal analysis with plotting.

TODO(guy): make non-blocking version.

Uses pyaudio: https://pypi.org/project/PyAudio/
See: https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio

Example command:
python3 ./input_device_signal_analysis.py
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
  device_id = None
  info = p.get_host_api_info_by_index(0)
  num_devices = info.get('deviceCount')
  print('Available input devices:')
  for i in range(num_devices):
    if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
      device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
      if platform == 'linux' and device_name == 'default':
        print(f'{i}: {device_name}  (recommended)')
        device_id = i
      else:
        print(f'{i:2d}: {device_name}')
  if not device_id:
    # Prompt user to enter input device ID
    device_id = input('Enter input device ID: ')
    return int(device_id)
  else:
    return device_id


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
    description=('Input device time-domain signal analysis: '
                 'blue color indicates possible speech; '
                 'red color indicates clipping saturation.'))
  parser.add_argument('-b', '--buffer_time', type=float, default=1.0,
                      help=('Signal buffer time (buffer is filled with chunks)'
                            ', seconds.'))
  parser.add_argument('-c', '--chunk_size', type=int, default=1600,
                      help='Audio stream chunk size.')
  parser.add_argument('-d', '--device', type=int, default=None,
                      help='Input device ID.')
  parser.add_argument('--filter_highpass', default=True,
                      action=argparse.BooleanOptionalAction,
                      help='Remove low frequency from the audio.')
  parser.add_argument('-g', '--gain_software', type=float, default=None,
                      help=('Gain applied in software to audio (it is not '
                            'applied to red color clipping saturation '
                            'detection), dB. i.e.: -g 20'))
  parser.add_argument('-p', '--plot', default=True,
                      action=argparse.BooleanOptionalAction,
                      help=('Show plot (recommended Linux input device is '
                            '"default" to mitigate stream crash).'))
  parser.add_argument('-r', '--rms_threshold', type=float, default=-50.,
                      help=('Threshold value for the buffer rms level, dB. '
                            'i.e.: -r -50'))
  parser.add_argument('-s', '--sampling_frequency', type=int, default=16000,
                      help=('Audio stream sampling frequency in hertz, Hz.'))
  parser.add_argument('--crest_factor_thresholds', type=float,
                      default=[9., 23.], nargs='+',
                      help=('Threshold value bounds for the crest factor '
                            'check.  Within these two values the signal may be '
                            'speech, i.e.: --crest_factor_thresholds 9. 23.'))
  parser.add_argument('--zero_crossings', default=True,
                      action=argparse.BooleanOptionalAction,
                      help=('Include zero crossings when analyzing energy '
                            '(this can catch non-speech noises such as '
                            'high frequency "hissing", or steady low frequency '
                            'tones).'))
  parser.add_argument('--zero_crossing_thresholds', type=float,
                      default=[400., 6000.], nargs='+',
                      help=('Threshold value bounds for the zero crossings per '
                            'second check applied when -z is True. Within '
                            'these two values the signal may be speech. '
                            'i.e.: --zero_crossing_thresholds 400. 6000.'))
  args = parser.parse_args()

  # TODO(guy): add more assertion tests on arguments.
  assert args.rms_threshold <= 0.
  for thresholds in [args.crest_factor_thresholds,
                     args.zero_crossing_thresholds]:
    assert len(thresholds) == 2
    for item in thresholds:
      assert item > 0.
    assert (thresholds[1] - thresholds[0]) > 0.

  # Parameters for input device audio capture.
  FORMAT = pyaudio.paFloat32
  CHANNELS = 1

  if args.device is None:
    # Get input device ID from user if not provided as argument.
    p = pyaudio.PyAudio()
    device_id = get_input_device_id(p)
    p.terminate()
  else:
    device_id = args.device

  if args.filter_highpass:
    # Design coefficients for a high pass filter.
    b, a  = signal.butter(4, 60, 'highpass', fs=args.sampling_frequency)

  # Create PyAudio object and select input device.
  p = pyaudio.PyAudio()
  device_info = p.get_device_info_by_index(device_id)

  # Check format is supported.
  assert p.is_format_supported(
    input_device=device_id, input_format=FORMAT,
    input_channels=CHANNELS, rate=args.sampling_frequency,
  )

  if args.plot:
    # Initialize plot figure.
    plt.figure()
    plt.pause(0.001)

  # Open input device stream.
  stream = p.open(format=FORMAT, 
                  channels=CHANNELS, rate=args.sampling_frequency,
                  input=True, input_device_index=device_info['index'], 
                  frames_per_buffer=args.chunk_size)

  # Initialize buffer variables.
  signal_buffer = np.zeros(int(args.sampling_frequency * args.buffer_time))
  plot_index = 0

  if platform == 'linux' and device_info['name'] != 'default': 
    print('\nWhen not using "default" Linux input device:')
    if args.plot: 
      print('  WARNING: plotting may cause crash, try --no-plot')
    if args.sampling_frequency != 48000:
      print(f'  WARNING: sampling_frequency {args.sampling_frequency} may '
             'cause crash, try -s 48000')

  print(
"""
Time domain analysis:
  Stream clipping saturation (red color)
  Possible speech signals (blue color)
Press Ctrl-C to quit.
""")

  # Loop to continuously capture and process audio.
  try:
    while True:
      # Read audio data from stream.
      audio_data = stream.read(args.chunk_size)

      # Convert audio data to numpy array.
      audio_buffer = np.frombuffer(audio_data, dtype=np.float32)

      # Add new chunk to the buffer.
      signal_buffer[:-args.chunk_size] = signal_buffer[args.chunk_size:]
      signal_buffer[-args.chunk_size:] = audio_buffer
      plot_index += args.chunk_size

      # Display text and plot.
      if plot_index >= int(args.sampling_frequency * args.buffer_time):
        peak_signal = peak(signal_buffer)

        # Filter the signal buffer to remove low frequencies.
        audio = signal_buffer        
        if args.filter_highpass:
          audio = signal.lfilter(b, a, audio)

        # Apply software gain to adjust signal level.
        if args.gain_software:
          # TODO(guy): check INT16.
          audio = audio * (10. ** (args.gain_software / 20.0))

        # Simple analysis of processed signal.
        rms = root_mean_square(audio)
        pk = peak(audio)
        crest_factor = pk - rms
        zero_crossings_per_second = int(
          count_zero_crossings(audio) / args.buffer_time)

        text_str = (f'RMS:{rms:>+5.1f}dB  '
                    f'Peak:{pk:>+5.1f}dB  '
                    f'C.F.:{crest_factor:>4.1f}  '
                    f'Z.C.:{zero_crossings_per_second:>5d}')
        
        # Apply heuristic analysis to find signals with energy.
        # These choices selected on 'MacBook Pro Microphone' with
        # Settings > Sound > Input level set at default (0.5).
        colour = 'black'
        if peak_signal > 0.:
          colour = 'red'
        elif (rms > args.rms_threshold and 
            (crest_factor >= args.crest_factor_thresholds[0] and
             crest_factor <= args.crest_factor_thresholds[1]
            ) and
            (not args.zero_crossings or 
             (zero_crossings_per_second >= args.zero_crossing_thresholds[0] and 
              zero_crossings_per_second <= args.zero_crossing_thresholds[1]))
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

  # Clean up resources.
  stream.stop_stream()
  stream.close()
  p.terminate()

if __name__ == '__main__':
    main()
