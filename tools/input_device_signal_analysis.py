"""
Input device signal analysis with plotting.

This blocks the audio device.
TODO(guy): make non-blocking version.

Uses pyaudio: https://pypi.org/project/PyAudio/ 
https://stackoverflow.com/questions/35970282/what-are-chunks-samples-and-frames-when-using-pyaudio
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pyaudio

from scipy import signal


def get_input_device_id(p):
    """Returns user selection of available input devices, using PyAudio."""
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get('deviceCount')
    print("Available input devices:")
    for i in range(num_devices):
        if p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0:
            device_name = p.get_device_info_by_host_api_device_index(0, i).get('name')
            print(f"{i}: {device_name}")
    # Prompt user to enter input device ID
    device_id = input("Enter input device ID: ")
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
    parser.add_argument('-f', '--filter_highpass', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Remove low frequency from the audio.')
    parser.add_argument('-p', '--plot', default=True,
                        action=argparse.BooleanOptionalAction,
                        help='Show plot.')
    parser.add_argument('-r', '--rms_threshold', type=float, default=-50.,
                        help=('Threshold value for the buffer rms level in dB, '
                              'e.g.: -50'))
    parser.add_argument('-s', '--sampling_frequency', type=int, default=16000,
                        help='Audio stream sampling frequency in hertz, Hz.')
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

    # Open microphone stream
    stream = p.open(format=FORMAT, 
                    channels=CHANNELS, rate=args.sampling_frequency,
                    input=True, input_device_index=device_info["index"], 
                    frames_per_buffer=args.chunk_size)

    if args.plot:
      # Initialize plot figure
      plt.figure()

    # Initialize buffer variables
    signal_buffer = np.zeros(int(args.sampling_frequency * args.buffer_time))
    plot_index = 0

    print(
      "Signals with energy are marked blue (simple analysis will try \n"
      "to reject constant noise signals).\nPress Ctrl-c to quit...")

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
            normalized_zero_crossings = zero_crossings / (
               args.sampling_frequency * args.buffer_time)

            text_str = (f"RMS:{rms:>+2.1f}dB  "
                        f"Peak:{rms:>+2.1f}dB  "
                        f"C.F.:{crest_factor:>4.1f}  "
                        f"Z.C.:{zero_crossings:>5d} "
                        f"{normalized_zero_crossings:>4.2f}")
            
            # Apply heuristic analysis to find signals with energy.
            # Checks for crest factor and zero crossings can catch 
            # constant noise types, such as high frequency 'hissing'.
            # These choices selected on 'MacBook Pro Microphone' with
            # Settings > Sound > Input level set at default (0.5).
            colour = 'black'
            if (rms > args.rms_threshold and 
                crest_factor > args.crest_factor_threshold and 
                (normalized_zero_crossings > 0.025 and 
                 normalized_zero_crossings < 0.25)
                ):
               colour = 'blue'
            
            if args.plot:
              plt.clf()
              plt.plot(audio, colour)
              plt.title(text_str)
              plt.ylim([-1., 1.])
              plt.grid()
              plt.pause(0.001)

            print(
               add_color_escape_code(text_str, colour))

            plot_index = 0

    except KeyboardInterrupt:
      print("\nCtrl-C exception, cleaning up resources.")

    # Clean up resources
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()
