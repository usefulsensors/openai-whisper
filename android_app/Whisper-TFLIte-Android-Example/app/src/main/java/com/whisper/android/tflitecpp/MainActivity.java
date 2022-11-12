package com.whisper.android.tflitecpp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TextView;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;

public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private Button playAudioButton;
    private Button recordAudioButton;
    private Button transcribeButton;
    private Spinner audioClipSpinner;
    private WavAudioRecorder mRecorder;
    private String wavFilename;
    private static String fileName = null;
    private final static String TAG = "TfLiteASRDemo";
    private MediaPlayer mediaPlayer = new MediaPlayer();
    private MediaRecorder recorder = null;
    // Requesting permission to RECORD_AUDIO
    private boolean permissionToRecordAccepted = false;
    private static final int REQUEST_RECORD_AUDIO_PERMISSION = 200;
    private String [] permissions = {Manifest.permission.RECORD_AUDIO};

    private final static String[] WAV_FILENAMES = {"jfk.wav","test.wav", "test_1.wav","android_record.wav"};

    // Requesting permission to RECORD_AUDIO
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode){
            case REQUEST_RECORD_AUDIO_PERMISSION:
                permissionToRecordAccepted  = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                break;
        }
        if (!permissionToRecordAccepted ) finish();

    }

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        audioClipSpinner = findViewById(R.id.audio_clip_spinner);
        // Record to the external cache directory for visibility
        fileName = getExternalCacheDir().getAbsolutePath();
        fileName += "/android_record.wav";
        Log.e(TAG, fileName);
        ActivityCompat.requestPermissions(this, permissions, REQUEST_RECORD_AUDIO_PERMISSION);
        ArrayAdapter<String>adapter = new ArrayAdapter<String>(MainActivity.this,
                android.R.layout.simple_spinner_item, WAV_FILENAMES);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        audioClipSpinner.setAdapter(adapter);
        audioClipSpinner.setOnItemSelectedListener(this);
        playAudioButton = findViewById(R.id.play);
        playAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(wavFilename.equals("android_record.wav")){
                    //Log.e(TAG, "Need to implement Record audio transcribe");
                    Log.e(TAG, fileName);
                    try {
                        mediaPlayer.reset();
                        mediaPlayer.setDataSource(fileName);
                        mediaPlayer.prepare();
                    } catch (Exception e) {
                        Log.e(TAG, e.getMessage());
                    }

                }else {
                    try (AssetFileDescriptor assetFileDescriptor = getAssets().openFd(wavFilename)) {
                        mediaPlayer.reset();
                        mediaPlayer.setDataSource(assetFileDescriptor.getFileDescriptor(), assetFileDescriptor.getStartOffset(), assetFileDescriptor.getLength());
                        mediaPlayer.prepare();
                    } catch (Exception e) {
                        Log.e(TAG, e.getMessage());
                    }

                }
                mediaPlayer.start();

            }
        });
        recordAudioButton = findViewById(R.id.record);
        recordAudioButton.setText("Record");
        mRecorder = WavAudioRecorder.getInstanse();
        mRecorder.setOutputFile(fileName);
        recordAudioButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (WavAudioRecorder.State.INITIALIZING == mRecorder.getState()) {
                    Log.e(TAG, "INITIALIZING");
                    mRecorder.setOutputFile(fileName);
                    mRecorder.prepare();
                    mRecorder.start();
                    recordAudioButton.setText("Stop");
                } else if (WavAudioRecorder.State.ERROR == mRecorder.getState()) {
                    Log.e(TAG, "ERROR");
                    mRecorder.release();
                    mRecorder = WavAudioRecorder.getInstanse();
                    mRecorder.setOutputFile(fileName);
                    recordAudioButton.setText("Record");
                } else {
                    Log.e(TAG, "OTHER state");
                    mRecorder.stop();
                    mRecorder.reset();
                    mRecorder.release();
                    mRecorder = WavAudioRecorder.getInstanse();
                    recordAudioButton.setText("Record");
                }
            }
        });
        transcribeButton = findViewById(R.id.recognize);
        // Example of a call to a native method
        TextView tv = findViewById(R.id.result);
        transcribeButton.setOnClickListener(new View.OnClickListener() {
            @RequiresApi(api = Build.VERSION_CODES.M)
            @Override
            public void onClick(View view) {
                try {
                    if(wavFilename.equals("android_record.wav")){
                        //Log.e(TAG, "Need to implement Record audio transcribe");
                        Log.e(TAG, fileName);
                        tv.setText(loadModelJNI(getAssets(), fileName, 1));
                    }else {
                        tv.setText(loadModelJNI(getAssets(), wavFilename, 0));
                    }
                } catch (Exception e) {
                    Log.e(TAG, e.getMessage());
                }
            }
        });
    }


    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */

    // Load model by TF Lite C++ API
    private native String loadModelJNI(AssetManager assetManager, String fileName, int is_recorded);

    @Override
    public void onItemSelected(AdapterView<?> parent, View v, int position, long id) {
        wavFilename = WAV_FILENAMES[position];
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (null != recorder) {
            recorder.release();
        }
    }

    public static class WavAudioRecorder {
        private final static int[] sampleRates = {44100, 22050, 16000, 11025, 8000};

        public static WavAudioRecorder getInstanse() {
            WavAudioRecorder result = null;
            int i=2;
            do {
                result = new WavAudioRecorder(MediaRecorder.AudioSource.MIC,
                        sampleRates[i],
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT);
            } while((++i<sampleRates.length) & !(result.getState() == State.INITIALIZING));
            return result;
        }

        /**
         * INITIALIZING : recorder is initializing;
         * READY : recorder has been initialized, recorder not yet started
         * RECORDING : recording
         * ERROR : reconstruction needed
         * STOPPED: reset needed
         */
        public enum State {INITIALIZING, READY, RECORDING, ERROR, STOPPED};

        public static final boolean RECORDING_UNCOMPRESSED = true;
        public static final boolean RECORDING_COMPRESSED = false;

        // The interval in which the recorded samples are output to the file
        // Used only in uncompressed mode
        private static final int TIMER_INTERVAL = 120;

        // Recorder used for uncompressed recording
        private AudioRecord audioRecorder = null;

        // Output file path
        private String          filePath = null;

        // Recorder state; see State
        private State state;

        // File writer (only in uncompressed mode)
        private RandomAccessFile randomAccessWriter;

        // Number of channels, sample rate, sample size(size in bits), buffer size, audio source, sample size(see AudioFormat)
        private short                    nChannels;
        private int                      sRate;
        private short                    mBitsPersample;
        private int                      mBufferSize;
        private int                      mAudioSource;
        private int                      aFormat;

        // Number of frames/samples written to file on each output(only in uncompressed mode)
        private int                      mPeriodInFrames;

        // Buffer for output(only in uncompressed mode)
        private byte[]                   buffer;

        // Number of bytes written to file after header(only in uncompressed mode)
        // after stop() is called, this size is written to the header/data chunk in the wave file
        private int                      payloadSize;

        /**
         *
         * Returns the state of the recorder in a WavAudioRecorder.State typed object.
         * Useful, as no exceptions are thrown.
         *
         * @return recorder state
         */
        public State getState() {
            return state;
        }


        private AudioRecord.OnRecordPositionUpdateListener updateListener = new AudioRecord.OnRecordPositionUpdateListener() {
            //	periodic updates on the progress of the record head
            public void onPeriodicNotification(AudioRecord recorder) {
                if (State.STOPPED == state) {
                    Log.d(WavAudioRecorder.this.getClass().getName(), "recorder stopped");
                    return;
                }
                int numOfBytes = audioRecorder.read(buffer, 0, buffer.length); // read audio data to buffer
    //			Log.d(WavAudioRecorder.this.getClass().getName(), state + ":" + numOfBytes);
                try {
                    randomAccessWriter.write(buffer); 		  // write audio data to file
                    payloadSize += buffer.length;
                } catch (IOException e) {
                    Log.e(WavAudioRecorder.class.getName(), "Error occured in updateListener, recording is aborted");
                    e.printStackTrace();
                }
            }
            //	reached a notification marker set by setNotificationMarkerPosition(int)
            public void onMarkerReached(AudioRecord recorder) {
            }
        };
        /**
         *
         *
         * Default constructor
         *
         * Instantiates a new recorder
         * In case of errors, no exception is thrown, but the state is set to ERROR
         *
         */
        public WavAudioRecorder(int audioSource, int sampleRate, int channelConfig, int audioFormat) {
            try {
                if (audioFormat == AudioFormat.ENCODING_PCM_16BIT) {
                    mBitsPersample = 16;
                } else {
                    mBitsPersample = 8;
                }

                if (channelConfig == AudioFormat.CHANNEL_IN_MONO) {
                    nChannels = 1;
                } else {
                    nChannels = 2;
                }

                mAudioSource = audioSource;
                sRate   = sampleRate;
                aFormat = audioFormat;

                mPeriodInFrames = sampleRate * TIMER_INTERVAL / 1000;		//?
                mBufferSize = mPeriodInFrames * 2  * nChannels * mBitsPersample / 8;		//?
                if (mBufferSize < AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)) {
                    // Check to make sure buffer size is not smaller than the smallest allowed one
                    mBufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat);
                    // Set frame period and timer interval accordingly
                    mPeriodInFrames = mBufferSize / ( 2 * mBitsPersample * nChannels / 8 );
                    Log.w(WavAudioRecorder.class.getName(), "Increasing buffer size to " + Integer.toString(mBufferSize));
                }

                audioRecorder = new AudioRecord(audioSource, sampleRate, channelConfig, audioFormat, mBufferSize);

                if (audioRecorder.getState() != AudioRecord.STATE_INITIALIZED) {
                    throw new Exception("AudioRecord initialization failed");
                }
                audioRecorder.setRecordPositionUpdateListener(updateListener);
                audioRecorder.setPositionNotificationPeriod(mPeriodInFrames);
                filePath = null;
                state = State.INITIALIZING;
            } catch (Exception e) {
                if (e.getMessage() != null) {
                    Log.e(WavAudioRecorder.class.getName(), e.getMessage());
                } else {
                    Log.e(WavAudioRecorder.class.getName(), "Unknown error occured while initializing recording");
                }
                state = State.ERROR;
            }
        }

        /**
         * Sets output file path, call directly after construction/reset.
         *
         * @param output file path
         *
         */
        public void setOutputFile(String argPath) {
            try {
                if (state == State.INITIALIZING) {
                    filePath = argPath;
                }
            } catch (Exception e) {
                if (e.getMessage() != null) {
                    Log.e(WavAudioRecorder.class.getName(), e.getMessage());
                } else {
                    Log.e(WavAudioRecorder.class.getName(), "Unknown error occured while setting output path");
                }
                state = State.ERROR;
            }
        }


        /**
         *
         * Prepares the recorder for recording, in case the recorder is not in the INITIALIZING state and the file path was not set
         * the recorder is set to the ERROR state, which makes a reconstruction necessary.
         * In case uncompressed recording is toggled, the header of the wave file is written.
         * In case of an exception, the state is changed to ERROR
         *
         */
        public void prepare() {
            try {
                if (state == State.INITIALIZING) {
                    if ((audioRecorder.getState() == AudioRecord.STATE_INITIALIZED) & (filePath != null)) {
                        // write file header
                        randomAccessWriter = new RandomAccessFile(filePath, "rw");
                        randomAccessWriter.setLength(0); // Set file length to 0, to prevent unexpected behavior in case the file already existed
                        randomAccessWriter.writeBytes("RIFF");
                        randomAccessWriter.writeInt(0); // Final file size not known yet, write 0
                        randomAccessWriter.writeBytes("WAVE");
                        randomAccessWriter.writeBytes("fmt ");
                        randomAccessWriter.writeInt(Integer.reverseBytes(16)); // Sub-chunk size, 16 for PCM
                        randomAccessWriter.writeShort(Short.reverseBytes((short) 1)); // AudioFormat, 1 for PCM
                        randomAccessWriter.writeShort(Short.reverseBytes(nChannels));// Number of channels, 1 for mono, 2 for stereo
                        randomAccessWriter.writeInt(Integer.reverseBytes(sRate)); // Sample rate
                        randomAccessWriter.writeInt(Integer.reverseBytes(sRate*nChannels*mBitsPersample/8)); // Byte rate, SampleRate*NumberOfChannels*mBitsPersample/8
                        randomAccessWriter.writeShort(Short.reverseBytes((short)(nChannels*mBitsPersample/8))); // Block align, NumberOfChannels*mBitsPersample/8
                        randomAccessWriter.writeShort(Short.reverseBytes(mBitsPersample)); // Bits per sample
                        randomAccessWriter.writeBytes("data");
                        randomAccessWriter.writeInt(0); // Data chunk size not known yet, write 0
                        buffer = new byte[mPeriodInFrames*mBitsPersample/8*nChannels];
                        state = State.READY;
                    } else {
                        Log.e(WavAudioRecorder.class.getName(), "prepare() method called on uninitialized recorder");
                        state = State.ERROR;
                    }
                } else {
                    Log.e(WavAudioRecorder.class.getName(), "prepare() method called on illegal state");
                    release();
                    state = State.ERROR;
                }
            } catch(Exception e) {
                if (e.getMessage() != null) {
                    Log.e(WavAudioRecorder.class.getName(), e.getMessage());
                } else {
                    Log.e(WavAudioRecorder.class.getName(), "Unknown error occured in prepare()");
                }
                state = State.ERROR;
            }
        }

        /**
         *
         *
         *  Releases the resources associated with this class, and removes the unnecessary files, when necessary
         *
         */
        public void release() {
            if (state == State.RECORDING) {
                stop();
            } else {
                if (state == State.READY){
                    try {
                        randomAccessWriter.close(); // Remove prepared file
                    } catch (IOException e) {
                        Log.e(WavAudioRecorder.class.getName(), "I/O exception occured while closing output file");
                    }
                    (new File(filePath)).delete();
                }
            }

            if (audioRecorder != null) {
                audioRecorder.release();
            }
        }

        /**
         *
         *
         * Resets the recorder to the INITIALIZING state, as if it was just created.
         * In case the class was in RECORDING state, the recording is stopped.
         * In case of exceptions the class is set to the ERROR state.
         *
         */
        public void reset() {
            try {
                if (state != State.ERROR) {
                    release();
                    filePath = null; // Reset file path
                    audioRecorder = new AudioRecord(mAudioSource, sRate, nChannels, aFormat, mBufferSize);
                    if (audioRecorder.getState() != AudioRecord.STATE_INITIALIZED) {
                        throw new Exception("AudioRecord initialization failed");
                    }
                    audioRecorder.setRecordPositionUpdateListener(updateListener);
                    audioRecorder.setPositionNotificationPeriod(mPeriodInFrames);
                    state = State.INITIALIZING;
                }
            } catch (Exception e) {
                Log.e(WavAudioRecorder.class.getName(), e.getMessage());
                state = State.ERROR;
            }
        }

        /**
         *
         *
         * Starts the recording, and sets the state to RECORDING.
         * Call after prepare().
         *
         */
        public void start() {
            if (state == State.READY) {
                payloadSize = 0;
                audioRecorder.startRecording();
                audioRecorder.read(buffer, 0, buffer.length);	//[TODO: is this necessary]read the existing data in audio hardware, but don't do anything
                state = State.RECORDING;
            } else {
                Log.e(WavAudioRecorder.class.getName(), "start() called on illegal state");
                state = State.ERROR;
            }
        }

        /**
         *
         *
         *  Stops the recording, and sets the state to STOPPED.
         * In case of further usage, a reset is needed.
         * Also finalizes the wave file in case of uncompressed recording.
         *
         */
        public void stop() {
            if (state == State.RECORDING) {
                audioRecorder.stop();
                try {
                    randomAccessWriter.seek(4); // Write size to RIFF header
                    randomAccessWriter.writeInt(Integer.reverseBytes(36+payloadSize));

                    randomAccessWriter.seek(40); // Write size to Subchunk2Size field
                    randomAccessWriter.writeInt(Integer.reverseBytes(payloadSize));

                    randomAccessWriter.close();
                } catch(IOException e) {
                    Log.e(WavAudioRecorder.class.getName(), "I/O exception occured while closing output file");
                    state = State.ERROR;
                }
                state = State.STOPPED;
            } else {
                Log.e(WavAudioRecorder.class.getName(), "stop() called on illegal state");
                state = State.ERROR;
            }
        }
    }
}