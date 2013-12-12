from __future__ import division
import numpy as np
from datetime import datetime, timedelta
import time
import pyaudio


class SoundCardDataSource(object):
    def __init__(self, buffer_duration, channels=1, sampling_rate=44100,
                 chunk_size=1024):
        self.fs = sampling_rate
        self.channels = channels
        self.chunk_size = chunk_size

        # Check format is supported
        self.pyaudio = pyaudio.PyAudio()
        self.stream = None
        dev = self.pyaudio.get_default_input_device_info()
        if not self.pyaudio.is_format_supported(
                rate=sampling_rate,
                input_device=dev['index'],
                input_channels=channels,
                input_format=pyaudio.paInt16):
            raise RuntimeError("Unsupported audio format or rate")

        buffer_size = buffer_duration * sampling_rate
        num_chunks = buffer_size // chunk_size
        self.buffer = np.empty((num_chunks, chunk_size, channels))
        self.next_chunk = 0

        def callback(in_data, frame_count, time_info, status):
            samples = (np.frombuffer(in_data, dtype=np.int16)
                       .reshape((-1, self.channels))
                       .astype(float) / 2**15)
            self.buffer[self.next_chunk, :, :] = samples
            self.next_chunk = (self.next_chunk + 1) % self.buffer.shape[0]
            return (None, pyaudio.paContinue)

        self.stream = self.pyaudio.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            frames_per_buffer=self.chunk_size,
            rate=self.fs,
            stream_callback=callback,
            input=True
        )

    def __del__(self):
        print "@@@@@@@ closing @@@@@@@@"
        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()

    def get_buffer(self):
        """Return all chunks joined together"""
        a = self.buffer[:self.next_chunk]
        b = self.buffer[self.next_chunk:]
        return np.concatenate((b, a), axis=0) \
                 .reshape((self.buffer.shape[0] * self.buffer.shape[1],
                           self.buffer.shape[2]))

    @property
    def timeValues(self):
        N = self.buffer.shape[0] * self.buffer.shape[1]
        return np.linspace(0, N/self.fs, N)
