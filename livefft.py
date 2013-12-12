#!/usr/bin/env pythonw

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
import pyqtgraph as pg
from recorder import SoundCardDataSource


def fft_slices(x):
    Nslices, Npts = x.shape
    numFreqs = Npts//2 + 1  # number of frequencies in one-sided spectrum
    window = np.hanning(Npts)

    # Calculate FFT
    fx = np.fft.fft(window[np.newaxis, :] * x, axis=1)

    # Convert to normalised PSD
    Pxx = abs(fx[:, :numFreqs])**2 / (np.abs(window)**2).sum()

    # Scale for one-sided (excluding DC and Nyquist frequencies)
    Pxx[1:-1] *= 2

    # And scale by frequency to get a result in (dB/Hz)
    # Pxx /= Fs
    return Pxx


class LiveFFTWindow(pg.GraphicsWindow):
    def __init__(self, recorder):
        super(LiveFFTWindow, self).__init__(title="Live FFT")
        self.recorder = recorder
        self.paused = False
        self.logScale = True

        # Setup plots
        self.p1 = self.addPlot()
        self.p1.setLabel('bottom', 'Time', 's')
        self.p1.setLabel('left', 'Amplitude')
        self.p1.setTitle("")
        self.ts = self.p1.plot(pen='y')
        self.nextRow()
        self.p2 = self.addPlot()
        self.p2.setLabel('bottom', 'Frequency', 'Hz')
        self.spec = self.p2.plot(pen=(50, 100, 200),
                                 brush=(50,100,200),
                                 fillLevel=-100)

        # Data ranges
        nFreqs = self.recorder.buffer.shape[1] // 2 + 1
        self.freqValues = self.recorder.fs * np.arange(nFreqs) / nFreqs
        self.timeValues = self.recorder.timeValues
        self.resetRanges()

        # Timer to update plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

    def resetRanges(self):
        self.p1.setRange(xRange=(0, self.timeValues[-1]), yRange=(-0.5, 0.5))
        if self.logScale:
            # Only show half the frequency range
            self.p2.setRange(xRange=(0, self.freqValues[-1] / 2),
                             yRange=(-80, 20))
            self.spec.setData(fillLevel=-100)
            self.p2.setLabel('left', 'PSD', 'dB / Hz')
        else:
            # Only show half the frequency range
            self.p2.setRange(xRange=(0, self.freqValues[-1] / 2),
                             yRange=(0, 10))
            self.spec.setData(fillLevel=0)
            self.p2.setLabel('left', 'PSD', '1 / Hz')

    def update(self):
        if self.paused:
            return
        data = self.recorder.get_buffer()
        Pxx = fft_slices(self.recorder.buffer[:, :, 0]).mean(axis=0)
        if self.logScale:
            Pxx = 10*np.log10(Pxx)

        self.ts.setData(x=self.timeValues, y=data[:, 0], autoDownsample=True)
        # Only show half the frequency range
        self.spec.setData(x=self.freqValues[:len(Pxx)//2], y=Pxx[:len(Pxx)//2])

    def keyPressEvent(self, event):
        if event.text() == " ":
            self.paused = not self.paused
            self.p1.setTitle("PAUSED" if self.paused else "")
        elif event.text() == "l":
            self.logScale = not self.logScale
            self.resetRanges()
        else:
            super(LiveFFTWindow, self).keyPressEvent(event)


# Setup plots
#QtGui.QApplication.setGraphicsSystem('opengl')
app = QtGui.QApplication([])
pg.setConfigOptions(antialias=True)

# Setup recorder
#FS = 11025
FS = 22000
#FS = 44000
recorder = SoundCardDataSource(0.5, sampling_rate=FS, chunk_size=2*1024)
win = LiveFFTWindow(recorder)

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
