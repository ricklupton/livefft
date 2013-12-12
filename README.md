livefft
=======

Real-time spectrum analyser in Python

Introduction
------------

A basic spectrum analyser which reads data from the sound card and
displays the time history and spectrum in real time.


Installation
------------

As of writing the `pyqtgraph` package isn't available in Anaconda. You
can install it using `pip`, or on OS X you can use my packaged
version; first add the channel to conda:

```
# Add the defaults only if this is the first time you have customized conda's channels
conda config --add channels defaults

# Add rcl33's channel for the pyqtgraph package on OS X
conda config --add channels http://conda.binstar.org/rcl33
```

Then install all packages in a new conda environment and activate it:

```
conda create -n livefft -f requirements.txt
source activate livefft
```

Then to run the live FFT:

```
./livefft.py
```

Keyboard shortcuts
------------------

 - `SPACE`: pause
 - `L`: toggle log/linear scale
