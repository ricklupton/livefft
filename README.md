livefft
=======

Real-time spectrum analyser in Python

Introduction
------------

A basic spectrum analyser which reads data from the sound card and
displays the time history and spectrum in real time.


Installation
------------

To install all packages in a new conda environment:

```
conda create -n livefft -f requirements.txt
source activate livefft
pip install pyqtgraph
```

Then to run the live FFT:

```
./livefft.py
```

Keyboard shortcuts
------------------

 - `SPACE`: pause
 - `L`: toggle log/linear scale
