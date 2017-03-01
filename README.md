livefft
=========

Real-time spectrum analyser in Python

Introduction
------------

A basic spectrum analyser which reads data from the sound card and
displays the time history and spectrum in real time.


Installation
---------------

### Ubuntu 16.04 LTS

Using the system Python 2, install dependencies using apt:
```
sudo apt install python-pyqtgraph python-pyaudio
```

Clone & run livefft:
```
git clone https://github.com/ricklupton/livefft
python livefft/livefft.py
```

### Anaconda

Using the [Anaconda python distribution](https://www.continuum.io/downloads),
create a new conda environment with the required pacakges and activate it:

```
conda create -n livefft -f requirements.txt
source activate livefft
```

Clone & run livefft:
```
git clone https://github.com/ricklupton/livefft
python livefft/livefft.py
```

Usage
------

The time signal is shown in the top plot, the spectrum below.

### Adjusting plots

Adjust the zoom of plots using the mouse wheel over the axes, or by dragging
with the right mouse button. Pan using the middle mouse button.

Tip: To zoom the frequency axis without losing the origin, place the mouse over
0 then spin the mouse wheel.

See the
[pyqtgraph documentation](http://www.pyqtgraph.org/documentation/mouse_interaction.html#d-graphics) for
more details.

### Sampling time

Change the length of the sampling buffer with the `+` and `-` keys. Using a
longer buffer can be useful to make the spectrum respond more slowly, or for
looking at low frequencies. Using a shorter buffer makes the spectrum respond
more quickly.

### Other keyboard shortcuts

 - `SPACE`: pause
 - `L`: toggle log/linear scale
 - `D`: toggle downsampling (for faster graph updates)
