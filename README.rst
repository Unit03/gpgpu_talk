(py)OpenCL Examples
===================

Used in `(py)OpenCL on video cards: GPGPU intro
<http://kolodziejj.info/talks/gpgpu/>`_ talk (polish only).

Please note, you must have working ``OpenCL`` installed on your system, along
with proper Installable Client Driver (ICD).

Python (2/3)
------------

First, install dependencies::

   pip install numpy pyopencl pytest

Use pure Python::

   python examples/test_vectors_cl.py

or ``py.test`` to run all examples::

   py.test

C
-

Go to ``examples`` directory, compile::

   cd examples/
   gcc -std=c99 vectors_cl.c -o vectors_cl -l OpenCL

and run::

   ./vectors_cl
