(py)OpenCL Examples
===================

Python
------

First, install depencies::

   pip install numpy pyopencl pytest

Use pure Python::

   python examples/test_vectors_cl.py

or ``py.test`` to run all examples::

   py.test

C
-

Compile::

   gcc -std=c99 examples/vectors_cl.c -o vectors_cl -l OpenCL

and run::

   ./vectors_cl
