#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from setuptools import setup

setup(name="GPGPU talk examples",
      version="1.0",
      author="Jacek Kołodziej",
      author_email="kolodziejj@gmail.com",
      install_requires=[
          "pyopencl",
          "pytest",
          ],
      )
