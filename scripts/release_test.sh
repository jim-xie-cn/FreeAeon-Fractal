#!/bin/bash

cd ../

python setup.py sdist bdist_wheel

twine upload \
  --repository testpypi \
  --verbose \
  -u __token__ \
  -p $1 \
  dist/*
