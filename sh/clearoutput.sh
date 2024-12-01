#!/bin/bash

# Find all Jupyter notebooks in the current dir
notebooks=$(ls -R | grep '\.ipynb$')

# Loop over each notebook and clear its output
for notebook in $notebooks; do
    echo "Clearing output from scripts/$notebook"
    jupyter nbconvert --clear-output --inplace "scripts/$notebook"
done
