#!/bin/bash
nvcc $1 -o main.run -lglut -lGL --Wno-deprecated-declarations
./main.run
rm main.run
