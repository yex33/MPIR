#!/bin/bash

#Download matrices
wget https://www.cise.ufl.edu/research/sparse/MM/TKK/cbuckle.tar.gz
wget https://www.cise.ufl.edu/research/sparse/MM/HB/1138_bus.tar.gz

#Unzip the files
tar -xvf cbuckle.tar.gz
mv cbuckle/cbuckle.mtx ./
rm -rf cbuckle.tar.gz cbuckle

tar -xvf 1138_bus.tar.gz
mv 1138_bus/1138_bus.mtx ./
rm -rf 1138_bus.tar.gz 1138_bus
