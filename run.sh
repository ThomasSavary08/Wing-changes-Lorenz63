#!/bin/bash

# Path of the file to check whether CLVs/training files have already been computed
CLV_path="./CLVs/trainCLV_1.npy"
data_path="./Data/x_train.npy"

# Check and run application
if [ -e "$CLV_path" ]; then
	if [ -e "$data_path" ]; then
		echo "Running animation..."
		python animation.py
	else
		echo "Creation of training files..."
		cd Data
		python preprocessing.py
		cd ..
		echo "Running animation..."
		python animation.py
	fi 
else
	echo "Computing CLVs..."
	cd CLVs
	python compute_CLV.py
	cd ..
	echo "Creation of training files..."
	cd Data
	python preprocessing.py
	cd ..
	echo "Running animation..."
	python animation.py
fi 