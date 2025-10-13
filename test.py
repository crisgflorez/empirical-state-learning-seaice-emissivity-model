import xarray as xr
import numpy as np
import seaice_model as sm
from seaice_sensors import SeaiceSensors
import argparse
import os
import sys
import timeit
from pathlib import Path

parser = argparse.ArgumentParser('Sea ice training v2')
parser.add_argument('--data', help='Directory containing the training data.', type=str)
parser.add_argument('--sensors', help='Sensor names for training.', type=str, nargs='+')
parser.add_argument('--output', help='Directory to store the output data.', type=str)
parser.add_argument('--tag', help='Add a tag name to distinguish output files.', type=str)
parser.add_argument('--modeltag', help='If not training, optionally use an existing model with a different tag name.', type=str, default=None)
parser.add_argument('--batchsize', help='Training batch size.', type=int, default=1024)
parser.add_argument('--stepstart', help='Step in training data from which to start (default 0)', type=int, default=0)
parser.add_argument('--nsteps',    help='Number of time steps (usually days) in the model (default all)', type=int, default=-1)
parser.add_argument('--nepochs',   help='Number of training epochs (default 8)', type=int, default=8)
parser.add_argument('--diagsonly', help='Compute output diagnostics from an already-trained model.',action='store_true')
parser.add_argument('--trainonly', help='Only train the model (needed for large datasets to avoid OOM GPU errors).',action='store_true')
parser.add_argument('--reproducible', help='Reproducible training; 3-5x slower.',action='store_true')
args = parser.parse_args()
print(args)