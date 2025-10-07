# empirical-state-learning-seaice-emissivity-model

Python code for training a hybrid empirical-physical network to represent passive microwave observations over ocean and sea ice areas. The unknowns, to be found by training, are the sea ice concentration, the physical properties of the sea ice and any overlying snow, and the microwave surface emissivity of the sea ice surface. These unknown state and modelling components are embedded in a network of known physical models.

The learning approach is a hybrid of data assimilation and machine learning that simultaneously trains an empirical model component and an empirical geophysical input state. The  empirical model component is a simple neural network and its input creates a latent space that defines the empirical geophysical state. It is proposed to call this an "empirical state" method.

The empirical sea ice emissivity model trained using this code can then be plugged into a weather forecasting system to add a sea ice concentration analysis to the atmospheric data assimilation system, and to enable assimilation of microwave data with strong surface sensitivities over sea ice for the first time

## Version 1: AMSR2 only and single-layer linear NN

Version 1 has been used to provide sea ice surface emissivity modelling in ECMWF's operational physical atmospheric data assimilation system, IFS cycle 49r1, since 12th November 2024, in order to allow assimilation of microwave imager observations over sea ice.

### Citations

Simultaneous inference of sea ice state and surface emissivity model using machine learning and data assimilation, A.J. Geer, JAMES, https://doi.org/10.1029/2023MS004080, 2024

Joint estimation of sea ice and atmospheric state from microwave imagers in operational weather forecasting, A.J. Geer, Q J R Met Soc, https://doi.org/10.1002/qj.4797, 2024

The code is archived at https://doi.org/10.5281/zenodo.10013542

The data is archived at https://doi.org/10.5281/zenodo.10009498

### Dependency versions

The Python code was run on Python 3.8.8-01 including Tensorflow and Keras 2.8.0 on the ECMWF ATOS supercomputer CPU nodes.

## Version 2: Multi-sensor training, expanded frequency coverage and 2-layer nonlinear NN

The version 2 framework was used to update the sea ice emissivity forward modelling for IFS cycle 50r1, which is intended to go operational in early 2025. In training, the "v2" empirical emissivity model has two gridded empirical state variables and one observation space empirical state variable, which is intended to provide more flexibility to fit fast-changing features of the summer sea ice (e.g. freeze-thaw processes). The model takes as input these three empirical variables, the modified skin temperature, and the frequency. It outputs the surface emissivity at that frequency for vertical and horizontal polarisation. Hence it is no longer tied to the AMSR2 channels. 

The model used in cycle 50r1 was trained against AMSR2, GMI and SSMIS (DMSP-F17) for the same training period as v1, 2020 to 2021, and using channels covering 10 to 190 GHz. In cycle 50r1, the sea ice concentration retrieved from the assimilation of AMSR2 observations in the atmospheric data assimilation component, using the empirical sea ice emissivity model, will also now be assimilated within the ocean data assimilation system in an outer-loop coupling framework.

From a technical perspective, the v2 code is run from the command line using command line arguments. There is no distinction between "monthly" and "yearly" training, but rather a shorter period of training can be selected from within the full available training dataset if required. All training and inference jobs (the latter referred to as "diagnostics") are run from a new top-level code seaice_training.py. It now possible to run the training on GPUs and training data is now provided through a Keras generator to manage memory. Training data is stored in subdirectories, each named for the relevant instrument (e.g. amsr2, gmi etc.) but within each directory, it uses the same NetCDF file structure as v1. Outputs and diagnostics are written to a variety of NetCDF files which should be more self-describing and comprehensive than in v1.

Further documentation and Zenodo data archiving is in preparation.

### Running training or interface from the command line

Example call to generate the v2 model in 50r1 using a full year of training data (note that CUDA reproducibility settings were not used, and hence the exact model cannot be replicated):

```bash
python3 seaice_training.py --data=[top level directory containing training data] --output=[directory for netcdf output files] --batchsize=1024 --tag=[identifier for this training run] --nepochs=25 --nsteps=365
```

```
options:
  -h, --help            show this help message and exit
  --data DATA           Directory containing the training data.
  --sensors SENSORS [SENSORS ...]
                        Sensor names for training.
  --output OUTPUT       Directory to store the output data.
  --tag TAG             Add a tag name to distinguish output files.
  --modeltag MODELTAG   If not training, optionally use an existing model with a different tag name.
  --batchsize BATCHSIZE
                        Training batch size.
  --stepstart STEPSTART
                        Step in training data from which to start (default 0)
  --nsteps NSTEPS       Number of time steps (usually days) in the model (default all)
  --nepochs NEPOCHS     Number of training epochs (default 8)
  --diagsonly           Compute output diagnostics from an already-trained model.
  --trainonly           Only train the model (needed for large datasets to avoid OOM GPU errors).
  --reproducible        Reproducible training; 3-5x slower.
```

### Dependency versions

Python 3.10.10-01
Tensorflow and Keras 2.17.0
Trained on NVIDIA A100-SXM4-40GB

## Dependencies  

tensorflow
xarray
numpy

## License

Copyright 2025 European Centre for Medium-Range Weather Forecasts (ECMWF)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In applying this licence, ECMWF does not waive the privileges and immunities
granted to it by virtue of its status as an intergovernmental organisation nor
does it submit to any jurisdiction.


