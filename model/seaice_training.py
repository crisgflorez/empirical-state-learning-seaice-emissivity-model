#
# (C) Copyright 2025 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" Top level control of the sea ice network parameter estimation using Keras and tensorflow """

import xarray as xr
import numpy as np
import tensorflow as tf
import seaice_model as sm
from seaice_sensors import SeaiceSensors
import argparse
import os
import sys
import timeit
from pathlib import Path


def predict_loop(model, distributor, nsplit, batchsize):
    model_tb=[]
    for isplit in range(nsplit):
        generator = sm.DataGenerator(distributor,isplit)
        tbout = model.predict(generator, batch_size = batchsize)
        if tf.rank(tbout) > 2:
            model_tb.append(tbout[:,:,0])
        else:
            model_tb.append(tbout)
    return model_tb

def get_args():
    parser = argparse.ArgumentParser('Sea ice training v2')
    parser.add_argument('--data', help='Directory containing the training data.', type=str, default='/perm/dnk8355/paper2026/netcdf_1april2024_31march2026')
    parser.add_argument('--sensors', help='Sensor names for training.', type=str, nargs='+', default=['METOP-B'])
    parser.add_argument('--output', help='Directory to store the output data.', type=str, default='/perm/dnk8355/paper2026/outputs_training/exp1_METOPB')
    parser.add_argument('--tag', help='Add a tag name to distinguish output files.', type=str, default='1april2024_31march2026_bg_emis07_with_losses_adjusted_obs_errors_bg_biasice0_ocean0_bg_bias_err1_7neurons_update_false_sic0_002_newimplementation_in_emisNN_no_angle_25epochs_sbatch_python3_12')
    parser.add_argument('--modeltag', help='If not training, optionally use an existing model with a different tag name.', type=str, default=None)
    parser.add_argument('--batchsize', help='Training batch size.', type=int, default=1024)
    parser.add_argument('--stepstart', help='Step in training data from which to start (default 0)', type=int, default=0)
    parser.add_argument('--nsteps', help='Number of time steps (usually days) in the model (default all)', type=int, default=-1)
    parser.add_argument('--nepochs', help='Number of training epochs (default 25)', type=int, default=8)
    parser.add_argument('--diagsonly', help='Compute output diagnostics from an already-trained model.', action='store_true')
    parser.add_argument('--trainonly', help='Only train the model (needed for large datasets to avoid OOM GPU errors).', action='store_true')
    parser.add_argument('--reproducible', help='Reproducible training; 3-5x slower.', action='store_true', default=True)

    # Read command line arguments
    # parse_known_args avoids errors from extra Jupyter/VSCode arguments
    args, unknown = parser.parse_known_args()

    if unknown:
        print("Ignoring unknown arguments:", unknown)

    return args

 
args = get_args()

# This is the top level directory location for all input and output data
ice_path=args.data+'/'
output_path=args.output+'/'
Path(output_path).mkdir(parents=True, exist_ok=True)

batchsize = args.batchsize
filename_append = args.tag
if args.modeltag is None:
    args.modeltag = args.tag

restrict_steps_to = args.nsteps
step_start = args.stepstart
nepochs = args.nepochs

do_diags = not args.trainonly
do_train = not args.diagsonly

# For larger training datasets, model.predict() crashes while model.fit() is fine - so predict over a split-up dataset
nsplit = 5 #Last version of Alan is 5 instead of 3

# ECMWF HPC specific config. 
os.environ['HDF5_USE_FILE_LOCKING']='FALSE'

tstart = timeit.default_timer()

# Reproducible results need the same seed, along with CUDA determinism (3 times slower) on GPUs.
if args.reproducible:
    tf.keras.utils.set_random_seed(409782)
    tf.config.experimental.enable_op_determinism()

sensor_info = SeaiceSensors(args.sensors)
nsensors = len(sensor_info.sensors)

sm.seaice_layers.obs_error = sensor_info.obs_error

if '10v' in sensor_info.channel_names:
    loss_channel_emis = np.where(sensor_info.channel_names == '10v')
    background_emis_value = 0.7
elif '24v' in sensor_info.channel_names:
    loss_channel_emis = np.where(sensor_info.channel_names == '24v')
    background_emis_value = 0.7
else:
    raise ValueError("No valid emissivity channel (10v or 24v) found for loss computation.")

if loss_channel_emis[0].size != 1:
    print("Error: there must be a channel 10v or 24v to constrain the surface emissivity estimate", file=sys.stderr)
    sys.exit(1)

nchannels, ngrid, nstep, nobs, nfields_float, nfields_int, x0, x0_int, y0, geolocation, grid = sm.training_data(
  ice_path, output_path, filename_append, sensor_info.sensors, sensor_info.channel_names,
  channel_maps=sensor_info.channel_maps, channel_basis=sensor_info.channel_basis,
  restrict_steps_to=restrict_steps_to, step_start=step_start)

pol0, pol1=sm.compute_polarization_coeffs(x0, sensor_info.polarisation_maps, sensor_info.sensor_type, sensor_info.zswath_width, sensor_info.zfov_spacing)
distributor = sm.TrainingDataDistributor(nobs, x0, x0_int, y0, pol0, pol1, batch_size=batchsize, nsplit=nsplit)

tf_strategy = tf.distribute.get_strategy()
with tf_strategy.scope():
    seaice_model = sm.SeaiceModel(nchannels=nchannels, ngrid=ngrid, nstep=nstep, nobs=nobs, grid=grid, 
      nfields_float=nfields_float, nfields_int=nfields_int, nsensors=nsensors,
      loss_channel_emis = loss_channel_emis[0][0],background_emis=background_emis_value,
      background_bias=sensor_info.background_bias, bg_error_bias=sensor_info.background_bias_error,
      nlag=1, alpha=[0.6,0.4], emissivity_mapping=(sensor_info.frequency_maps,sensor_info.polarisation_maps))
    seaice_model.initialize(ice_path+'ifs_seaice_initials_METOP-B_1apr2024_31march2026_without_land_without_nans.nc', ice_path+'ifs_tsfc_METOP-B_1apr2024_31march2026_dailyx_without_land.nc') 

    # Callback to allow updating the sea ice loss functions during training (in practice no effect as default loss is also 0.002)
    class BatchEpochCallback(tf.keras.callbacks.Callback):
        """
        Callback to:
        1. Update sea ice loss after certain epochs.
        2. Record the loss values for each batch.
        """
        def __init__(self):
            super().__init__()
            # Dictionary to save losses per batch
            self.batch_losses = {
                'loss': [],
                'loss_channel_weighted': [],
                'seaice_loss': [],
                'tsfc_loss': [],
                'emis_loss': [],
                'bias_loss': []
            }

        def on_epoch_begin(self, epoch, logs=None):
            if epoch >= 3:
                global seaice_model
                seaice_model.seaice_layer.update_loss(0.002) #Alan had a value of 0.002 I update it to 0.02
            
            for key in self.batch_losses.keys():
                self.batch_losses[key].append([])

        def on_train_batch_end(self, batch, logs=None):
            # logs contains the loss values for this batch
            if logs:
                for key in self.batch_losses.keys():
                    if key in logs:
                        self.batch_losses[key][-1].append(logs[key])


    model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs)
    model.summary()
    model.compile(optimizer="adam", loss=sm.seaice_layers.loss_channel_weighted, metrics=[sm.seaice_layers.loss_channel_weighted]) 

    if do_diags:
        # TB outputs of the initial network
        model_tb = predict_loop(model, distributor, nsplit, batchsize)
        sm.save_model_outputs(model_tb, geolocation,
          output_path+'tbsim_initial_'+filename_append+'.nc', sensor_info.channel_names)

    if do_train:
        distributor.makeSplit(1)
        generator = sm.DataGenerator(distributor,0)
        batch_epoch_callback = BatchEpochCallback()  # keep the object alive to store batch losses
        history = model.fit(generator, epochs = nepochs, batch_size=batchsize, callbacks=[batch_epoch_callback])
        seaice_model.save(history, filename_append, output_path, callback=batch_epoch_callback)
    else:
        seaice_model.load(args.modeltag, output_path)

    if do_diags:
        print("DEBUG: Entering second do_diags")
        distributor.makeSplit(nsplit)

        # Trained TB outputs 
        model_tb = predict_loop(model, distributor, nsplit, batchsize)
        sm.save_model_outputs(model_tb, geolocation,
          output_path+'tbsim_'+filename_append+'.nc', sensor_info.channel_names)

        # TB outputs with zero SIC
        seaice_weights_list = seaice_model.seaice_layer.get_weights() #for obtaining trained weights from seaice_layer which are the sea ice fraction 
        seaice_weights_list[0] = np.zeros( seaice_model.seaice_layer.seaice.shape,dtype=np.float32) #here we add zeros to sea ice fraction
        seaice_model.seaice_layer.set_weights(seaice_weights_list) #here we update sea ice fraction to zero
        model_tb = predict_loop(model, distributor, nsplit, batchsize) #predict uses the new weights (zeros) because model that comes from model = tf.keras.Model(seaice_model.inputs, seaice_model.outputs) contains that layer (seaice_model.seaice_layer) 
        sm.save_model_outputs(model_tb, geolocation,
          output_path+'tbzero_'+filename_append+'.nc', sensor_info.channel_names)

        # Emissivities at observation locations
        model2 = tf.keras.Model(seaice_model.inputs, seaice_model.emis_seaice)
        model2.compile()
        model_emis = predict_loop(model2, distributor, nsplit, batchsize)
        sm.save_model_outputs(model_emis, geolocation,
          output_path+'ice_emis_'+filename_append+'.nc', sensor_info.channel_names, varname='ice_emis', mask=False)

tend = timeit.default_timer()
seconds_elapsed = tend - tstart
print("Seconds elapsed: ",seconds_elapsed)

