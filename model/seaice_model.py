#
# (C) Copyright 2025 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" Defines the sea ice Bayesian network as a Keras model, plus routines to read training data and save results"""

import tensorflow as tf
import numpy as np
import xarray as xr
import seaice_layers
import os
import pandas as pd

#julian_day_attrs = {'units':'days since -4714-11-24 12:00:00.000','calendar':'proleptic_gregorian'} This has been changed to put in the netcdf files the correct date and time
julian_day_attrs = {'long_name': 'Astronomical Julian Day',
            'standard_name': 'julian_day',
            'description': 'Continuous astronomical Julian Day starting at 4713 BCE-01-01 12:00 UTC'
        }

class SeaiceModel:
    """
    Define a Keras model for the sea ice Bayesian network
    """
    def __init__(self, nchannels=18, nprop_grid=2, nprop_obs=1, ngrid=1, nstep=1, nobs=1, nlag=0, alpha=[1.0],
                 bg_error_seaice=0.002, bg_error_false_sic=0.02, bg_error_emis=1e-5, background_emis=0.8, seaice_use_loss=False,
                 seaice_use_pdf_loss=True, seaice_use_tsfc_loss=True, penalise_false_sic=True, emis_use_bounds_loss=True,
                 loss_channel_emis=0, background_bias=None, bg_error_bias=None,
                 width_nn=7, grid=None, nfields_float=7, nfields_int=1, nsensors=3, emissivity_mapping=None):
        """
        Initialize the network structure and internal and external dimensions
        """
        self.setup=dict()
        self.setup['nsensors'] = nsensors
        self.setup['nfields_float'] = nfields_float
        self.setup['nfields_int'] = nfields_int
        self.setup['nchannels'] = nchannels
        self.setup['nbiases'] = 2
        self.setup['nprop_grid'] = nprop_grid
        self.setup['nprop_obs'] = nprop_obs
        self.setup['ngrid'] = ngrid
        self.setup['nstep'] = nstep
        self.setup['nlag']  = nlag
        self.setup['nobs']  = nobs
        self.setup['npol']  = 2
        self.setup['alpha'] = alpha
        self.setup['width_nn'] = width_nn
     
        self.setup['bg_error_false_sic'] = bg_error_false_sic
        self.setup['bg_error_seaice'] = bg_error_seaice
        self.setup['seaice_use_tsfc_loss'] = seaice_use_tsfc_loss     
        self.setup['penalise_false_sic'] = penalise_false_sic

        self.setup['emis_use_bounds_loss'] = emis_use_bounds_loss
        self.setup['bg_error_emis']   = bg_error_emis
        self.setup['background_emis'] = background_emis
        self.setup['loss_channel_emis'] = loss_channel_emis

        self.setup['bg_error_bias']   = bg_error_bias
        self.setup['background_bias'] = background_bias

        self.grid = grid
     
        # TF model training is much faster if the inputs are a single tensor
        self.inputs_float = tf.keras.Input(shape=(nfields_float+nchannels*7,)) #7 is the number of channel-dependent input fields other than OBSVALUE
        self.inputs_int   = tf.keras.Input(shape=(nfields_int,),dtype="int32")
        self.inputs_pol0 = tf.keras.Input(shape=(nchannels,))
        self.inputs_pol1 = tf.keras.Input(shape=(nchannels,))
        self.inputs = [self.inputs_float, self.inputs_int, self.inputs_pol0, self.inputs_pol1]
                
        # Split the input tensor into named variables
        tsfc_norm = self.inputs_float[:,0]
        geolocation = tf.cast(self.inputs_float[:,1:3],tf.int32)
        tsfc = self.inputs_float[:,3]
        windspeed = self.inputs_float[:,4]
        cloud_fraction = self.inputs_float[:,5]
        iobs = self.inputs_int[:,0]
        isensor = tf.cast(self.inputs_float[:,6],tf.int32)
        #scanpos = self.inputs_float[:,7] #This is not int32 to allow for NaN values in scanpos
        zenith = self.inputs_float[:,8]
        pol0 = self.inputs_pol0
        pol1 = self.inputs_pol1

        emis_ocean = self.inputs_float[:,nfields_float:nfields_float+nchannels]
        tausfc_clear = self.inputs_float[:,nfields_float+nchannels:nfields_float+2*nchannels]
        tdown_clear = self.inputs_float[:,nfields_float+2*nchannels:nfields_float+3*nchannels]
        tup_clear = self.inputs_float[:,nfields_float+3*nchannels:nfields_float+4*nchannels]
        tausfc_cloud = self.inputs_float[:,nfields_float+4*nchannels:nfields_float+5*nchannels]
        tdown_cloud = self.inputs_float[:,nfields_float+5*nchannels:nfields_float+6*nchannels]
        tup_cloud = self.inputs_float[:,nfields_float+6*nchannels:nfields_float+7*nchannels]

        # Define layer objects
        self.ice_prop_layer_grid = seaice_layers.IcePropertyGrid(nprop_grid, ngrid, nstep)
        self.ice_prop_layer_obs  = seaice_layers.IcePropertyObsSpace(nobs, nprop_obs)
        self.seaice_emis_layer = seaice_layers.SeaiceEmisNN(nchannels, width=width_nn,
          background=background_emis, emissivity_mapping=emissivity_mapping, nobs=nobs,
          npol=self.setup['npol'], bg_error=bg_error_emis, use_loss=emis_use_bounds_loss,
          loss_channel = loss_channel_emis)
        self.ocean_emis_layer = seaice_layers.OceanEmis(nchannels)
        self.seaice_layer = seaice_layers.SeaiceFraction(nchannels, ngrid, nstep,
          nlag, nobs, alpha=alpha, bg_error=bg_error_seaice, bg_error_false_sic=bg_error_false_sic,
          use_loss=seaice_use_loss, use_pdf_loss=seaice_use_pdf_loss,
          use_tsfc_loss=seaice_use_tsfc_loss, penalise_false_sic=penalise_false_sic)
        self.surface_terms_layer = seaice_layers.SurfaceRadiationTerms(nchannels)
        self.specular_clear_layer = seaice_layers.Specular()
        self.specular_cloud_layer = seaice_layers.Specular()
        self.bias_layer = seaice_layers.BiasCorrection(nchannels, nsensors=nsensors, nobs=nobs, 
            background=background_bias, bg_error=bg_error_bias)
            
        print('Emis bg error',self.seaice_emis_layer.bg_error)
        print('Emis background',self.seaice_emis_layer.background)
        print('Seaice fraction bg error',self.seaice_layer.bg_error)
        print('Seaice fraction alpha',self.seaice_layer.alpha)
        print('Seaice fraction nobs',self.seaice_layer.nobs)
        print('Use seaice background loss',self.seaice_layer.use_loss)
        print('Use seaice PDF loss',self.seaice_layer.use_pdf_loss)
        print('Use seaice Tsfc loss',self.seaice_layer.use_tsfc_loss)
        print('Bias correction bg error',self.bias_layer.bg_error)
                        
        # Plug together the model
        ice_prop_grid = self.ice_prop_layer_grid(geolocation)
        ice_prop_obs  = self.ice_prop_layer_obs(iobs)
        ice_prop = tf.concat([ice_prop_grid,ice_prop_obs],1)
        self.emis_seaice = self.seaice_emis_layer(tsfc_norm, ice_prop, isensor, pol0, pol1)
        emis_ocean_bc = self.ocean_emis_layer(windspeed, emis_ocean)
        seaice_fraction = self.seaice_layer(geolocation, tsfc)
        surface_emitted, emis_mixed = self.surface_terms_layer(seaice_fraction, emis_ocean_bc, self.emis_seaice, tsfc)
        clear_tb = self.specular_clear_layer(surface_emitted, emis_mixed, tausfc_clear, tdown_clear, tup_clear)
        cloudy_tb = self.specular_cloud_layer(surface_emitted, emis_mixed, tausfc_cloud, tdown_cloud, tup_cloud)
        allsky_tb = (     tf.tensordot(cloud_fraction, tf.ones((nchannels)),0)  * cloudy_tb + 
                     (1 - tf.tensordot(cloud_fraction, tf.ones((nchannels)),0)) * clear_tb)
        bias_corrected_tb = self.bias_layer(allsky_tb, seaice_fraction, isensor)
        augmented_y = tf.stack([bias_corrected_tb,bias_corrected_tb],axis=2)
        self.outputs = augmented_y  

    def save(self, history, fappend, outpath, callback):
        """
        Save the sea ice model details to disk
        """
        grid=np.arange(self.setup['ngrid'])
        step=np.arange(self.setup['nstep']) 
        presteps = -1 + (-1*np.arange(self.setup['nlag']))
        lagstep=np.concatenate([presteps, step])
        lag_julian_day=np.concatenate([(self.grid['julian_day'])[0]+self.grid['stepsize']*presteps, self.grid['julian_day']])
        channel=np.arange(self.setup['nchannels'])
        biases=np.arange(self.setup['nbiases'])
        sensors=np.arange(self.setup['nsensors'])
        prop_grid=np.arange(self.setup['nprop_grid'])
        prop_obs=np.arange(self.setup['nprop_obs'])
        out_pol=np.arange(self.setup['npol'])
        emis_inputs=np.arange(self.setup['nprop_grid']+self.setup['nprop_obs']+2)
        iobs=np.arange(self.setup['nobs'])
        epoch=np.arange(len(history.history['loss']))
        n_extra_layers = 0
        width=np.arange(self.setup['width_nn'])

        filename_append = fappend+'.nc'

        da1 = xr.Dataset()
        da1_coords_seaice = {"grid":grid,"lagstep":lagstep}
        da1['seaice'] = xr.DataArray(data=self.seaice_layer.seaice[:,:],dims=da1_coords_seaice.keys(),coords=da1_coords_seaice)
        da1['lon'] = xr.DataArray(data=self.grid['lon'],dims=("grid"))
        da1['lat'] = xr.DataArray(data=self.grid['lat'],dims=("grid"))
        da1['julian_day'] = xr.DataArray(data=lag_julian_day,dims=("lagstep"),attrs=julian_day_attrs)
        da1.to_netcdf(outpath+'seaice_'+filename_append)

        da2 = xr.Dataset()
        for key, value in self.setup.items():
            if key == 'alpha':
                da2_coords={"alphas":np.arange(len(value))}
            elif key == 'background_bias':
                da2_coords={"area":biases,"sensor":sensors}
            elif key == 'bg_error_bias':
                da2_coords={"sensor":sensors}
            else:
                da2_coords={"one":[1]}
            da2[key] = xr.DataArray(data=value,dims=da2_coords.keys(),coords=da2_coords)

        da2_coords_layer1 = {"in":emis_inputs,"width":width}
        da2['emis_layer1_weights'] = xr.DataArray(data=self.seaice_emis_layer.layers[0].get_weights()[0],dims=da2_coords_layer1.keys(),
            coords=da2_coords_layer1)
        da2['emis_layer1_biases'] = xr.DataArray(data=(self.seaice_emis_layer.layers[0].get_weights()[1])[:],dims=("width"),
            coords={"width":width})
        da2_coords_layer2 = {"width":width,"out":out_pol}
        da2['emis_layer2_weights'] = xr.DataArray(data=self.seaice_emis_layer.layers[1].get_weights()[0],dims=da2_coords_layer2.keys(),
            coords=da2_coords_layer2)
        da2['emis_layer2_biases'] = xr.DataArray(data=self.seaice_emis_layer.layers[1].get_weights()[1],dims=("out"),
            coords={"out":out_pol})

        print(self.seaice_emis_layer.weights)

        da2['ocean_emis_bias'] = xr.DataArray(data=(self.ocean_emis_layer.weights[0])[:],dims=("channel"),
                            coords={"channel":channel})
        da2['tb_bias'] = xr.DataArray(data=self.bias_layer.weights[0],dims=("biases","sensors","channel"),
                            coords={"channel":channel,"sensors":sensors,"biases":biases})
        # Save epoch-level losses
        for key, value in history.history.items():
            da2[key] = xr.DataArray(data=value,dims=("epoch"),coords={"epoch":epoch})
        
        # Save batch-level losses
        batch_losses = callback.batch_losses  # dict with lists per epoch
        # Nb of epochs 
        n_epochs = len(next(iter(batch_losses.values())))

        # Max number of batches in any epoch
        max_batches = max(len(epoch_list) for epoch_list in batch_losses['loss'])

        # Create a matrix epoch x batch for each loss type
        for key, epoch_lists in batch_losses.items():
            # Matrix to hold the loss values, initialized with NaNs
            loss_matrix = np.full((n_epochs, max_batches), np.nan)

            # Fill the matrix with actual loss values
            for e, batch_list in enumerate(epoch_lists):
                loss_matrix[e, :len(batch_list)] = batch_list

            # Save to xarray DataArray
            da2[key + "_per_batch"] = xr.DataArray(
                data=loss_matrix,
                dims=("epoch", "batch"),
                coords={"epoch": np.arange(n_epochs), "batch": np.arange(max_batches)}
            )

        da2.to_netcdf(outpath+'models_'+filename_append)

        da3=xr.Dataset()
        da3['properties'] = xr.DataArray(data=self.ice_prop_layer_grid.properties[:,:,:],dims=("grid","step","prop"),
            coords={"grid":grid,"step":step,"prop":prop_grid})
        da3['lon'] = xr.DataArray(data=self.grid['lon'],dims=("grid"))
        da3['lat'] = xr.DataArray(data=self.grid['lat'],dims=("grid"))
        da3['julian_day'] = xr.DataArray(data=self.grid['julian_day'],dims=("step"),attrs=julian_day_attrs)
        da3.to_netcdf(outpath+'properties_grid_'+filename_append)
     
        da4=xr.Dataset()
        da4['properties'] = xr.DataArray(data=self.ice_prop_layer_obs.properties[:,:],dims=("iobs","prop"),
            coords={"iobs":iobs,"prop":prop_obs})
        da4.to_netcdf(outpath+'properties_obs_'+filename_append)
        da4.close()
        
    def load(self, filename_append, outpath):
        """
        Initialise the network trainable weights from previously saved states
        """
        properties_grid = xr.open_dataset(outpath+'properties_grid_'+filename_append+'.nc')
        self.ice_prop_layer_grid.set_weights([properties_grid.properties])

        properties_obs = xr.open_dataset(outpath+'properties_obs_'+filename_append+'.nc')
        self.ice_prop_layer_obs.set_weights([properties_obs.properties])

        seaice = xr.open_dataset(outpath+'seaice_'+filename_append+'.nc')
        seaice_weights_list = self.seaice_layer.get_weights()
        seaice_weights_list[0] = seaice.seaice
        self.seaice_layer.set_weights(seaice_weights_list)

        models = xr.open_dataset(outpath+'models_'+filename_append+'.nc')
        self.seaice_emis_layer.set_weights([models.emis_layer1_weights,
          models.emis_layer1_biases,models.emis_layer2_weights,models.emis_layer2_biases])
        self.ocean_emis_layer.set_weights([models.ocean_emis_bias])
        self.bias_layer.set_weights([models.tb_bias])

    #def load_weights(self, filename_append, outpath):
    #    """
    #    Initialise the network trainable weights from previously saved states
    #    """
    #    models = xr.open_dataset(outpath+'models_'+filename_append+'.nc')
    #    self.seaice_emis_layer.set_weights([models.emis_layer1_weights,
    #        models.emis_layer1_biases,models.emis_layer2_weights,models.emis_layer2_biases])
    #    self.ocean_emis_layer.set_weights([models.ocean_emis_bias])
    #    self.bias_layer.set_weights([models.tb_bias])


    def initialize(self, ifs_seaice_file, ifs_tsfc_file):
        seaice_weights_list = [seaice_layers.seaice_initializer(self.seaice_layer.seaice.shape, ifs_seaice_file)]
        if self.seaice_layer.use_tsfc_loss:
            seaice_weights_list.append(seaice_layers.tsfc_initializer(self.seaice_layer.tsfc.shape, ifs_tsfc_file))
        self.seaice_layer.set_weights(seaice_weights_list)
        self.bias_layer.set_weights([self.bias_layer.background])

class DataGenerator(tf.keras.utils.Sequence):
    """
    A generator for each batch of data, to prevent the full data blowing up the GPU. Keras 2 approach (different at 3).
    """
    def __init__(self,distributor,isplit):
        self.distributor = distributor
        self.isplit = isplit

    def __len__(self):
        return self.distributor.nBatches(self.isplit)

    def __getitem__(self,batch_number):
        return self.distributor.getBatch(batch_number,self.isplit)

class TrainingDataDistributor():
    """
    An object hopefully outside the Tensorflow world and hence hopefully doesn't get copied (bad if ends on GPU!)
    The "nsplit" approach allows the training dataset to be further split up (sharded).
    (Developed because model.predict() has an apparent bug - Non-OK-status: GpuLaunchKernel - for larger datasets.)
    """
    def __init__(self,nobs,x,x_int,y,pol0,pol1,batch_size=1024,nsplit=1):
        self.batch_size = batch_size
        self.nobs = nobs
        self.x = x
        self.x_int = x_int
        self.y = y
        self.pol0 = pol0
        self.pol1 = pol1
        self.makeSplit(nsplit)

    def makeSplit(self,nsplit):
        self.nsplit = nsplit
        self.split_size = np.int64(np.ceil(np.int64(self.nobs)/self.nsplit))
        self.split_start = np.zeros((nsplit),dtype=np.int64)
        self.split_end = np.zeros((nsplit),dtype=np.int64)
        self.split_nobs = np.zeros((nsplit),dtype=np.int64)
        for isplit in range(nsplit):
            self.split_start[isplit] = np.int64(np.int64(isplit)*self.split_size)
            self.split_end[isplit]   = np.min([self.split_start[isplit]+self.split_size,self.nobs])
            self.split_nobs[isplit]  = self.split_end[isplit] - self.split_start[isplit]

    def nBatches(self,isplit):
        return np.int64(np.ceil(np.int64(self.split_nobs[isplit])/self.batch_size))

    def getBatch(self,batch_number,isplit):
        istart = np.int64(np.int64(batch_number)*self.batch_size + self.split_start[isplit])
        iend = np.min([istart+self.batch_size,self.split_end[isplit]])
        return [self.x[istart:iend,:], self.x_int[istart:iend,:],self.pol0[istart:iend,:],
            self.pol1[istart:iend,:]
        ] , self.y[istart:iend,:]

def training_data(icedir, outdir, fappend, sensors, channel_names, nsteps_per_day=1, restrict_steps_to=-1,
    restrict_nobs_to=-1, step_start=0, channel_basis=None, channel_maps=None):

    """
    Load training data for the sea ice network
    """

    fields_1d = ['JULIAN_DAY','','TSFC','WINDSPEED10M','CLOUD_FRACTION','SCANPOS','ZENITH']
    fields_chan = ['OBSVALUE','EMIS_WATER','TAUSFC','TDOWN','TUP','TAUSFC_CLD','TDOWN_CLD','TUP_CLD']  

    nstep_all = []
    nobs_all = []
    ngrid_all = []
    istep_all = []
    nchan_all = []
    tstart_all = []
  
    if len(sensors)>1:
        IGRID='COMMON_IGRID'
        fields_1d[1] = IGRID
    else:
        IGRID='IGRID'
        fields_1d[1] = IGRID

    # Build common basis to combine instruments
    for sensor in sensors:
        filebase = icedir+sensor+'_' #'/'+sensor+'_'

        julian_day = xr.open_dataset(filebase+fields_1d[0]+'.nc')
        istep = np.floor(nsteps_per_day*(julian_day.JULIAN_DAY.data - np.floor(julian_day.JULIAN_DAY.data.min()) - 0.375))
        istep = istep.astype(np.int32)-step_start
        nstep = istep.max() + 1
        nobs = julian_day.JULIAN_DAY.data.size
        # Mapping anything in the ECMWF long windows 21-09 UTC and 09-21 UTC to a starting 'day' at 00 UTC.
        tstart = np.floor(julian_day.JULIAN_DAY.data.min()-0.375)+0.5+step_start/nsteps_per_day
        julian_day.close()

        istep_all.append(istep)
        tstart_all.append(tstart)

        if restrict_steps_to > 0:
            # For debugging and trials, this provides reduced or expanded model size (and reduces training data to match)
            nstep = restrict_steps_to

        nobs = np.count_nonzero(np.logical_and(0 <= istep,istep < nstep))

        if restrict_nobs_to > 0:
            # This further restricts the number of observations used in the training for debugging and trials (multiplied by the number of sensors)
            nobs = np.min([restrict_nobs_to,nobs])

        nstep_all.append(nstep)
        nobs_all.append(nobs)
        print("    Number of obs:",nobs)

        igrid = xr.open_dataset(filebase+fields_1d[1]+'.nc')
        ngrid = len(np.unique(igrid[IGRID]))
        igrid.close()
        
        ngrid_all.append(ngrid)
                       
        obsvalue = xr.open_dataset(filebase+fields_chan[0]+'.nc')
        nchan = obsvalue.OBSVALUE.shape[1]
        obsvalue.close()
        
        nchan_all.append(nchan)

    # Shortcut to avoid specifying channel settings if just one sensor
    if channel_basis is None and len(sensors)==1:
        channel_basis=np.arange(nchan_all[0])
        channel_maps=[channel_basis]

    ngrid = np.max(ngrid_all)
    nstep = np.max(nstep_all)
    nobs_total = sum(nobs_all)
    nchannels=len(channel_basis)
    nfields_float = 9 #7
    nfields_int = 1
    ndata = nfields_float +(len(fields_chan) - 1)*nchannels

    print("Total nobs:",nobs_total)
    print("Number of inputs per obs:",ndata)
    print("Number of channels:",nchannels)

    x0 = np.zeros((nobs_total,ndata),np.float32)
    x0_int = np.zeros((nobs_total,nfields_int),np.int32)
    y0 = np.zeros((nobs_total,nchannels,2),np.float32)
    lon = np.zeros((nobs_total),np.float32)
    lat = np.zeros((nobs_total),np.float32)
    julian_day = np.zeros((nobs_total),np.float64)

    #grid = xr.open_dataset("/perm/dnk8355/odb_files_test/METOP-B_1april2024_31march2025_lat_lon_corrected_ref_above50_without_land.nc")
    #grid = xr.open_dataset(icedir+'grid.nc')


    noff = 0
    isensor=0
    for sensor, nobs, istep, channel_map in zip(sensors, nobs_all, istep_all, channel_maps):
        filebase = icedir+sensor+ '_' #'/'+sensor+'_'

        print("Loading",sensor)

        ibegin = np.count_nonzero(istep < 0)
        ilast  = np.count_nonzero(istep < nstep)


        obs = {}
        for field_name in fields_1d: 
            file_path = filebase + field_name + '.nc'
            if os.path.exists(file_path):
                obs[field_name] = xr.open_dataset(file_path)
            else:
                obs[field_name] = None  


        x0[noff:noff+nobs,0] = np.maximum(273.0 - obs["TSFC"].TSFC[ibegin:ilast],0.0)/30.0
        x0[noff:noff+nobs,1] = obs[IGRID].IGRID[ibegin:ilast]
        x0[noff:noff+nobs,2] = istep[ibegin:ilast]
        x0[noff:noff+nobs,3] = obs["TSFC"].TSFC[ibegin:ilast]
        x0[noff:noff+nobs,4] = obs["WINDSPEED10M"].WINDSPEED10M[ibegin:ilast]
        x0[noff:noff+nobs,5] = obs["CLOUD_FRACTION"].CLOUD_FRACTION[ibegin:ilast]
        x0_int[noff:noff+nobs,0] = noff+np.arange(nobs)
        x0[noff:noff+nobs,6] = isensor+np.zeros(nobs)

        # SCANPOS
        if obs['SCANPOS'] is not None:
            x0[noff:noff+nobs,7] = obs['SCANPOS'].SCANPOS[ibegin:ilast]
        else:
            x0[noff:noff+nobs,7] = np.nan
            print(f"Warning: SCANPOS file missing for {sensor}, filled with nans.")

        # ZENITH
        if obs['ZENITH'] is not None:
            x0[noff:noff+nobs,8] = obs['ZENITH'].ZENITH[ibegin:ilast]
        else:
            x0[noff:noff+nobs,8] = np.nan
            print(f"Warning: ZENITH file missing for {sensor}, filled with nans.")

        isensor += 1
        lon=xr.open_dataset(filebase+'NEAREST_LONS.nc').NEAREST_LONS.values
        lat=xr.open_dataset(filebase+'NEAREST_LATS.nc').NEAREST_LATS.values

        lon[noff:noff+nobs] = lon[ibegin:ilast]
        lat[noff:noff+nobs] = lat[ibegin:ilast]
        julian_day[noff:noff+nobs] = obs["JULIAN_DAY"].JULIAN_DAY[ibegin:ilast]
        
        for dataset in obs.values():
            if dataset is not None:
                dataset.close()

        doff = nfields_float
        for field in fields_chan[1:]:
            obs = xr.open_dataset(filebase+field+'.nc')
            x0[noff:noff+nobs,doff+channel_map] = (obs.data_vars[field])[ibegin:ilast,:]
            obs.close()
            doff += nchannels
        
        # Obsvalue (where channel present)
        obs = xr.open_dataset(filebase+fields_chan[0]+'.nc')
        y0[noff:noff+nobs,channel_map,0] = obs.OBSVALUE[ibegin:ilast,:]
        obs.close()

        # Channel mask
        y0[noff:noff+nobs,channel_map,1] = 1
              
        noff += nobs

    fixed_grid=xr.open_dataset('/home/dnk8355/perm/paper2026/grib_files_NH_SH/'+sensor+'_1april2024_31march2026_lat_lon_corrected_ref_above44_without_land.nc')
    fixed_grid.close()
    geolocation = {'lon':lon, 'lat':lat, 'julian_day':julian_day, 'mask':y0[:,:,1]}
    grid = {'lon':fixed_grid.lon.values, 'lat':fixed_grid.lat.values, 'julian_day':min(tstart_all)+(np.arange(nstep)/nsteps_per_day), 'stepsize':1.0/nsteps_per_day}

    print("Training data generated")

    # Summary of training data and grid-to-obs mappings for post-processing and diagnostic investigations
    da = xr.Dataset()
    tbout = np.copy(y0[:,:,0])
    tbout[np.where(y0[:,:,0] == 0)] = np.nan
    da['tb'] = xr.DataArray(data=tbout,dims=("iobs","channel"),coords={"iobs":x0_int[:,0],"channel":channel_basis})
    append_geolocation_in_obs_space(geolocation, da, julian_day_attrs)
    da['igrid'] = xr.DataArray(data=x0[:,1].astype(np.int32),dims=("iobs"))
    da['istep'] = xr.DataArray(data=x0[:,2].astype(np.int32),dims=("iobs"))
    da['isensor'] = xr.DataArray(data=x0[:,6].astype(np.int32),dims=("iobs"))
    da['channel_name'] = xr.DataArray(channel_names,dims=("channel"))
    datetime_fromjulianday = pd.to_datetime(geolocation['julian_day'],origin='julian',unit='D')
    da['date_time_fromjd'] = xr.DataArray(datetime_fromjulianday.values,dims='iobs',
    attrs={'standard_name': 'datetime','long_name': 'Datetime converted from astronomical Julian Day'})
    da.to_netcdf(outdir+'tbobs_'+fappend+'.nc')
    da.close()

    return nchannels, ngrid, nstep, nobs_total, nfields_float, nfields_int, x0, x0_int, y0, geolocation, grid


def compute_polarization_coeffs(x0, polarisation_maps, sensor_type, zswath_width, zfov_spacing):
    """
    Calculate pol0 y pol1 (nobs, nchannels) which are the mixing polarization coefficients
    """

    nobs_total = x0.shape[0]
    nsensors   = polarisation_maps.shape[0]
    nchannels  = polarisation_maps.shape[1]
    #polarisation_maps     shape (nsensors, nchannels)

    # OUTPUTS
    pol0 = np.zeros((nobs_total, nchannels), np.float32)
    pol1 = np.zeros((nobs_total, nchannels), np.float32)

    # INPUT DATA
    isensor_all = x0[:, 6].astype(int)
    scanpos_all = x0[:, 7]

    zswath = np.array(zswath_width)
    zfov   = np.array(zfov_spacing)

    # ---------------------------------------------------
    # Precompute cos² y sin² per observation
    # ---------------------------------------------------
    cos2_all = np.ones(nobs_total, np.float32)     # conical by default
    sin2_all = np.zeros(nobs_total, np.float32)

    for isens in range(nsensors):
        idx = np.where(isensor_all == isens)[0]
        if len(idx) == 0:
            continue

        if sensor_type[isens] == "cross-track":
            theta = (-zswath[isens]
                    + (scanpos_all[idx] - 1.0) * zfov[isens]) #Scan angle for cross track sensor
            th = np.deg2rad(theta)
            cos2_all[idx] = np.cos(th)**2
            sin2_all[idx] = np.sin(th)**2

    for ch in range(nchannels):

        # (nobs,2): [id_sensor, id_channel]
        index = np.column_stack([isensor_all, np.full(nobs_total, ch, dtype=int)])

        # nominal polarization : (nobs,)
        pol_nom = polarisation_maps[index[:,0], index[:,1]]

        # type of sensor per observation: (nobs,)
        stype_obs = np.array(sensor_type)[index[:,0]]

        # ----- conical -----
        mask_con = (stype_obs == "conical")

        # V
        mask_V = mask_con & (pol_nom == 0)
        pol0[mask_V, ch] = 1.0
        pol1[mask_V, ch] = 0.0

        # H
        mask_H = mask_con & (pol_nom == 1)
        pol0[mask_H, ch] = 0.0
        pol1[mask_H, ch] = 1.0

        # ----- cross-track -----
        mask_cross = (stype_obs == "cross-track")

        # QV → (cos², sin²)
        mask_QV = mask_cross & (pol_nom == 0)
        pol0[mask_QV, ch] = cos2_all[mask_QV]
        pol1[mask_QV, ch] = sin2_all[mask_QV]

        # QH → (sin², cos²)
        mask_QH = mask_cross & (pol_nom == 1)
        pol0[mask_QH, ch] = sin2_all[mask_QH]
        pol1[mask_QH, ch] = cos2_all[mask_QH]

    return pol0, pol1

def save_model_outputs(model_tb_list, geolocation, fname, channel_names, varname='tb', mask=True):

    tbout = np.concatenate(model_tb_list,axis=0)
    iobs=np.arange((tbout.shape)[0])
    channel=np.arange((tbout.shape)[1])

    if mask:
        tbout[np.where(geolocation['mask'] == 0)] = np.nan

    da = xr.Dataset()
    da[varname] = xr.DataArray(data=tbout,dims=("iobs","channel"),coords={"iobs":iobs,"channel":channel})
    #julian_day_attrs = {'units':'days since -4714-11-24 12:00:00.000','calendar':'proleptic_gregorian'}
    julian_day_attrs = {'long_name': 'Astronomical Julian Day',
            'standard_name': 'julian_day',
            'description': 'Continuous astronomical Julian Day starting at 4713 BCE-01-01 12:00 UTC'
        }
    append_geolocation_in_obs_space(geolocation, da, julian_day_attrs)
    datetime_fromjulianday = pd.to_datetime(geolocation['julian_day'],origin='julian',unit='D')
    da['date_time_fromjd'] = xr.DataArray(datetime_fromjulianday.values,dims='iobs',
    attrs={'standard_name': 'datetime','long_name': 'Datetime converted from astronomical Julian Day'})
    da['channel_name'] = xr.DataArray(channel_names,dims=("channel"))
    da.to_netcdf(fname)

def append_geolocation_in_obs_space(geolocation, dataset,julian_day_attrs):
    dataset['lon'] = xr.DataArray(data=geolocation['lon'],dims=("iobs"))
    dataset['lat'] = xr.DataArray(data=geolocation['lat'],dims=("iobs"))
    dataset['julian_day'] = xr.DataArray(data=geolocation['julian_day'],dims=("iobs"),attrs=julian_day_attrs)



