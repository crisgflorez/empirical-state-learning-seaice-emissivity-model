#
# (C) Copyright 2025 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" Contains the physical layers and observation loss function for the sea ice Bayesian network """

import xarray as xr
import tensorflow as tf
import numpy as np
obs_error = None

class IcePropertyGrid(tf.keras.layers.Layer):
    """
    Empirical properties of sea ice, representing ice and snow microstructure and other physical influences on 
    the surface emissivity. The layer weights contain maps of these properties as a function of grid point and
    timestep. Calling this layer takes the geolocation as input and returns the empirical properties at that 
    location in time and space. 
       Geolocation (0) is the igrid, (1) the istep
    """
    def __init__(self, nprop=1, ngrid=1, nstep=31):
        super(IcePropertyGrid, self).__init__()
        self.properties = self.add_weight(shape=(ngrid,nstep,nprop), initializer="zeros",  trainable=True)
    def call(self, geolocation):
        return tf.gather_nd(self.properties,geolocation)

class IcePropertyObsSpace(tf.keras.layers.Layer):
    """
    Empirical ice properties in observation space (e.g. to represent intra daily variations better than on the daily grid).  
    """
    def __init__(self, nobs, nprop=1):
        super(IcePropertyObsSpace, self).__init__()
        self.properties = self.add_weight(shape=(nobs,nprop), initializer=tf.zeros, trainable=True)
    def call(self,iobs):
        return tf.gather(self.properties,(iobs))


class SeaiceEmisNN(tf.keras.layers.Layer):
    """
    Sea ice emissivity empirical model - dense multi-layer neural network version
    """
    def __init__(self, channels=10, width=7, nobs=1, npol=2, activation='sigmoid', bg_error=1e-5, background=0.8,
      use_loss=False, emissivity_mapping=None, loss_channel=0):
        super(SeaiceEmisNN, self).__init__()
        self.layers = list()
        self.layers.append(tf.keras.layers.Dense(width,activation=activation))
        self.layers.append(tf.keras.layers.Dense(npol,activation=activation))
        self.frequency_mapping=emissivity_mapping[0]
        self.polarisation_mapping=emissivity_mapping[1]
        self.bg_error = bg_error
        self.background = background
        self.use_loss = use_loss
        self.nobs = nobs
        self.channels = channels
        self.loss_channel = loss_channel
    def call(self, tsfc, ice_properties, isensor, scanpos, zswath_width, zfov_spacing):
        scan_angle = (-tf.gather(tf.cast(zswath_width, tf.float32), isensor)
            + (tf.cast(scanpos, tf.float32) - 1.0)
            * tf.gather(tf.cast(zfov_spacing, tf.float32), isensor)
        ) #If zswath_width, zfov_spacing or scanpos is NaN, scan_angle will be NaN
        
        #When scan_angle is NaN, set it to 0.0
        #This happens when the sensor is a conical scanner without a scan angle or when we have missing data of zswath_width or zfov_spacing 
        scan_angle = tf.where(tf.math.is_nan(scan_angle), tf.zeros_like(scan_angle, dtype=tf.float32), scan_angle)

        scan_angle_rad = scan_angle * (np.pi / 180.0)
        inputs = tf.concat([tf.reshape(tsfc,(-1,1)),ice_properties],1)
        out = []
        for i in range(self.channels):
            index = tf.concat([tf.reshape(isensor,(-1,1)),tf.reshape(i+tf.zeros_like(isensor),(-1,1))],1)
            freq = tf.reshape(tf.gather_nd(self.frequency_mapping,   index),(-1,1))/100.0
            pol  = tf.reshape(tf.gather_nd(self.polarisation_mapping,index),(-1,1))
            inputs_freq = tf.concat([inputs,freq],1)
            mid1 = self.layers[0](inputs_freq)
            pol_pair = self.layers[1](mid1)
            # Polarisation mixing (add cos^2 scan angle for cross-track QV/QH version eventually)
            pol_pair_corr_v = pol_pair[:,0]*tf.cos(scan_angle_rad)**2 + pol_pair[:,1]*tf.sin(scan_angle_rad)**2
            pol_pair_corr_h = pol_pair[:,0]*tf.sin(scan_angle_rad)**2 + pol_pair[:,1]*tf.cos(scan_angle_rad)**2

            one_channel = (1-pol[:,0])*pol_pair_corr_v + pol[:,0]*pol_pair_corr_h
            out.append(one_channel)
        out = tf.stack(out,1)
        if self.use_loss:
            emis_loss = tf.reduce_sum(tf.math.square(
              tf.math.maximum(-1*(out[:,self.loss_channel]-self.background),0.0)))/tf.square(self.bg_error)/self.nobs
            self.add_loss(emis_loss)
            self.add_metric(emis_loss,name='emis_loss',aggregation='mean')
        return out


def seaice_initializer(shape, ifs_seaice_file, dtype=tf.float32):
    """
    Initializer for the sea ice concentration maps
    """
    ifs_seaice = xr.open_dataset(ifs_seaice_file)
    if hasattr(ifs_seaice,'SEAICE'):
        seaice_map = tf.convert_to_tensor(ifs_seaice.SEAICE, dtype=dtype)
    if hasattr(ifs_seaice,'seaice'):
        seaice_map = tf.convert_to_tensor(ifs_seaice.seaice, dtype=dtype)
    ifs_seaice.close()
    if tf.rank(seaice_map) == 2:
        map_shape = seaice_map.shape
        if map_shape == shape:
            return seaice_map
        elif map_shape[1] < shape[1] and map_shape[0] == shape[0]:
            # Add any required padding when sea ice fields are being lagged
            return tf.concat([tf.repeat(seaice_map[:,0:1],shape[1] - map_shape[1],1),seaice_map],1)
        else:
            # Assume we have are using a shorter debug/develop training set than in the file (only
            # works correctly with the same sea ice lag in the file as selected for the training)
            return seaice_map[:,0:shape[1]]
    else: 
        return tf.repeat(tf.reshape(seaice_map,(shape[0],1)),shape[1],1)


def tsfc_initializer(shape, ifs_tsfc_file, dtype=tf.float32):
    """
    Initializer for the ocean surface temperature used in a sea ice fraction loss term
    """
    ifs_tsfc = xr.open_dataset(ifs_tsfc_file)
    tsfc_map = tf.convert_to_tensor(ifs_tsfc.TSFC, dtype=dtype)
    ifs_tsfc.close()
    map_shape = tsfc_map.shape
    if shape[1] <= map_shape[1]:
        return tsfc_map[:,0:shape[1]]
    else:
        # Pad out end of timeseries with last day for purely debugging / memory testing
        tsfc_padding = tf.repeat(tsfc_map[:,-1:],shape[1] - map_shape[1],1)
        padded_initial = tf.concat([tsfc_map,tsfc_padding],1)
        return padded_initial 


class OceanEmis(tf.keras.layers.Layer):
    """
    Ocean emissivity layer including a windspeed bias correction. Input vector is same as inputs (e.g. windspeed = 4)
    """
    def __init__(self, channels=10):
        super(OceanEmis, self).__init__()
        self.windspeed_bias = self.add_weight(shape=(channels), initializer="zeros", trainable=True)
        self.nchans=channels
    def call(self, windspeed, ocean_emis):
        return tf.tensordot(windspeed,self.windspeed_bias,0) + ocean_emis


class SeaiceFraction(tf.keras.layers.Layer):
    """
    Layer encapsulating the sea ice fraction maps
    
        Inputs to the call method are geolocation (igrid, istep). Returned is the sea ice fraction.
        
        nlag is number of lagged sea-ice timesteps weighted by alpha, with 0=current time and N being prior times
        nlag = 0 means that sea-ice is not smoothed
    
    """
    def __init__(self, channels=10, ngrid=1, nstep=31, nlag=0, nobs=1, alpha=[1.0], bg_error=0.002, bg_error_false_sic=0.02,
                 train=True, use_loss=False, use_pdf_loss=True, use_tsfc_loss=True, penalise_false_sic=True):
        super(SeaiceFraction, self).__init__()
        self.seaice = self.add_weight(shape=(ngrid,nstep+nlag), initializer=tf.zeros, trainable=train)
        if use_loss:    
            self.seaice_background = self.add_weight(shape=(ngrid,nstep+nlag), initializer=seaice_initializer, trainable=False) 
        if use_tsfc_loss:  
            self.tsfc = self.add_weight(shape=(ngrid,nstep), initializer=tf.zeros, trainable=False)
        self.nchans = channels
        self.nlag = nlag
        self.nobs = nobs
        self.alpha = alpha
        self.bg_error = bg_error
        self.use_loss = use_loss
        self.use_pdf_loss = use_pdf_loss
        self.use_tsfc_loss = use_tsfc_loss
        self.penalise_false_sic = penalise_false_sic
        self.min_trace = 0.0
        self.max_trace = 0.3
        self.bg_error_false_sic = bg_error_false_sic
        self.peak_scaling = 0.2
    def call(self, geolocation, tsfc, training=None):
        seaice_at_obs = self.alpha[0]*tf.gather_nd(self.seaice,tf.add(geolocation,[[0,self.nlag]]))
        for i in range(1,self.nlag+1):
            seaice_at_obs = seaice_at_obs + self.alpha[i]*tf.gather_nd(self.seaice,tf.add(geolocation,[[0,self.nlag-i]]))
        if self.use_loss and training:
            seaice_loss = tf.reduce_sum(tf.math.squared_difference(self.seaice,self.seaice_background)/(tf.square(self.bg_error)))/self.nobs
            self.add_loss(seaice_loss) 
            self.add_metric(seaice_loss,name='seaice_loss',aggregation='mean')
        if self.use_pdf_loss and training:
            seaice_loss = tf.reduce_sum(tf.math.square(tf.math.maximum(self.seaice,1.0)-1.0)+tf.math.square(tf.math.maximum(-1*self.seaice,0.0)))/tf.square(self.bg_error)
            if self.penalise_false_sic:
                halfway_trace = self.min_trace+(self.max_trace-self.min_trace)/2.0
                halfwidth_trace = halfway_trace - self.min_trace
                bend = self.peak_scaling * halfwidth_trace / (1.0+self.peak_scaling)
                height = 1.0/tf.square(self.bg_error_false_sic) * halfwidth_trace * bend
                seaice_loss += tf.reduce_sum(tf.where(tf.math.logical_and(self.seaice > self.min_trace, self.seaice <= self.min_trace + bend),
                  tf.math.square(self.seaice-self.min_trace)/tf.square(self.bg_error_false_sic),0.0))
                seaice_loss += tf.reduce_sum(tf.where(tf.math.logical_and(self.seaice > self.min_trace + bend, self.seaice < self.max_trace - bend),
                  height - self.peak_scaling/tf.square(self.bg_error_false_sic) * tf.math.square(halfway_trace-self.seaice),0.0))
                seaice_loss += tf.reduce_sum(tf.where(tf.math.logical_and(self.seaice >= self.max_trace - bend, self.seaice < self.max_trace),
                  tf.math.square(self.max_trace-self.seaice)/tf.square(self.bg_error_false_sic),0.0))
            seaice_loss = seaice_loss/self.nobs
            self.add_loss(seaice_loss) 
            self.add_metric(seaice_loss,name='seaice_loss',aggregation='mean')
        if self.use_tsfc_loss and training:
            tsfc_loss = (tf.reduce_mean(tf.where(self.seaice[:,self.nlag:] > 0.01,1.0,0.0)*tf.math.maximum(self.tsfc-273.2,0.0)*4.0))/self.nobs
            self.add_loss(tsfc_loss) 
            self.add_metric(tsfc_loss,name='tsfc_loss',aggregation='mean')
        return seaice_at_obs
    def update_loss(self, bg_error_false_sic=0.2):
        self.bg_error_false_sic = bg_error_false_sic

class SurfaceRadiationTerms(tf.keras.layers.Layer):
    """
    Mixed surface emitted radiation and mixed surface emissivity
    """
    def __init__(self, channels=10, min_temp_seawater=271.35):
        super(SurfaceRadiationTerms, self).__init__()
        self.nchans=channels
        self.min_temp_seawater=min_temp_seawater
    def call(self,sic,emis_ocean,emis_seaice,tsfc):
        tsfc_ice = tf.tensordot(tsfc,tf.ones((self.nchans)),0)
        tsfc_ocean = tf.maximum(tsfc_ice,self.min_temp_seawater)
        sic_chan = tf.tensordot(sic,tf.ones((self.nchans)),0)
        surface_emitted_radiation = sic_chan * emis_seaice * tsfc_ice + (1-sic_chan) * emis_ocean * tsfc_ocean
        mixed_emis = sic_chan * emis_seaice + (1-sic_chan) * emis_ocean
        return surface_emitted_radiation, mixed_emis


class Specular(tf.keras.layers.Layer):
    """
    Simple radiative transfer model (specular).
    """
    def __init__(self):
        super(Specular, self).__init__()
    def call(self,tsfcrad,emis,tausfc,tdown,tup):
        # Emitted, reflected and atmospheric emission term
        return tsfcrad*tausfc + (1-emis)*tausfc*tdown + tup


class BiasCorrection(tf.keras.layers.Layer):
    """
    Bias correction for TB output - with seaice fraction and ocean fraction as the predictors
    Defaults are for AMSR2, SSMIS F17 and GMI (which is treated as an anchor) with an ocean and sea ice bias zone
    """
    def __init__(self, channels=18, nsensors=3, nbiases=2, background=[[5.0,0.0,0.0],[2.5,0.0,0.0]], nobs=1, bg_error=[0.5,4.0,0.001]):
        super(BiasCorrection, self).__init__()
        self.background_basis = tf.constant(background, dtype=tf.float32)
        self.background = tf.tensordot(self.background_basis,tf.ones((channels)),0)
        self.nchans = channels
        self.nsensors = nsensors
        self.nobs = nobs
        self.bg_error = bg_error
        self.instrument_bias = self.add_weight(shape=(nbiases,nsensors,channels), initializer=tf.zeros, trainable=True)
    def call(self, tb, seaice_fraction, isensor):
        bias = (tf.tensordot(    seaice_fraction,tf.ones((self.nchans)),0)*tf.gather_nd(self.instrument_bias[0,:,:],tf.reshape(isensor,(-1,1)))
             +  tf.tensordot(1 - seaice_fraction,tf.ones((self.nchans)),0)*tf.gather_nd(self.instrument_bias[1,:,:],tf.reshape(isensor,(-1,1))))
        bias_loss = tf.reduce_sum((tf.math.squared_difference(self.instrument_bias,self.background))/
                                  tf.tensordot(tf.square(self.bg_error),tf.ones((self.nchans)),0))/self.nobs
        self.add_loss(bias_loss) 
        self.add_metric(bias_loss,name='bias_loss',aggregation='mean')
        return tf.math.add(tb,bias)


@tf.function
def loss_channel_weighted(y_true, y_pred):
    """
    Loss equivalent to the data assimilation observation term, weighted as a function of observation error.
    Returned loss is a vector over the batch. The TF printed loss is a running average. The total TF 
    loss is sum(all obs&batches)/Nobs   
    Note that the second slice of y is a data mask indicating where the observations are valid.
    """   
    normdep = tf.math.divide(y_true[:,:,1]*(y_true[:,:,0] - y_pred[:,:,0]),obs_error)
    obs_loss = tf.reduce_sum(tf.square(normdep), axis=-1)
    return obs_loss 

