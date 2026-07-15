#
# (C) Copyright 2025 ECMWF - https://www.ecmwf.int
#
# This software is licensed under the terms of the Apache Licence Version 2.0 which can be obtained 
# at https://www.apache.org/licenses/LICENSE-2.0
#
# In applying this licence, ECMWF does not waive the privileges and immunities granted to it by virtue of 
# its status as an intergovernmental organisation nor does it submit to any jurisdiction.

""" Describes the basic characteristics of microwave sensors used in the sea ice training """

import numpy as np

class SeaiceSensors():
    """
    Prepares sensor information for training, given a list of sensor names
    """

    # Name unifies and identifies basic channel properties (approx. frequency, polarisaton)
    all_channel_names = np.array(['1v','1h','6v','6h','7v','7h','10v','10h','19v','19h',
      '24v','24h','37v','37h','50v','50h','53v','53h','89v','89h','166v','166h',
      '183pm7v','183pm7h','183pm3v','183pm3h','183pm1v','183pm1h'])

    # Frequency, GHz, of most standard channels
    all_channel_frequencies = np.array([1.4135, 1.4135, 6.925, 6.925, 7.3, 7.3, 10.65, 10.65, 
      18.7, 18.7, 23.8, 23.8, 36.5, 36.5, 50.3, 50.3, 52.8, 52.8, 89.0, 89.0, 166.5, 166.5, 
      183.31, 183.31, 183.31, 183.31, 183.31, 183.31], dtype=np.float32)

    all_channel_obs_error = np.array([2.5, 4.0, 2.5, 4.0, 2.5, 4.0, 2.5, 4.0, 2.5, 4.5, 
      2.5, 5.0, 4.0, 7.0, 5, 5, 2, 2, 4.5, 10.0, 12.0, 12.0, 12.0, 8.0, 8.0, 4.0, 4.0, 2.0], dtype=np.float32)

    def __init__(self, sensors):

        self.all_sensors = {}
        self.init_sensors()

        # 0=v; 1=h: horizontal and vertical polarisations in the local earth surface reference frame
        self.all_channel_polarisation = np.array([0.0,1.0]*(len(self.all_channel_names) // 2), dtype=np.float32)

        used_channels = []
        self.sensors = []      
        for key in sensors:
            if key in self.all_sensors:
                used_channels += self.all_sensors[key]['channel']
                print('Using sensor',key,self.all_sensors[key]['channel'])
                self.sensors.append(key)  
            else:
                print(f"Sensor '{key}' not in sensor definition class")

        unique_channels = np.unique(used_channels)
        iUsedChannels = np.where(np.isin(self.all_channel_names, unique_channels))[0]
                   
        # 'Output' of this class 
        self.channel_names = self.all_channel_names[iUsedChannels]
        self.channel_maps = []
        self.frequency_maps = []
        self.polarisation_maps = []
        self.background_bias = []
        self.background_bias_error = []
        self.zswath_width = []
        self.zfov_spacing = []
        self.sensor_type = []
        print('Unified channel basis:', self.channel_names)
           
        frequency_basis = self.all_channel_frequencies[iUsedChannels]
        for key in self.sensors:
            iChannelMap = []
            for channel in self.all_sensors[key]['channel']:
                iChannelMap.append(np.where(self.channel_names == channel)[0][0])
            iChannelMap = np.array(iChannelMap)
            print(key,'channel map:',iChannelMap)
            self.channel_maps.append(iChannelMap)

            frequency_actual = frequency_basis.copy()
            if 'frequency' in self.all_sensors[key]:
                frequency_actual[iChannelMap] = self.all_sensors[key]['frequency']
            self.frequency_maps.append(frequency_actual)
            
            self.polarisation_maps.append(self.all_channel_polarisation[iUsedChannels])
            
            self.background_bias.append(self.all_sensors[key]['background_bias'])
 
            self.background_bias_error.append(self.all_sensors[key]['background_bias_error'])

            self.zswath_width.append(self.all_sensors[key]['zswath_width'])

            self.zfov_spacing.append(self.all_sensors[key]['zfov_spacing'])
            
            self.sensor_type.append(self.all_sensors[key]['type_sensor'])

        self.frequency_maps = np.stack(self.frequency_maps)
        self.polarisation_maps = np.stack(self.polarisation_maps)
        self.background_bias = np.transpose(self.background_bias)
        
        self.obs_error = self.all_channel_obs_error[iUsedChannels]

        # --- Adjust obs_error for AMSU-A family sensors if present ---
        amsua_sensors = ['METOP-B', 'METOP-C', 'NOAA-15', 'NOAA-18', 'NOAA-19']

        # Obs errors for AMSU-A channels original values
        #amsua_obs_errors = {
        #    '24v': 2.5,
        #    '37v': 3.5, 
        #    '50v': 8.5, 
        #    '53v': 8.0, 
        #    '89v': 4.5  
        #}

        # Obs errors for AMSU-A channels new values
        #amsua_obs_errors = {
        #    '24v': 4.5, 
        #    '37v': 5.0, 
        #    '50v': 5.0, 
        #    '53v': 2.0, 
        #    '89v': 4.5  
        #}
        #Adjusted obs errors
        amsua_obs_errors = {
            '24v': 5, 
            '37v': 5, 
            '50v': 3.5, 
            '53v': 2, 
            '89v': 5.5  
        }
        # If any of the specified sensors are in the amsua list, update the obs_error accordingly
        if any(s in self.sensors for s in amsua_sensors):
            for ch_name, new_err in amsua_obs_errors.items():
                # Only update if the channel is in the unified channel list
                if ch_name in self.channel_names:
                    idx = np.where(self.channel_names == ch_name)[0][0]
                    self.obs_error[idx] = new_err
        print("Final unified obs_error:", self.obs_error)

        self.channel_basis = np.arange(np.max(np.concatenate(self.channel_maps))+1)
        
    def init_sensors(self):
        self.all_sensors['smap_allsky'] = {
          'channel': ['1v','1h'],
          'background_bias': [0.0,0.0],
          'background_bias_error': 2.0,
          'zswath_width':np.nan,
          'zfov_spacing':np.nan,
          'type_sensor': 'conical'}

        self.all_sensors['amsr2'] = {
          'channel': ['10v','10h','19v','19h','24v','24h','37v','37h','89v','89h'],
          'background_bias': [5.0,2.5],
          'background_bias_error': 0.5,
          'zswath_width':75,
          'zfov_spacing':0.620,
          'type_sensor': 'conical'}

        # SSMIS needs to over-ride the default frequencies, being an older non-standard sensor
        self.all_sensors['ssmisf17'] = {
          'channel': ['19v','19h','24v','37v','37h','89v','89h','166h','183pm7h','183pm3h','183pm1h'],
          'frequency': [19.35, 19.35, 22.235, 37.0, 37.0, 91.655, 91.655, 150.0, 183.31, 183.31, 183.31],
          'background_bias': [0.0,0.0],
          'background_bias_error': 4.0,
          'zswath_width':72,
          'zfov_spacing':2.441,
          'type_sensor': 'conical'}

        # GMI, noting that input files were unwittingly created with a different ordering of 183 GHz channels
        self.all_sensors['gmi'] = {
          'channel': ['10v','10h','19v','19h','24v','37v','37h','89v','89h','166v','166h','183pm3v','183pm7v'],
          'background_bias': [0.0,0.0],
          'background_bias_error': 0.001,
          'zswath_width':70,
          'zfov_spacing':0.633,
          'type_sensor': 'conical'}
        
        # amsu-a onboard METOP-B
        self.all_sensors['METOP-B'] = {
          'channel': ['24v','37v','50v','53v'],
          'frequency': [23.8, 31.4, 50.3, 52.8],
          'background_bias': [0,0], #[5,2.5] [0,0]
          'background_bias_error': 1.0, #0.001 1.0
          'zswath_width':48.333333,
          'zfov_spacing': 3.333333,
          'type_sensor':'conical' } #'conical' 'cross-track'
        
        # amsu-a onboard METOP-C
        self.all_sensors['METOP-C'] = {
          'channel': ['24v','37v','50v','53v','89v'],
          'frequency': [23.8, 31.4, 50.3, 52.8, 89],
          'background_bias': [0,0],
          'background_bias_error': 1.0,
          'zswath_width':48.333333,
          'zfov_spacing': 3.333333,
          'type_sensor': 'cross-track'}
        
        # amsu-a onboard NOAA15
        self.all_sensors['NOAA-15'] = {
          'channel': ['24v','37v','50v','53v','89v'],
          'frequency': [23.8, 31.4, 50.3, 52.8, 89],
          'background_bias': [0.0,0.0],
          'background_bias_error': 1.0,
          'zswath_width':48.333333,
          'zfov_spacing': 3.333333,
          'type_sensor': 'cross-track'}

        # amsu-a onboard NOAA18
        self.all_sensors['NOAA-18'] = {
          'channel': ['24v','37v','50v','53v','89v'],
          'frequency': [23.8, 31.4, 50.3, 52.8, 89],
          'background_bias': [0.0,0.0],
          'background_bias_error': 1.0,
          'zswath_width':48.333333,
          'zfov_spacing': 3.333333,
          'type_sensor': 'cross-track'}
        
        # amsu-a onboard NOAA19
        self.all_sensors['NOAA-19'] = {
          'channel': ['24v','37v','50v','53v','89v'],
          'frequency': [23.8, 31.4, 50.3, 52.8, 89],
          'background_bias': [0.0,0.0],
          'background_bias_error': 1.0,
          'zswath_width':48.333333,
          'zfov_spacing': 3.333333,
          'type_sensor': 'cross-track'}
        