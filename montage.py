#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:06:24 2020

Script for createing montages

Ing.,Mgr. (MSc.) Jan Cimbálník
Biomedical engineering
International Clinical Research Center
St. Anne's University Hospital in Brno
Czech Republic
&
Mayo systems electrophysiology lab
Mayo Clinic
200 1st St SW
Rochester, MN
United States
"""

# Standard library imports

# Third party imports
import numpy as np

# Local imports
# import help_functions


   



def getLV(unipol, getHV='no'):
        """
        returns the channels with a below average variability
        getHV = 'yes' returns also the above average channels
        """
        
        
        
        ### ------- CALCULATE VARIANCES FOR ALL THE CHANNELS ------- ###
        v = list()
        for i in range(unipol.shape[1]):
            v.append(np.var(unipol[:,i]));
        
        
        ### ------- GET AVERAGE VARIANCE AND ADD IT TO THE LIST ------- ###
        vAverage = np.mean(v)  
        v.append(vAverage)      
        v.sort()
    
    
        ### ------- SELECT CHANNELS WITH BELOW AVERAGE VARIANCE ------- ###
        vList = list()
        HV = list()
        for i in range(unipol.shape[1]):
            if np.var(unipol[:,i])<vAverage:
                vList.append(i)
            else:
                HV.append(i)
    
        below_average = unipol[:,vList]
        above_average = unipol[:,HV]     
    
        
        ### ------- RETURN THE DATASET ------- ###
        if getHV == 'no':
            return below_average
        else:
            return below_average, above_average
        
        
def low_var_montage(data):
    """
    perform low variance montage
    
    Parameters
    ----------
    data :  numpy array of raw channel data

    Returns
    -------
    mData: numpy array of data with new montage

    """  
    
    ref_ch = getLV(data)
    rch = np.mean(ref_ch, axis=1)
    
    newMontage = data.T - rch
    
    return newMontage.T    
        
def channel_sort_list(channels):
    """
    Function to ad 0s to channels and sort them. Meant for FNUSA
    
    Parameters:
    -----------
    channels - list\n
    
    Returns:
    --------
    mod_df - modified and sorted dataframe\n
    """
    
    # Rename channels add 0s - for good ordering
    modified_channels = []
    for channel in channels:
        digits = [x for x in channel if x.isdigit()]
        if len(digits) == 1:
            dig_idx = channel.index(digits[0])
            mod_chan = channel[:dig_idx]+'0'+channel[dig_idx:]
            modified_channels.append(mod_chan)
        else:
            modified_channels.append(channel)
            
    modified_channels.sort()
    
    for ci,channel in enumerate(modified_channels):
        digits = [x for x in channel if x.isdigit()]
        if not len(digits):
            continue
        if digits[0] == '0':
            dig_idx = channel.index(digits[0])
            modified_channels[ci] = modified_channels[ci][0:dig_idx] + modified_channels[ci][dig_idx+1:]

    return modified_channels
        
        
# Bipolar
def define_pairs(channels):
# =============================================================================
#     channels = ["B'1", "B'2", "B1", "B2", "Bo'1", "B'12"]
#     channels = ch_names
# =============================================================================
    
    channels = channel_sort_list(channels)
    
    channel_bases = []
    for channel in channels:
        channel_bases.append(''.join([x.strip() for x in channel if x.isalpha() or x=="'"]))
    
    channel_nums = []
    not_use_num = []
    for i, channel in enumerate(channels):
        num = ''.join([x.strip() for x in channel if x.isnumeric()])
        if num != '':
            channel_nums.append(int(''.join([x for x in channel if x.isnumeric()])))
        else:
            not_use_num.append(i)
            
    if len(not_use_num)>0: 
        ch_numbers = np.arange(len(channels))
        ch_numbers = [x for x in ch_numbers if x not in not_use_num]
        channel_bases = [channel_bases[k] for k in ch_numbers]   
        channels = [channels[k] for k in ch_numbers]  
    
    bipolar_pairs = []
    bipolar_names = []

    for i, ch in enumerate(channels[:-1]):
        channel_base = channel_bases[i]
        ch_num = channel_nums[i]
        if channel_bases[i+1] == channel_base and channel_nums[i+1] == ch_num+1:
            bipolar_pairs.append([ch, channels[i+1]])
            bipolar_names.append(ch + '_' + str(channel_nums[i+1]))
        
    # We have bipolar pairs - we can create monatges
    return bipolar_pairs, bipolar_names


def bipolar_montage(data, channels):
    
    if len(data.T) != len(channels):
       assert('Number of channel names must be equal to number of signals.')
       
    ch_dict = dict(zip(channels,np.arange(len(channels))))
    pairs, names = define_pairs(channels)
     
    bi_data = np.zeros([len(data), len(pairs)], dtype='float64')
     
    for i, pair in enumerate(pairs):
        bi_data[:,i] = data[:,ch_dict[pair[0]]] - data[:,ch_dict[pair[1]]]
         
    return bi_data, names
 