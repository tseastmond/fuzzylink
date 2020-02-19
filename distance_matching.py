#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Tanner S Eastmond
Date Updated: 2/18/2020
Purpose: This includes a few functions to match places based on geographic
    distance using latitude and longitude coordinates. These functions use
    numpy, pandas, scipy.spatial.distance.cdist, and time.time.

'''

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from time import time



def GetNClosest(data1, data2, indecies, latvars, lonvars, nummatches=10,
                chunksize=1000):
    '''
    Description
    -----------
    This function takes two data frames and returns a crosswalk between the
    indecies from 'data1' and the indecies for the specified number of closest
    matches from 'data2', in order of distance, least to most.


    Parameters
    ----------
    data1      - Pandas DataFrame - The first DataFrame.
    data2      - Pandas DataFrame - The second DataFrame.
    indecies   - list - A list with two entries specifying the columns of the
                 index variables for each DataFrame, which should be unique for
                 both DataFrames: ['index for data1', 'index for data2']
    latvars    - list - A list with two entries specifying the columns of the
                 latitude variables for each DataFrame, in order.
    lonvars    - list - A list with two entries specifying the columns of the
                 longitude variables for each DataFrame, in order.
    nummatches - int - The desired number of matches from 'data2' for each
                 index in 'data1'.
    chunksize  - int - The number of rows to do at a time from 'data1'. This
                 is specified because if you have many rows in both data sets,
                 the process can be extremely memory intensive. If you have the
                 necessary computing power, set chunksize to be the length of
                 'data1'.


    Returns
    -------
    A Pandas DataFrame with two columns, the first being the indecies from
    'data1' and the second being a list with the specified number of indecies
    for the closest places from 'data2'.
    '''
    # Start the timer.
    start_time = time()


    # Copy each data set.
    df1 = data1.copy().reset_index(drop=True)
    df2 = data2.copy().reset_index(drop=True)


    # Loop over increments of chunksize.
    for x in range(0,len(df1),chunksize):
        # Get the argument numbers for the closest matches.
        args = list(cdist(np.array(list(zip(df1.loc[x:x+chunksize, latvars[0]],df1.loc[x:x+chunksize, lonvars[0]]))),
                                    np.array(list(zip(df2[latvars[1]],df2[lonvars[1]])))).argpartition(
                                           range(nummatches),axis=1)[:,0:nummatches])


        # Make a DataFrame for the return values.
        df1.loc[x:x+chunksize, '__matches__'] = pd.Series(args)

        # Get the correct indecies from the original DataFrame 2.
        if nummatches == 1:
            df1.loc[x:x+chunksize, '__matches__'] = df1.loc[x:x+chunksize, '__matches__'].apply(lambda x: df2[indecies[1]].loc[int(x)])
        else:
            df1.loc[x:x+chunksize, '__matches__'] = df1.loc[x:x+chunksize, '__matches__'].apply(lambda x: [df2[indecies[1]].loc[y] for y in x])


        # Print the time.
        now = time()
        print('Progress:      ', round((x/len(df1))*100,2), '%')
        print('Time Remaining:', round((((now-start_time)/(x+chunksize)) * (len(df1) - min(x + chunksize, len(df1))))/3600,3), 'Hours')
        print(' ')


    # Keep only the two index columns and return.
    df1 = df1[[indecies[0], '__matches__']]
    if nummatches == 1:
        df1['__matches__'] = df1['__matches__'].astype(np.int64)
    else:
        df1['__matches__'] = df1['__matches__'].apply(lambda x: [int(y) for y in x])

    return df1
