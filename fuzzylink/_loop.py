#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 15:19:36 2020

@author: tanner
"""

import numpy as np

from ._rowfilter import _rowfilter


###############################################################################
###############################################################################
###############################################################################
# Make a function to loop over unique blocks.
def _loop(full, vals, idcols, procnum, output, progress, nomismatch, fuzzy,
          onlycheck, strthresh, numthresh, weight, allowmiss, nummatches,
          dup=False):
    '''
    A function to loop over exact match blocks and compile matches.
    

    Parameters
    ----------
    full : Pandas DataFrame
        This contains all of the data for matching.
    vals : list
        This includes all unique values of the "__exact__" column in the full
        dataframe that are assigned to the process running the function.
    idcols : list
        A list with two values, the id variable for the rows to be matched
        first and the id variable for the comparison pool second.
    procnum : int
        The number of the process running the function.
    output : dict
        A dictionary where the output from each process will be saved.
    progress : dict
        A dictionary used to report the progress from each process.
    nomismatch : list
        A list of columns requiring no discrepancy in nonmissing values, but
        it will allow a match between missing and a value.
    fuzzy : list
        A list of columns requiring a fuzzy match, with the threshold 
        specified by strthresh or numthresh, depending on the column type.
    onlycheck : str
        The name of a column with only True and False, where True signifies
        that the row is to be matched against all other observations and False
        signifies that a row is to be including in the comparison pool for
        other matches but not matched itself. If a column name is specified
        for this variable then there cannot be any cases in the matched sample 
        where only rows with False specified are matched together since those 
        rows will not be explicitly checked against the comparison pool.
    strthresh : float or dict
        The threshold for Jaro-Winkler score below which the rows are not a 
        match. If a dictionary is passed, it must have one entry per fuzzy 
        column specifying the threshold for each.
    numthresh : float
        The threshold for numeric variables absolute difference outside of 
        which the rows are not a match.
    weight : float or dict
        The weight for each column to be applied in calculating the score.       
    allowmiss : bool
        Allow a mismatch in fuzzy due to missing values.
    nummatches : int or None
        The number of matches to be returned, or None if all are to be 
        returned. In particular, the code will return either all matches or the
        nummatches best matches.
    dup : bool
        An indicator that is true if we are running DeDup, false if we are
        running Match.
                 
    
    Returns
    -------
    None.
    '''
    '''
    vals=splitvals[0]
    procnum=1
    output=dict()
    progress=dict()
    nomismatch=[]
    strthresh=0.97
    numthresh=1
    weight=1 
    allowmiss=False
    nummatches=None
    '''
    # Get the proper number of matches to be made.
    if onlycheck == '':
        fulllen = len(np.where(full['__exact__'].isin(vals))[0])
    else:
        fulllen = len(np.where((full[onlycheck] == True) 
                               & (full['__exact__'].isin(vals)))[0])

    
    # Set initial values.
    progress['tot'+str(procnum)] = fulllen
    progress['p'+str(procnum)] = 0
    progress['m'+str(procnum)] = 0
    
    matched = dict()
    arr = dict()
    
    
    # Get a list of relevant columns.
    if onlycheck == '':
        onlycheckcol = []
    else:
        onlycheckcol = [onlycheck]
    
    cols = ['__exact__'] + nomismatch + fuzzy + onlycheckcol + idcols 
    cols = list(set(cols))


    # Loop over unique values of exact columns.
    for val in vals:
        # First set up a dictionary of numpy arrays within the matching block.
        f = np.where(full.__exact__ == val)
        
        for col in cols:
            arr[col] = full[col].values[f]
            
        l = len(arr['__exact__'])
            
        
        # Skip if there are no values to be matched or only 1 value in the
        #  block.
        if onlycheck != '':
            if len(np.where(arr[onlycheck] == False)[0]) == 0:
                progress['p'+str(procnum)] += l
                continue
        
        if l == 1:
            progress['p'+str(procnum)] += 1
            continue
        
        
        # Define the vecorized row filter function and the structure of the 
        #  output.
        funcProxy = np.vectorize(_rowfilter, otypes=[dict])
        
        dtype = [('score', float), (idcols[0], full[idcols[0]].dtype),
                 (idcols[1], full[idcols[1]].dtype)]
        
        
        # Run the vectorized row filter function over each cell in the nxn
        #  block (i.e. check every value against every other value using the
        #  row filter function), redefine the array, and sort each row of that
        #  array to return the correct number of matches.
        if nummatches is not None:
            matchedvals = np.partition(
                np.array(
                    np.fromfunction(lambda i,j: funcProxy(i, 
                                         j, arr, 
                                         dict({'all': idcols}),
                                         dict({'all': nomismatch}),
                                         dict({'all': fuzzy}),
                                         strthresh, numthresh,
                                         onlycheck, weight, allowmiss), 
                                         (l, l), dtype=int)
                    , dtype=dtype), 
                -nummatches, axis=1, order='score')[:, -nummatches:]
        else:
            matchedvals = np.sort(
                np.array(
                    np.fromfunction(lambda i,j: funcProxy(i, 
                                         j, arr, 
                                         dict({'all': idcols}),
                                         dict({'all': nomismatch}),
                                         dict({'all': fuzzy}),
                                         strthresh, numthresh,
                                         onlycheck, weight, allowmiss), 
                                         (l, l), dtype=int)
                    , dtype=dtype), 
                axis=1, order='score') 
            
        
        # Collect each of the matches with a positive score into a dictionary.
        new = dict()
        for _, y, z in matchedvals[np.where(matchedvals['score'] > 0)]:
            if y not in new.keys():
                new[y] = set([z])
            else:
                new[y] = new[y] | set([z])
        
        
        # Update the number of matches and add those that did not match to the
        #  output dictionary.
        progress['m'+str(procnum)] += len(new)
        
        if dup:
            for x in np.unique(matchedvals[idcols[0]]):
                if x not in new.keys():
                    new[x] = set()
        
        if onlycheck != '':
            progress['p'+str(procnum)] += len(np.where(arr[onlycheck] == True)[0])
        else:
            progress['p'+str(procnum)] += len(arr['__exact__'])
            
            
        if dup:
            # Ensure that each matched group contains all members, e.g. if 1 matches
            #  to 2 and 3, and 2 matches to only 3, assign all to the same group.
            matches = set()
            
            for key, value in new.items():
                new[key] = new[key] | set([key])
                
                if key not in matches:
                    matches = matches | new[key]
                    
                    for x in value:
                        if x == key:
                            continue
                        
                        if new[key] != new[x]:
                            new[key] = new[key] | new[x]
                            new[x] = new[key]
                            
                else:
                    new[key] = new[min(new[key])]
            
        for key, value in new.items():
            new[key] = [value]

            
        # Append to the overal matched dictionary.
        if len(new) > 0:
            matched.update(new)
               

    # Save the matches to the output dictionary.
    output['matched{0}'.format(procnum)] = matched