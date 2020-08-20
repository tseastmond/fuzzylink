#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import Process, Manager
import random
from time import time, sleep

import pandas as pd

from ._loop import _loop
from ._timer import _timer


###############################################################################
###############################################################################
###############################################################################
def Match(tomatch, comparison, idcols, exact, nomismatch=[], fuzzy=[], 
          colmap='', strthresh=0.9, numthresh=1, weight=1, allowmiss=False, 
          disp=1, nummatches=None, cores=1):
    '''
    This takes a Pandas DataFrame with potential duplicate rows and matches
    them based on the columns you specify. It then returns two dataframes,
    one with matched observations and one with unmatched observations.
    It requires at least one column on which to make an exact match and allows
    fuzzy matches based on any other columns.

    The matched data will be collapsed (i.e. each row matched with other
    rows will be collapsed into one row) based on the aggregation you specify.
    If you pass a dictionary into agg of the following form:

        {'mode' : [col1, col2, col3], 'sum' : [col6], 'mean' : [col5, col7]}

    it will then apply the different aggregation to each of the columns.
    Requires the Python jellyfish, os, pandas, sys, and time modules.


    Parameters
    ----------
    df         - Pandas DataFrame
    exact      - list - List of columns requiring an exact match.
    nomismatch - list - List of columns requiring no discrepancy in nonmissing
                 values, but will allow a match between missing and a value.
    fuzzy      - list - List of columns requiring a fuzzy match, with the
                 threshold specified by strthresh or numthresh, depending
                 on the column type.
    onlycheck  - str - The name of a column with only True and False, where
                 True signifies that the row is to be matched against
                 all other observations and False signifies that a row is to be
                 including in the comparison pool for other matches but not
                 matched itself. If a column name is specified for this 
                 variable then there cannot be any cases in the matched sample 
                 where only rows with False specified are matched together 
                 since those rows will not be explicitly checked against the 
                 comparison pool.
    strthresh  - float or dict - The threshold for Jaro-Winkler score below
                 which the rows are not a match. If a dictionary is passed,
                 it must have one entry per fuzzy column specifying
                 the threshold for each.
    numthresh  - float - The threshold for numeric variables absolute
                 difference outside of which the rows are not a match.
    allowmiss  - bool - Allow a mismatch in fuzzy due to missing values.
    weight     - float or dict - The weight for each column to be applied in
                 calculating the score.
    disp       - float - How many seconds to wait before updating the progress
                 display.
    nummatches - None or int - None if all matches are to be collected or an
                 integer if a particular number of matches is to be returned.
    cores      - int - The number of separate processes to simultaneously run
                 using the multiprocessing Process class.
    idcols  - None or list - list with two values, the id variable for the
                 rows to be matched first and the id variable for the 
                 comparison pool second, None if the rows are to be 
                 aggregated as described above.


    Returns
    -------
    matched_df   - A Pandas DataFrame containing matched rows.
    unmatched_df - A Pandas DataFrame containing unmatched rows.
    '''
    # Get copies of our data and distinguish them.
    tomatch = tomatch.copy()
    comparison = comparison.copy()
    
    onlycheck = '__check__'
    
    tomatch['__check__'] = True
    comparison['__check__'] = False
    
    
    # Rename columns if necessary.
    if colmap != '':
        tomatch.rename(columns=colmap, inplace=True)
    
    
    # Keep only relevant columns.       
    cols = exact + nomismatch + fuzzy + [onlycheck]
    cols = list(set(cols))
    
    tomatch = tomatch[cols + [idcols[0]]]
    comparison = comparison[cols + [idcols[1]]]
    
        
    # Append the data together.
    full = tomatch.append(comparison, ignore_index=True)
    
    del tomatch
    del comparison
    
    
    # Get a unique integer id for our 'exact' columns. 
    full['__exact__'] = ''
    
    for col in exact:
        full['__exact__'] += ',' + full[col].astype(str)
        del full[col]

    full['__exact__'] = full['__exact__'].rank(method='dense', na_option='top').astype(int)
    

    # Get the unique values of exact columns and save the singular values.
    vals = list(full.loc[full[onlycheck] == True, '__exact__'].value_counts()\
        .reset_index().query('index != ","')['index'])
        
        
    # Keep only those rows with at least one row from tomatch and comparison in
    #  the block.
    full = full.loc[full['__exact__'].isin(vals), :]
    temp = full.groupby('__exact__')[onlycheck].count().reset_index()
    temp.columns = ['__exact__', '__count__']
    full = full.merge(temp, how='left', on='__exact__')
    full = full.loc[full['__count__'] > 1, :]
    
    del temp
    del full['__count__']

        
    # Split up the unique values into the number of designated cores.
    # Make a list of empty lists.
    splitvals = [[] for _ in range(cores)]
    
    # Loop over vals. Since the values that will take the longest are ordered 
    # first to last, we want to split them over processes.
    _temp = pd.DataFrame(vals, columns=['n']).reset_index()    
    _temp['index'] = _temp['index']%cores
    
    for num in range(cores):
        splitvals[num] = list(_temp.loc[_temp['index'] == num, 'n'])
        
    del _temp

            
    # Give extra values to the last processes as they will be done the fastest.
    for num in range(len(splitvals)):
        if num + 1 > len(splitvals)/2:
            break
        
        if len(splitvals[-1 - num]) < len(splitvals[num]):
            splitvals[-1 - num].append(splitvals[num][-1])
            del splitvals[num][-1]
        
        else:
            break
        
    
    # Set up the dictionary for the output from our processes.
    manager = Manager()
    output = manager.dict()
    progress = manager.dict()
    
    
    # Start the timer.
    start_time = time()
    
    
    # Start the number of processes requested.
    all_procs = []
    for proc in range(1, cores+1):
        # Initialize the process.
        random.shuffle(splitvals[proc-1]) # This will give us more acurate timing
        p = Process(target=_loop, args=(full.loc[full['__exact__']\
                                                .isin(splitvals[proc-1])\
                                                .copy()],
                                        splitvals[proc-1], idcols, proc,
                                        output, progress, nomismatch, fuzzy,
                                        onlycheck, strthresh, numthresh,
                                        weight, allowmiss, nummatches))                                      
        
        p.start()
        print('\n\nStarted Process', proc)
        
        # Save the processes in a list.
        all_procs.append(p)
        
        # Drop the associated values from full so as to not double down on the
        #  memory usage.
        full = full.loc[~(full['__exact__'].isin(splitvals[proc-1])), :]
        
        
    # Start a process to track and print remaining time.
    if disp is not None:
        timer = Process(target=_timer, args=(progress, cores, start_time, disp), daemon=True)
        timer.start()
    
    
    # # Wait for all processes to finish, then terminate the timer.
    for p in all_procs:
        p.join()
    
    if disp is not None:
        sleep(disp*2)
        timer.terminate()
    
    print('Cleaning up the output....')
    
    
    # Collect the output.
    matched = dict()
    
    for proc in range(1, cores+1):
        if len(output['matched{0}'.format(proc)]) > 0:
            matched.update(output['matched{0}'.format(proc)])

    matched = pd.DataFrame.from_dict(matched, orient='index').reset_index()
    
    if len(matched) > 0:
        matched.columns = idcols


    # Save to the unmatched DataFrame.
    if onlycheck != '':
        if len(full.loc[full[onlycheck] == True]) > 0:
            temp = full.loc[full[onlycheck] == True, :].apply(lambda x:\
                            [x[idcols[0]], set()],\
                            axis=1, result_type='expand')
            temp.columns = idcols
            matched = matched.append(temp, ignore_index=True)
    else:
        if len(full) > 0:
            temp = full.apply(lambda x:\
                            [x[idcols[0]], set()],\
                            axis=1, result_type='expand')
            temp.columns = idcols
            matched = matched.append(temp, ignore_index=True)


    # Return the two DataFrames.
    return matched
