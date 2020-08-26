#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from multiprocessing import Process, Manager
from os import kill
from random import shuffle
from time import time, sleep

import pandas as pd

from ._loop import _loop
from ._memory_check import _memory_check
from ._timer import _timer


###############################################################################
###############################################################################
###############################################################################
def Match(tomatch, comparison, idcols, exact, nomismatch=[], fuzzy=[], 
          colmap='', strthresh=0.9, numthresh=1, weight=1, allowmiss=False, 
          disp=1, nummatches=None, cores=1):
    '''
    This takes two Pandas DataFrames and matches the observations together
    according to the specified criteria.

    Parameters
    ----------
    tomatch : Pandas DataFrame
        The full dataset containing the observations to be matched.
    comparison : Pandas DataFrame
        The full dataset containing the comparison observations.
    idcols : list
        A list where the first element is the name of the column which uniquely
        identifies each row in the "tomatch" DataFrame and the second element
        is the name of the column which uniquely identifies each row in the
        "comparison" DataFrame.
    exact : list
        A list of columns on which the algorithm matches exactly. If the names
        of columns to be compared are different in "tomatch" and "comparison",
        this should be the names of the columns in "comparison" and "colmap"
        will handle the name differences.
    nomismatch : list
        A list of columns requiring no discrepancy in nonmissing values, but
        it will allow a match between missing and a value. If the names of
        columns to be compared are different in "tomatch" and "comparison",
        this should be the names of the columns in "comparison" and "colmap"
        will handle the name differences.
    colmap : dict
        A dictionary with the names of the columns in "tomatch" as keys and the
        names of the associated columns in "comparison".
    fuzzy : list
        A list of columns requiring a fuzzy match, with the threshold 
        specified by strthresh or numthresh, depending on the column type. If
        the names of columns to be compared are different in "tomatch" and
        "comparison", this should be the names of the columns in "comparison"
        and "colmap" will handle the name differences.
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
    disp : int or None
        The number of seconds between each update of the printed output in the
        console. If None there will be no printed progress in the console.
    nummatches : int
        The number of matches to find from the comparison pool.
    cores : int
        The number of process to run simultaneously.
            
    
    Returns
    -------
    A Pandas DataFrame with two columns, the first with the identifier from
    "tomatch" specified in the first entry of "idcols" and the second with a
    set containing all of the values of the identifier from "comparison"
    specified in the second entry of "idcols" that match with the given row.
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
        full[col] = full[col].fillna('').astype(str)
        full = full.loc[full[col] != '', :]
        full['__exact__'] += ',' + full[col]
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
    processes = manager.list()
        
    
    # Start the memory check.
    mem = Process(target=_memory_check, args=(processes,output), daemon=True)
    mem.start()
    
    
    # Start the timer.
    start_time = time()
    
    
    # Start the number of processes requested.
    all_procs = []
    for proc in range(1, cores+1):
        # Initialize the process.
        shuffle(splitvals[proc-1]) # This will give us more acurate timing
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
        processes.append(p.pid)
        
        # Break if memory is too high.
        if 'end' in output.keys():
            raise MemoryError('Memory Usage Too High, Exiting...')
        
        # Drop the associated values from full so as to not double down on the
        #  memory usage.
        full = full.loc[~(full['__exact__'].isin(splitvals[proc-1])), :]
        
        
    # Start a process to track and print remaining time.
    if disp is not None:
        timer = Process(target=_timer, args=(progress, cores, start_time, disp), daemon=True)
        processes.append(timer.pid)
        timer.start()
    
    
    # Wait for all processes to finish, make sure they all end in case of user
    #  stopping the program.
    for p in all_procs:
        try:
            p.join()
        except KeyboardInterrupt:
            for proc in processes:
                try:
                    kill(proc, 9)
                    print('Killed', proc)
                except:
                    pass
                
                raise KeyboardInterrupt
        
        
    # Break if processes ended because memory was too high.
    if 'end' in output.keys():
        raise MemoryError('Memory Usage Too High, Exiting...')

    
    # Terminate the timer and memory check.
    if disp is not None:
        sleep(disp*2)
        timer.terminate()
        mem.terminate()
    
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
