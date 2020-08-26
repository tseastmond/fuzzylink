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
# Make a function for the actual matching.
def DeDup(full, idvar, exact, nomismatch=[], fuzzy=[], strthresh=0.9, 
          numthresh=1, weight=1, allowmiss=False, disp=1, cores=1):
    '''
    A function to identify duplicates within a Pandas DataFrame. 
    

    Parameters
    ----------
    full : Pandas DataFrame
        The full dataset in which we want to eliminate duplicates.
    idvar : str
        The column name of a column which uniquely identifies each row.
    exact : list
        A list of columns on which the algorithm matches exactly.
    nomismatch : list
        A list of columns requiring no discrepancy in nonmissing values, but
        it will allow a match between missing and a value.
    fuzzy : list
        A list of columns requiring a fuzzy match, with the threshold 
        specified by strthresh or numthresh, depending on the column type.
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
    cores : int
        The number of process to run simultaneously.
            
    
    Returns
    -------
    A Pandas DataFrame with two columns, the first with a copy of "idvar" and
    the second with a set containing all of the values of "idvar" that match
    with the given row.
    '''
    # Keep only relevant columns and copy the DataFrame.
    cols = exact + nomismatch + fuzzy + [idvar]
    cols = list(set(cols))
        
    full = full[cols].copy()
    
    
    # Get a unique integer id for our 'exact' columns. 
    full['__exact__'] = ''
    
    for col in exact:
        full[col] = full[col].fillna('').astype(str)
        full = full.loc[full[col] != '', :]
        full['__exact__'] += ',' + full[col]
        del full[col]

    full['__exact__'] = full['__exact__'].rank(method='dense', na_option='top').astype(int)
    

    # Get the unique values of exact columns.
    vals = list(full['__exact__'].value_counts().reset_index()\
        .query('index != "," and __exact__ > 1')['index'])
        
        
    # Split up the unique values into the number of designated cores.
    # Make a list of empty lists.
    splitvals = [[] for _ in range(cores)]
    
    # Loop over vals. Since the values that will take the longest are ordered 
    #  first to last, we want to split them over processes.
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

    
    # Mark the start time.
    start_time = time()
    
    
    # Start the number of processes requested.
    all_procs = []
    full['__id__'] = full[idvar]
    for proc in range(1, cores+1):
        # Initialize the process.
        shuffle(splitvals[proc-1]) # This will give more acurate timing
        p = Process(target=_loop, args=(full.loc[full['__exact__']\
                                                .isin(splitvals[proc-1])\
                                                .copy()],
                                        splitvals[proc-1], [idvar, '__id__'], 
                                        proc, output, progress, nomismatch,
                                        fuzzy, '', strthresh, numthresh,
                                        weight, allowmiss, None, True))                            
        
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
        timer = Process(target=_timer,
                        args=(progress, cores, start_time, disp), daemon=True)
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
    
    print('Cleaning up the output....')
    
    
    # Collect the output.
    matched = dict()
    
    for proc in range(1, cores+1):
        if len(output['matched{0}'.format(proc)]) > 0:
            matched.update(output['matched{0}'.format(proc)])
    
    
    # Make a DataFrame for the results and rename columns.
    matched = pd.DataFrame.from_dict(matched, orient='index').reset_index()
    
    if len(matched) > 0:
        matched.columns = [idvar, 'duplicates']


    # Get rows that originally were in their own block, so were not matched.
    if len(full) > 0:
        temp = full.apply(lambda x: [x[idvar], set([x[idvar]])],\
                          axis=1, result_type='expand')
        temp.columns = [idvar, 'duplicates']
        matched = matched.append(temp, ignore_index=True)


    # Return the two DataFrames.
    return matched
