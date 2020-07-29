#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Tanner S Eastmond
Date Updated: 7/8/2020
Purpose: This is a program used to perform fuzzy matches between two sets of
  records.
'''

from jellyfish import jaro_winkler
from multiprocessing import Process, Manager
import numpy as np
import os
import pandas as pd
import sys
from time import time, sleep




###############################################################################
###############################################################################
###############################################################################
# Make a function to filter out rows.
def RowFilter(row, val, nomismatch=[], fuzzy=[], strthresh=0.9,
              numthresh=1, weight=1, allowmiss=False):
    '''    
    Compares two series of data and returns 'True' if the series are a match
    based on the specified criteria or 'False' otherwise.

    This is an appendage to Match(.) and is intended for use in an 'apply' or
    'map' command.


    Parameters
    ----------
    row        - Pandas series with the values to check against 'val'.
    val        - Pandas series with the main values we are trying to match.
    nomismatch - list - List of columns requiring no discrepancy in nonmissing
                 values, but will allow a match between missing and a value.
    fuzzy      - list - List of columns requiring a fuzzy match, with the
                 threshold specified by strthresh or numthresh, depending
                 on the column type.
    strthresh  - float or dict - The threshold for Jaro-Winkler score below
                 which the rows are not a match. If a dictionary is passed,
                 it must have one entry per fuzzy column specifying
                 the threshold for each.
    numthresh  - float or dict - The threshold for numeric variables absolute
                 difference outside of which the rows are not a match.
    weight     - float or dict - The weight for each column to be applied in
                 calculating the score.
    allowmiss  - bool - Allow a mismatch in fuzzy due to missing values.


    Returns
    -------
    A Pandas Series with 'True' for matches and 'False' for nonmatches.
    '''
    # Convert to dictionaries as needed.
    if type(strthresh) != dict and fuzzy != []:
        strthresh = {col : strthresh for col in fuzzy}

    if type(numthresh) != dict and fuzzy != []:
        numthresh = {col : numthresh for col in fuzzy}
        
    if type(weight) != dict and (nomismatch != [] or fuzzy != []):
        weight = {col : weight for col in nomismatch + fuzzy}
        
        
    # Initialize a variable to keep the score.
    score = 0.0


    # First check the nomismatch columns.
    for col in nomismatch:
        if pd.isnull(row[col]) or pd.isnull(val[col]):
            pass
        elif str(row[col]).strip() == '' or str(val[col]).strip() == '':
            pass
        else:
            if str(row[col]).strip() != str(val[col]).strip():
                return [False, 0.0]
            
            else:
                score += weight[col]


    # Next check the fuzzy columns.
    for col in fuzzy:
        if type(row[col]) == str:
            if allowmiss and (row[col] == '' or val[col] == '' or \
                              pd.isnull(row[col]) or pd.isnull(val[col])):
                pass
            
            elif (row[col] == '' or pd.isnull(row[col])) \
                and (val[col] == '' or pd.isnull(val[col])):
                pass
            
            elif pd.isnull(row[col]) or pd.isnull(val[col]):
                return [False, 0.0]
            
            else:
                jaro = jaro_winkler(row[col].lower(), val[col].lower())
                
                if jaro < strthresh[col]:
                    return [False, 0.0]
                
                else:
                    score += jaro * weight[col]
                    
        else:
            if allowmiss and (pd.isnull(row[col]) or pd.isnull(val[col])):
                pass
            
            elif (row[col] == '' or pd.isnull(row[col])) \
                and (val[col] == '' or pd.isnull(val[col])):
                pass
            
            elif pd.isnull(row[col]) or pd.isnull(val[col]):
                return [False, 0.0]
            
            else:
                dist = abs(row[col] - val[col])
                
                if dist > numthresh[col]:
                    return [False, 0.0]
                
                else:
                    score += dist * weight[col]


    # If we passed all of the checks, return True.
    return [True, score]




###############################################################################
###############################################################################
###############################################################################
# Make a function to aggregate rows.
def RowAgg(col, agg='mode'):
    '''
    Aggregates all passed rows into one based on the specified criteria.

    This is an appendage to Match(.) and is intended for use in an 'apply' or
    'map' command.


    Parameters
    ----------
    col        - Pandas series to aggregate.
    agg        - str or dict - The mode of aggregation for columns not
                 specified in 'exact'. May be one of the following,
                 the default is 'mode':
                     - 'mode'  - Take the mode of the values in matched rows.
                     - 'sum'   - Take the sum of the values in matched rows.
                     - 'mean'  - Take the mean of the values in matched rows.
                     - 'count' - Count the number of nonmissing/nonempty
                                 values in matched rows.
                     - 'len'   - Take the longest string from among the
                                 options.
                     - 'all'   - Return a string with all former values 
                                 separated by ";;".
                     - A dictionary using these options as the keys, which will
                       apply the different aggregation to each column:

                           {'mode' : [col1, col2, col3], 'sum' : [col6],
                           'mean' : [col5, col7]}


    Returns
    -------
    A single value aggregated from the column as specified.
    '''
    # Make agg a dictionary as needed.
    if type(agg) != dict:
        agg = {agg : col.name}


    # Ensure that the dictionary has all needed keys.
    for x in ['mode', 'sum', 'mean', 'count', 'len', 'all']:
        if x not in agg.keys():
            agg[x] = []


    # Reset the index for our column.
    col = col.reset_index(drop=True)


    # Check what the specified mode of aggregation is.
    if col.name in agg['mode']:
        # For 'mode', return the mode of the column. If the mode is an empty 
        # string, return that only if it is the only possible value.
        col.fillna('', inplace=True)

        if col.value_counts().index[0] != '' or \
            len(col.value_counts().index) == 1:
            return col.value_counts().index[0]

        else:
            return col.value_counts().index[1]

    elif col.name in agg['sum']:
        # For 'sum', return the sum of the column.
        return sum(col)

    elif col.name in agg['mean']:
        # For 'mean', return the mean of the column.
        return np.mean(col)

    elif col.name in agg['count']:
        # For 'count', return the number of nonmissing observations.
        return len(list(filter(None, col)))

    elif col.name in agg['len']:
        # For 'len', return the longest string in the column.
        col.fillna('', inplace=True)

        return sorted(col, key=len)[-1]

    elif col.name in agg['all']:
        # For 'all', concatenate all values in the column into a list and
        #   return that list.
        col = col.loc[(col != '') & (col.notnull())]
        #col.fillna('', inplace=True)

        return ';;'.join([str(x) for x in col.value_counts().index])

    else:
        # If nothing is specified, return the first value.
        return col[0]
    
    
    
    
###############################################################################
###############################################################################
###############################################################################
# Make a function to filter out rows.
def Timer(progress, cores, start_time, disp):
    '''
    This keeps track of progress for our matching and prints it out to the 
    console.
    

    Parameters
    ----------
    progress : TYPE
        DESCRIPTION.
    cores : TYPE
        DESCRIPTION.
    start_time : TYPE
        DESCRIPTION.
    disp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.
    '''
    sleep(10)
    
    while True:          
        sleep(disp)
        s = ''
        
        now = time()
        finished = 0
        
        for x in range(1, cores+1):
            s1 = 'Process ' + str(x) + ':' + ' '*(3-len(str(x))) + '[' + \
            '='*round(40*(progress['p'+str(x)]/progress['tot'+str(x)])) +\
            ' '*(40 - round(40*(progress['p'+str(x)]/progress['tot'+str(x)]))) +\
            '] ' + str(round(100*(progress['p'+str(x)]/progress['tot'+str(x)]), 1)) + '%  -  ' +\
            'Est. ' + str(round((((now-start_time)/progress['p'+str(x)]) *(progress['tot'+str(x)] - progress['p'+str(x)]))/3600,2)) +\
            ' Hours Remaining\n\n'
            
            s = ''.join([s, s1])
            
            if progress['p'+str(x)]/progress['tot'+str(x)] == 1:
                finished += 1
    
        os.system('clear')
        print(s)
        sys.stdout.flush()
        
        if finished == cores:
            return

        
        
    
    
###############################################################################
###############################################################################
###############################################################################
# Make a function to loop over unique blocks.
def Loop(full, vals, procnum, output, progress,
         nomismatch, fuzzy, onlycheck, agg, strthresh, numthresh, weight, 
         allowmiss, nummatches, crosswalk):
    '''
    This actually runs the loop to match values using the blocks designated by
    "vals".
    
    
    Parameters
    ----------
    full       - Pandas Dataframe - This contains all of the data for matching.
    vals       - list - This includes all unique values of the "__exact__"
                 column in the full dataframe.
    procnum    - int - The number of the process running.
    output     - dict - A dictionary where the output from each process will be
                 saved.
    fulllen    - int - The total number of rows to be checked.
    start_time - float - The time the program started running.
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
    agg        - str or dict - The mode of aggregation for columns not
                 specified in 'exact'. May be one of the following,
                 the default is 'mode':
                     - 'mode'  - Take the mode of the values in matched rows.
                     - 'sum'   - Take the sum of the values in matched rows.
                     - 'mean'  - Take the mean of the values in matched rows.
                     - 'count' - Count the number of nonmissing/nonempty
                                 values in matched rows.
                     - 'len'   - Take the longest string from among the
                                 options.
                     - 'all'   - Return a string with all former values 
                                 separated by ";;".
                     - A dictionary using these options as the keys, which will
                       apply the different aggregation to each column:

                           {'mode' : [col1, col2, col3], 'sum' : [col6],
                           'mean' : [col5, col7]}
    strthresh  - float or dict - The threshold for Jaro-Winkler score below
                 which the rows are not a match. If a dictionary is passed,
                 it must have one entry per fuzzy column specifying
                 the threshold for each.
    numthresh  - float - The threshold for numeric variables absolute
                 difference outside of which the rows are not a match.
    weight     - float or dict - The weight for each column to be applied in
                 calculating the score.       
    allowmiss  - bool - Allow a mismatch in fuzzy due to missing values.
    disp       - float - A real number between 0-100 indicating how often you
                 want to display the progress (i.e. disp=5 means the code will
                 print out every time it has progressed 5 percent).
    nummatches - None or int - None if all matches are to be collected or an
                 integer if a particular number of matches is to be returned.
    crosswalk  - None or list - list with two values, the id variable for the
                 rows to be matched first and the id variable for the 
                 comparison pool second, None if the rows are to be 
                 aggregated as described above.
                 
    
    Returns
    -------
    Nothing
    '''
    # Get the proper number of matches to be made.
    if onlycheck == '':
        fulllen = len(full.loc[full['__exact__'].isin(vals)])
    else:
        fulllen = len(full.loc[(full[onlycheck] == True) 
                               & (full['__exact__'].isin(vals))])
        
    progress['tot'+str(procnum)] = fulllen
        
    # Initialize matched and unmatched.
    matched = []
    unmatched = []    
    
    # Loop over unique values of exact columns.
    total = 0
    check = 0
    #count = disp
    for val in vals:
        # Make a new DataFrame to house matches on exact.
        temp = full.loc[full['__exact__'] == val, :].reset_index(drop=True)
        check += len(temp)
        
        # Delete the values from full.
        full = full.loc[full['__exact__'] != val, :]        
        

        # Sort the values correctly if onlycheck is specified.
        if onlycheck != '':
            temp.sort_values(onlycheck, ascending=False, inplace=True)
            temp.reset_index(drop=True, inplace=True)


        # Check for discrepancies in the nomismatch columns and fuzzy columns.
        while len(temp) > 0:
            # Stop if onlycheck is specified and the specified sample is done.
            if onlycheck != '':
                if temp[onlycheck].sum() == 0:
                    unmatched += temp.to_dict('r')
                    break

            # Get the matching indecies for the first entry.
            if onlycheck != '':
                fil = temp.loc[(temp.index == 0) | 
                    (temp[onlycheck] == False), :].apply(RowFilter, axis=1, 
                    result_type='expand', args=(temp.loc[0], nomismatch, fuzzy, 
                    strthresh, numthresh, weight, allowmiss))
                                                         
                score = fil[1]
                fil = fil[0]
                
                score.loc[0] = 0.0
                                                         
                if nummatches is not None:
                    fil.loc[score != score.max()] = False
                    fil.loc[list(fil.loc[fil == True].index)[nummatches:]]\
                        = False
                    
                    fil.loc[0] = True
                    
                fil = pd.concat([fil, temp['index']], axis=1)[0].fillna(False)

            else:
                fil = temp.apply(RowFilter, axis=1, 
                                        result_type='expand', 
                                        args=(temp.loc[0],
                                        nomismatch, fuzzy, strthresh,
                                        numthresh, weight, allowmiss))
                
                score = fil[1]
                fil = fil[0]
                
                score.loc[0] = 0.0
                
                if nummatches is not None:
                    fil.loc[score != score.max()] = False
                    fil.loc[list(fil.loc[fil == True].index)[nummatches:]]\
                        = False
                    
                    fil.loc[0] = True

            # Get the matching rows and delete from temp.
            new = temp.loc[fil, :]

            temp = temp.loc[~fil, :].reset_index(drop=True)
        

            # Update total to track our progress.
            if onlycheck == '':
                total += len(new)
            else:
                total += len(new.loc[new[onlycheck] == True])
                
            progress['p'+str(procnum)] = total

            # Print our progress.
            #if count - round((total/fulllen)*100,2) <= 0:
            #    now = time()
            #    print('\n\nProcess Number', procnum)
            #    print('Progress:         ', round((total/fulllen)*100,2), '%')
            #    print('Time Remaining:   ', round((((now-start_time)/total) * 
            #        (fulllen - total))/3600,2), 'Hours')
            #    print(' ')
            #    sys.stdout.flush()
            #    count += disp

            # Move forward if only one observation.
            if len(new) == 1:
                unmatched += new.to_dict('r')
                continue


            # Aggregate the rows.
            if crosswalk != None:
                new[crosswalk[0]] = new[crosswalk[0]].astype(str)
                new[crosswalk[1]] = new[crosswalk[1]].astype(str)
                
                matched += [{crosswalk[0]: list(new[crosswalk[0]])[0], 
                             crosswalk[1]: 
                                 ';;'.join(list(new[crosswalk[1]])[1:])}]
            else:
                matched += [pd.DataFrame(new.apply(RowAgg, axis=0, args=(agg,))).to_dict('d')[0]]
            
    del full
    
    # Return the data.
    #print(unmatched)
    output['matched{0}'.format(procnum)] = matched
    output['unmatched{0}'.format(procnum)] = unmatched




###############################################################################
###############################################################################
###############################################################################
# Make a function for the actual matching.
def Match(df, exact, nomismatch=[], fuzzy=[], onlycheck='', agg='mode', 
          strthresh=0.9, numthresh=1, weight=1, allowmiss=False, disp=5, 
          nummatches=None, cores=1, crosswalk=None):
    '''
    This takes a Pandas DataFrame with potential duplicate rows and matches
    them based on the columns you specify. It then returns two dataframes,
    one with matched observations and one with unmatched observations.
    It requires at least one column on which to make an exact match and allows
    fuzzy matches based on any other columns. It also requires one column 
    named "index" which uniquely identifies each row.

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
    agg        - str or dict - The mode of aggregation for columns not
                 specified in 'exact'. May be one of the following,
                 the default is 'mode':
                     - 'mode'  - Take the mode of the values in matched rows.
                     - 'sum'   - Take the sum of the values in matched rows.
                     - 'mean'  - Take the mean of the values in matched rows.
                     - 'count' - Count the number of nonmissing/nonempty
                                 values in matched rows.
                     - 'len'   - Take the longest string from among the
                                 options.
                     - 'all'   - Return a string with all former values 
                                 separated by ";;".
                     - A dictionary using these options as the keys, which will
                       apply the different aggregation to each column:

                           {'mode' : [col1, col2, col3], 'sum' : [col6],
                           'mean' : [col5, col7]}
    strthresh  - float or dict - The threshold for Jaro-Winkler score below
                 which the rows are not a match. If a dictionary is passed,
                 it must have one entry per fuzzy column specifying
                 the threshold for each.
    numthresh  - float - The threshold for numeric variables absolute
                 difference outside of which the rows are not a match.
    allowmiss  - bool - Allow a mismatch in fuzzy due to missing values.
    weight     - float or dict - The weight for each column to be applied in
                 calculating the score.
    disp       - float - A real number between 0-100 indicating how often you
                 want to display the progress (i.e. disp=5 means the code will
                 print out every time it has progressed 5 percent).
    nummatches - None or int - None if all matches are to be collected or an
                 integer if a particular number of matches is to be returned.
    cores      - int - The number of separate processes to simultaneously run
                 using the multiprocessing Process class.
    crosswalk  - None or list - list with two values, the id variable for the
                 rows to be matched first and the id variable for the 
                 comparison pool second, None if the rows are to be 
                 aggregated as described above.


    Returns
    -------
    matched_df   - A Pandas DataFrame containing matched rows.
    unmatched_df - A Pandas DataFrame containing unmatched rows.
    '''

    # Combine our exact columns into one.
    if onlycheck == '':
        onlycheckcol = []
    else:
        onlycheckcol = [onlycheck]
    
    if crosswalk != None:
        cols = ['index'] + exact + nomismatch + fuzzy + onlycheckcol + crosswalk
    else:
        cols = ['index'] + exact + nomismatch + fuzzy + onlycheckcol
        
    cols = list(set(cols))
        
    full = df[cols].copy()
    full['__exact__'] = ''

    for col in exact:
        full['__exact__'] += ',' + full[col].astype(str)


    # Fill missing values with missing string.
    full['__exact__'].fillna('', inplace=True)


    # Get the unique values of exact columns and save the singular values.
    if onlycheck != '':
        vals = list(full.loc[full[onlycheck] == True, '__exact__'].value_counts()\
            .reset_index().query('index != ","')['index'])
                
        vals1 = pd.Series(full['__exact__'].value_counts()\
                          .reset_index()['index'])
            
        vals1 = list(vals1.loc[~vals1.isin(list(vals))])

    else:
        vals = full['__exact__'].value_counts().reset_index()\
            .query('index != "," and __exact__ > 1')['index']
            
        vals1 = full['__exact__'].value_counts().reset_index()\
            .query('index == "," or __exact__ == 1')['index']
       
        
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
        p = Process(target=Loop, args=(full.loc[full['__exact__']\
                                                .isin(splitvals[proc-1])\
                                                .copy()], 
                                       splitvals[proc-1], proc, output, 
                                       progress, nomismatch, fuzzy, 
                                       onlycheck, agg, strthresh, 
                                       numthresh, weight, allowmiss,
                                       nummatches, crosswalk))                                      
        
        p.start()        
        print('\n\nStarted Process', proc)
        
        # Save the processes in a list.
        all_procs.append(p)
        
        # Drop the associated values from full so as to not double down on the
        #  memory usage.
        full = full.loc[~(full['__exact__'].isin(splitvals[proc-1])), :]
        
        
    # Start a process to track and print remaining time.
    p = Process(target=Timer, args=(progress, cores, start_time, disp), daemon=True)
    p.start()
    
    
    # Join the processes.
    for p in all_procs:
        p.join()
    
        
    # Collect the output.    
    matched = []
    unmatched = []
    
    for proc in range(1, cores+1):
        if len(output['matched{0}'.format(proc)]) > 0:
            matched += output['matched{0}'.format(proc)]
            
        if len(output['unmatched{0}'.format(proc)]) > 0:
            unmatched += output['unmatched{0}'.format(proc)]
            
    
    # Convert to DataFrames.
    matched = pd.DataFrame.from_dict(matched)
    unmatched = pd.DataFrame.from_dict(unmatched)
        

    # Save to the unmatched DataFrame.
    if len(full) > 0:
        unmatched = unmatched.append(full, ignore_index=True)


    # Drop the __exact__ column.
    if len(matched) > 0 and crosswalk == None:
        matched.drop('__exact__', axis=1, inplace=True)

    unmatched.drop('__exact__', axis=1, inplace=True)


    # Return the two DataFrames.
    return matched, unmatched
