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
import pandas as pd
import sys
from time import time




###############################################################################
###############################################################################
###############################################################################
# Make a function to filter out rows.
def RowFilter(row, val, nomismatch=[], fuzzy=[], strthresh=0.9,
              numthresh=1, allowmiss=False):
    '''
    Description
    -----------
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
    numthresh  - float - The threshold for numeric variables absolute
                 difference outside of which the rows are not a match.
    allowmiss  - bool - Allow a mismatch in fuzzy due to missing values.


    Returns
    -------
    'True' for matches and 'False' for nonmatches.
    '''
    # Make a dictionary as needed.
    if type(strthresh) != dict and fuzzy != []:
        strthresh = {col : strthresh for col in fuzzy}

    if type(numthresh) != dict and fuzzy != []:
        numthresh = {col : numthresh for col in fuzzy}


    # First check the nomismatch columns.
    for col in nomismatch:
        if pd.isnull(row[col]) or pd.isnull(val[col]):
            pass
        elif str(row[col]).strip() == '' or str(val[col]).strip() == '':
            pass
        else:
            if str(row[col]).strip() != str(val[col]).strip():
                return False


    # Next check the fuzzy columns.
    for col in fuzzy:
        if type(row[col]) == str:
            if allowmiss and (row[col] == '' or val[col] == '' or pd.isnull(row[col]) or pd.isnull(val[col])):
                pass
            elif (row[col] == '' or pd.isnull(row[col])) and (val[col] == '' or pd.isnull(val[col])):
                pass
            elif pd.isnull(row[col]) or pd.isnull(val[col]):
                return False
            else:
                if jaro_winkler(row[col], val[col]) < strthresh[col]:
                    return False
        else:
            if allowmiss and (pd.isnull(row[col]) or pd.isnull(val[col])):
                pass
            elif (row[col] == '' or pd.isnull(row[col])) and (val[col] == '' or pd.isnull(val[col])):
                pass
            elif pd.isnull(row[col]) or pd.isnull(val[col]):
                return False
            else:
                if abs(row[col] - val[col]) > numthresh[col]:
                    return False


    # If we passed all of the checks, return True.
    return True




###############################################################################
###############################################################################
###############################################################################
# Make a function to aggregate rows.
def RowAgg(col, agg='mode'):
    '''
    Description
    -----------
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
                     - 'all'   - Return a list with all of the values.
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
        # For 'mode', return the mode of the column. If the mode is an empty string,
        #   return that only if it is the only possible value.
        col.fillna('', inplace=True)

        if col.value_counts().index[0] != '' or len(col.value_counts().index) == 1:
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
# Make a function to loop over unique blocks.
def Loop(matched, unmatched, full, vals, procnum, output, start_time,  
         nomismatch, fuzzy, onlycheck, agg, strthresh, numthresh, allowmiss, 
         disp):
    '''
    Description
    -----------
    This actually runs the loop to match values using the blocks designated by
    "vals".
    
    
    Parameters
    ----------
    matched    - Pandas Dataframe - This is where matched values will be 
                 stored.
    unmatched  - Pandas Dataframe - This is where unmatched values will be 
                 stored.
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
                 matched itself. If a column name is specified for this variable
                 then there cannot be any cases in the matched sample where only
                 rows with False specified are matched together since those
                 rows will not be explicitly checked against the comparison
                 pool.
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
                     - 'all'   - Return a list with all of the values.
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
    disp       - float - A real number between 0-100 indicating how often you
                 want to display the progress (i.e. disp=5 means the code will
                 print out every time it has progressed 5 percent).
                 
    
    Returns
    -------
    Two dataframes, the first with matched values and the second with unmatched 
    values.
    '''
    # Loop over unique values of exact columns.
    if onlycheck == '':
        fulllen = len(full.loc[full['__exact__'].isin(vals)])
    else:
        fulllen = len(full.loc[(full[onlycheck] == True) & (full['__exact__'].isin(vals))])
    
    total = 0
    check = 0
    count = disp
    for val in vals:
        # Make a new DataFrame to house matches on exact.
        temp = full.loc[full['__exact__'] == val, :].reset_index(drop=True)
        check += len(temp)
        

        # Sort the values correctly if onlycheck is specified.
        if onlycheck != '':
            temp.sort_values(onlycheck, ascending=False, inplace=True)
            temp.reset_index(drop=True, inplace=True)


        # Check for discrepancies in the nomismatch columns and fuzzy columns.
        while len(temp) > 0:
            # Stop if onlycheck is specified and the specified sample is done.
            if onlycheck != '':
                if temp[onlycheck].sum() == 0:
                    unmatched = unmatched.append(temp, ignore_index=True)
                    break

            # Get the matching indecies for the first entry.
            if onlycheck != '':
                fil = temp.loc[(temp.index == 0) | (temp[onlycheck] == False), :].apply(RowFilter, axis=1, args=(temp.loc[0],
                    nomismatch, fuzzy, strthresh,
                    numthresh, allowmiss))
                
                fil = pd.concat([fil, temp['index']], axis=1)[0].fillna(False)

            else:
                fil = temp.apply(RowFilter, axis=1, args=(temp.loc[0],
                    nomismatch, fuzzy, strthresh,
                    numthresh, allowmiss))

            # Get the matching rows and delete from temp.
            new = temp.loc[fil, :]

            temp = temp.loc[~fil, :].reset_index(drop=True)
            

            # Update total to track our progress.
            if onlycheck == '':
                total += len(new)
            else:
                total += len(new.loc[new[onlycheck] == True])

            # Print our progress.
            if count - round((total/fulllen)*100,2) <= 0:
                now = time()
                print('\n\nProcess Number', procnum)
                print('Progress:         ', round((total/fulllen)*100,2), '%')
                print('Time Remaining:   ', round((((now-start_time)/total) * (fulllen - total))/3600,2), 'Hours')
                print(' ')
                sys.stdout.flush()
                count += disp

            # Move forward if only one observation.
            if len(new) == 1:
                unmatched = unmatched.append(new, ignore_index=True)
                continue


            # Aggregate the rows.
            matched = matched.append(new.apply(RowAgg, axis=0, args=(agg,)), ignore_index=True)
    
    # Return the data.
    output['matched{0}'.format(procnum)] = matched
    output['unmatched{0}'.format(procnum)] = unmatched




###############################################################################
###############################################################################
###############################################################################
# Make a function for matching.
def Match(df, exact, nomismatch=[], fuzzy=[], onlycheck='', agg='mode', strthresh=0.9,
          numthresh=1, allowmiss=False, disp=5, cores=1):
    '''
    Description
    -----------
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
                 matched itself. If a column name is specified for this variable
                 then there cannot be any cases in the matched sample where only
                 rows with False specified are matched together since those
                 rows will not be explicitly checked against the comparison
                 pool.
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
                     - 'all'   - Return a list with all of the values.
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
    disp       - float - A real number between 0-100 indicating how often you
                 want to display the progress (i.e. disp=5 means the code will
                 print out every time it has progressed 5 percent).
    cores      - int - The number of separate processes to simultaneously run
                 using the multiprocessing Process class.


    Returns
    -------
    matched_df, unmatched_df - A Pandas DataFrame containing the matched rows
                               followed by a Pandas DataFrame containing
                               unmatched rows.
    '''

    # Define the matches DataFrame.
    matched = pd.DataFrame()
    unmatched = pd.DataFrame()


    # Combine our exact columns into one.
    full = df.copy()
    full['__exact__'] = ''

    for col in exact:
        full['__exact__'] += ',' + full[col].astype(str)


    # Fill missing values with missing string.
    full['__exact__'].fillna('', inplace=True)


    # Get the unique values of exact columns and save the singular values.
    if onlycheck != '':
        t = full.loc[full[onlycheck] == True, '__exact__'].value_counts().reset_index().query('index != ","')['index']
        
        vals = []
        
        vals1 = list(full['__exact__'].value_counts().reset_index()['index'])
        
        for x in t:
            if x in vals1:
                vals += [x]
                vals1.remove(x)

    else:
        vals = full['__exact__'].value_counts().reset_index().query('index != "," and __exact__ > 1')['index']
        vals1 = full['__exact__'].value_counts().reset_index().query('index == "," or __exact__ == 1')['index']

    # Start the timer.
    start_time = time()
       
        
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
    
    
    # Start the number of processes requested.
    all_procs = []
    for proc in range(1, cores+1):
        # Initialize the process.
        p = Process(target=Loop, args=(matched.copy(), unmatched.copy(), full, 
                                                    splitvals[proc-1], proc, output, 
                                                    start_time, nomismatch, fuzzy, 
                                                    onlycheck, agg, strthresh, 
                                                    numthresh, allowmiss, disp))
        
        p.start()        
        
        # Save the processes in a list.
        all_procs.append(p)
    
    
    # Join the processes.
    for p in all_procs:
        p.join()
    
        
    # Collect the output.
    for proc in range(1, cores+1):
        matched = matched.append(output['matched{0}'.format(proc)], ignore_index=True)
        unmatched = unmatched.append(output['unmatched{0}'.format(proc)], ignore_index=True)
        

    # Save to the unmatched DataFrame.
    if len(full.loc[full['__exact__'].isin(vals1)]) > 0:
        unmatched = unmatched.append(full.loc[full['__exact__'].isin(vals1), :], ignore_index=True)


    # Drop the __exact__ column.
    if len(matched) > 0:
        matched.drop('__exact__', axis=1, inplace=True)

    unmatched.drop('__exact__', axis=1, inplace=True)


    # Return the two DataFrames.
    return matched, unmatched
