#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Tanner S Eastmond
Date Updated: 2/12/2020
Version: 1.1
Purpose: This is a program used to perform fuzzy matches between two sets of
  records.
'''

from jellyfish import jaro_winkler
import numpy as np
import pandas as pd
import sys




###############################################################################
###############################################################################
###############################################################################
# Make a function to filter out rows.
def RowFilter(row, val, nomismatch=[], fuzzy=[], strthresh=0.9,
              numthresh=1, allowmiss=False):
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
    numthresh  - float - The threshold for numeric variables absolute
                 difference outside of which the rows are not a match.
    allowmiss  - bool - Allow a mismatch in fuzzy due to missing values.

    Returns
    -------
    'True' for matches and 'False' for nonmatches.
    '''
    # Make a dictionary as needed.
    if type(strthresh) != dict:
        strthresh = {col : strthresh for col in fuzzy}

    if type(numthresh) != dict:
        numthresh = {col : strthresh for col in fuzzy}


    # First check the nomismatch columns.
    for col in nomismatch:
        if row[col] not in ['', val[col]]:
            return False


    # Next check the fuzzy columns.
    for col in fuzzy:
        if type(row[col]) == str:
            if allowmiss and (row[col] == '' or val[col] == '' or pd.isnull(row[col]) or pd.isnull(val[col])):
                pass
            elif (row[col] == '' or pd.isnull(row[col])) and (val[col] == '' or pd.isnull(val[col])):
                pass
            else:
                if jaro_winkler(row[col], val[col]) < strthresh[col]:
                    return False
        else:
            if allowmiss and (pd.isnull(row[col]) or pd.isnull(val[col])):
                pass
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
        #   return that only if it is the only possible value. This assumes at
        #   least one nonmissing value.
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
        return sorted(col, key=len)[-1]

    elif col.name in agg['all']:
        # For 'all', concatenate all values in the column into a list and
        #   return that list.
        return list(col.value_counts().index)

    else:
        # If nothing is specified, return the first value.
        return col[0]




###############################################################################
###############################################################################
###############################################################################
# Make a function for matching.
def Match(df, exact, nomismatch=[], fuzzy=[], agg='mode', strthresh=0.9,
          numthresh=1, allowmiss=False, disp=5):
    '''
    This takes a Pandas DataFrame with potential duplicate rows and matches
    them based on the columns you specify. It then returns two dataframes,
    one with matched obeservations and one with unmatched observations.
    It requires at least one column on which to make an exact match and allows
    fuzzy matches based on any other columns.

    The matched data will be collapsed (i.e. each row matched with other
    rows will be collapsed into one row) based on the aggregation you specify.
    If you pass a dictionary into agg of the following form:

        {'mode' : [col1, col2, col3], 'sum' : [col6], 'mean' : [col5, col7]}

    it will then apply the different aggregation to each of the columns.
    Requires the Python jellyfish, pandas, and sys modules.

    Parameters
    ----------
    df         - Pandas DataFrame
    exact      - list - List of columns requiring an exact match.
    nomismatch - list - List of columns requiring no discrepancy in nonmissing
                 values, but will allow a match between missing and a value.
    fuzzy      - list - List of columns requiring a fuzzy match, with the
                 threshold specified by strthresh or numthresh, depending
                 on the column type.
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
    matched_df, unmatched_df - A Pandas DataFrame containing the matched rows
                               followed by a Pandas DataFrame containing
                               unmatched rows.
    '''

    # Define the matches DataFrame.
    matched = pd.DataFrame()


    # Combine our exact columns into one.
    unmatched = df.copy()
    unmatched['__exact__'] = ''

    for col in exact:
        unmatched['__exact__'] += ',' + unmatched[col]


    # Fill missing values with missing string.
    unmatched.fillna('', inplace=True)


    # Loop over unique values of exact columns.
    vals = unmatched['__exact__'].value_counts().reset_index().query('index != "," and __exact__ > 1')['index']

    total = 0
    count = disp
    for val in vals:
        # Make a new DataFrame to house matches on exact.
        temp = unmatched.loc[unmatched['__exact__'] == val, :].reset_index(drop=True)


        # Update total to track our progress.
        total += 1

        # Print our progress.
        if count - round((total/len(vals))*100,2) <= 0:
            print('Progress:', round((total/len(vals))*100,2), '%')
            count += disp


        # Check for discrepancies in the nomismatch columns and fuzzy columns.
        while len(temp) > 0:
            # Get the matching indecies for the first entry.
            fil = temp.apply(RowFilter, axis=1, args=(temp.loc[0],
                nomismatch, fuzzy, strthresh,
                numthresh, allowmiss))

            # Get the matching rows and delete from temp.
            new = temp.loc[fil, :]

            temp = temp.loc[~fil, :].reset_index(drop=True)

            # Move forward if only one observation.
            if len(new) == 1:
                continue


            # Aggragate the rows.
            matched = matched.append(new.apply(RowAgg, axis=0, args=(agg,)), ignore_index=True)


    # Save to the unmatched DataFrame.
    if len(matched) > 0:
        unmatched = unmatched.loc[~(unmatched['__exact__'].isin(matched['__exact__']))]
        matched.drop('__exact__', axis=1, inplace=True)


    # Drop the __exact__ column.
    unmatched.drop('__exact__', axis=1, inplace=True)


    # Return the two DataFrames.
    return matched, unmatched
