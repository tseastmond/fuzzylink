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


# Make a function to filter out rows.
def RowFilter(row, val, nomismatch=[], fuzzy=[], strthresh=0.9,
              numthresh=1, allowmiss=False):
    # Make a dictionary as needed.
    if type(strthresh) != dict:
        strthresh = {: strthresh}

    if type(numthresh) != dict:
        numthresh = {col.name : numthresh}



    # Loop over the nomismatch columns.
    for col in nomismatch:
        if row[col] not in ['', val]:
            return False

    # Loop over the fuzzy columns.
    for col in fuzzy:
        jaro_winkler(row[col], val) < strthresh[col]

# Make a function to filter out rows.
def RowFilter(col, nomismatch=[], fuzzy=[], strthresh=0.9,
              numthresh=1, allowmiss=False):
    if type(strthresh) != dict:
        strthresh = {col.name : strthresh}

    if type(numthresh) != dict:
        numthresh = {col.name : numthresh}

    col = col.reset_index(drop=True)

    if col.name in nomismatch:
        return [col[0] in ['', y] for y in col]

    elif col.name in fuzzy:
        if type(col[0]) == str:
            if allowmiss:
                return [(jaro_winkler(col[0], y) > strthresh[col.name]) or (col[0] == '') or (y == '') for y in col]
            else:
                return [jaro_winkler(col[0], y) > strthresh[col.name] for y in col]

        else:
            if allowmiss:
                return [(abs(col[0] - y) <= numthresh[col.name]) or (pd.isnull(col[0])) or (pd.isnull(y)) for y in col]
            else:
                return [abs(col[0] - y) <= numthresh[col.name] for y in col]



# Make a function to aggregate rows.
def RowAgg(col, agg='mode'):
    if type(agg) != dict:
        agg = {agg : col.name}

    for x in ['mode', 'sum', 'mean', 'count', 'len', 'all']:
        if x not in agg.keys():
            agg[x] = []

    col = col.reset_index(drop=True)

    if col.name in agg['mode']:
        if col.value_counts().index[0] != '' or len(col.value_counts().index) == 1:
            return col.value_counts().index[0]

        else:
            return col.value_counts().index[1]

    elif col.name in agg['sum']:
        return sum(col)

    elif col.name in agg['mean']:
        return np.mean(col)

    elif col.name in agg['count']:
        return len(list(filter(None, col)))

    elif col.name in agg['len']:
        return sorted(col, key=len)[-1]

    elif col.name in agg['all']:
        return list(col.value_counts().index)

    else:
        return col[0]



# Make a function for matching.
def Match(df, exact, nomismatch=[], fuzzy=[], agg='mode', strthresh=0.9,
          numthresh=1, allowmiss=False, disp=5):
    '''
    This takes a Pandas DataFrame with potential duplicate rows and matches
    them based on the columns you specify. It then returns two dataframes,
    one with matched obeservations and one with unmatched observations.
    It requires at least one column on which to make an exact match.

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
        temp = unmatched.loc[unmatched['__exact__'] == val, :]


        # Update total to track our progress.
        total += 1

        # Print our progress.
        if count - round((total/len(vals))*100,2) <= 0:
            print('Progress:', round((total/len(vals))*100,2), '%')
            count += disp


        # Check for discrepancies in the nomismatch columns and fuzzy columns.
        while len(temp) > 0:
            # Get the matching indecies for the first entry.
            fil = temp.apply(RowFilter, axis=0, args=(nomismatch, fuzzy, strthresh)).all(axis=1)

            # Get the matching rows and delete from temp.
            new = temp.loc[fil, :]

            temp = temp.loc[~fil, :]

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
