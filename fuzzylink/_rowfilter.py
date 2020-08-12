#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from jellyfish import jaro_winkler
import pandas as pd


###############################################################################
###############################################################################
###############################################################################
def _rowfilter(i, j, arr, idcols, nomismatch, fuzzy,
              strthresh, numthresh, onlycheck, weight, allowmiss):
    '''
    A function to calculate whether two rows are a match. If a comparison does
    not meet the given thresholds (i.e. does not match on the "nomismatch"
    columns or does not meet the threshold specified for string and numeric
    variables) it returns a 0 for the score. This is intended to work with
    np.fromfunction.
    

    Parameters
    ----------
    i : int
        The index of the row to be matched.
    j : int
        The index of the row to be compared.
    arr : dict of np.ndarrays
        Each key is one of the specified columns and all associated values are
        in the corresponding array.
    idcols : dict
        A dictionary with only key "all" containing a list with two values, 
        the id variable for the rows to be matched first and the id variable
        for the comparison pool second. This is necessary to work with
        np.fromfunction.
    nomismatch : dict
        A dictionary with only key "all" containing a list of columns requiring
        no discrepancy in nonmissing values, but will allow a match between
        missing and a value. This is necessary to work with
        np.fromfunction.
    fuzzy : dict
        A dictionary with only key "all" containing a list of columns requiring
        a fuzzy match, with the threshold specified by strthresh or numthresh,
        depending on the column type. This is necessary to work with
        np.fromfunction.
    strthresh : float or dict
        The threshold for Jaro-Winkler score below which the rows are not a 
        match. If a dictionary is passed, it must have one entry per fuzzy 
        column specifying the threshold for each.
    numthresh : float
        The threshold for numeric variables absolute difference outside of 
        which the rows are not a match.
    onlycheck : str
        The name of a column with only True and False, where True signifies
        that the row is to be matched against all other observations and False
        signifies that a row is to be including in the comparison pool for
        other matches but not matched itself. If a column name is specified
        for this variable then there cannot be any cases in the matched sample 
        where only rows with False specified are matched together since those 
        rows will not be explicitly checked against the comparison pool.
    weight : float or dict
        The weight for each column to be applied in calculating the score.       
    allowmiss : bool
        Allow a mismatch in fuzzy due to missing values.
                 
    
    Returns
    -------
    A tuple containing three values, in order: Match Score, id value from the 
    rows to be matched, and id value of the corresponding checked match.
    '''
    # Initialize a variable to keep the score and unpack the column lists.
    score = 0.0 
        
    fuzzy = fuzzy['all']
    nomismatch = nomismatch['all']
    idcols = idcols['all']
    
    
    # Do not check the same row with itself.
    if i == j:
        return (0.0, arr[idcols[0]][i], arr[idcols[1]][j])
    
    
    # If onlycheck is specified, keep if the matching row is not to be
    #  compared or if the comparison row is not in the comparison group.
    if onlycheck != '':
        if arr[onlycheck][i] == False or arr[onlycheck][j] == True:
            return (0.0, arr[idcols[0]][i], arr[idcols[1]][j])
    
    
    # Convert to dictionaries as needed.
    if type(strthresh) != dict and fuzzy != []:
        strthresh = {col : strthresh for col in fuzzy}

    if type(numthresh) != dict and fuzzy != []:
        numthresh = {col : numthresh for col in fuzzy}
        
    if type(weight) != dict and (nomismatch != [] or fuzzy != []):
        weight = {col : weight for col in nomismatch + fuzzy}


    # First check the nomismatch columns.
    for col in nomismatch:
        if pd.isnull(arr[col][i]) or pd.isnull(arr[col][j]):
            pass
        elif str(arr[col][i]).strip() == '' or str(arr[col][j]).strip() == '':
            pass
        else:
            if str(arr[col][i]).strip() != str(arr[col][j]).strip():
                return (0.0, arr[idcols[0]][i], arr[idcols[1]][j])
            
            else:
                score += weight[col]


    # Next check the fuzzy columns.
    for col in fuzzy:
        if type(arr[col][i]) == str:
            if allowmiss and (arr[col][i] == '' or arr[col][j] == '' or\
                pd.isnull(arr[col][i]) or pd.isnull(arr[col][j])):
                pass
            
            elif (arr[col][i] == '' or pd.isnull(arr[col][i])) \
                and (arr[col][j] == '' or pd.isnull(arr[col][j])):
                pass
            
            elif pd.isnull(arr[col][i]) or pd.isnull(arr[col][j]):
                return (0.0, arr[idcols[0]][i], arr[idcols[1]][j])
            
            else:
                jaro = jaro_winkler(arr[col][i].lower(), arr[col][j].lower())
                
                if jaro < strthresh[col]:
                    return (0.0, arr[idcols[0]][i], arr[idcols[1]][j])
                
                else:
                    score += jaro * weight[col]
                    
        else:
            if allowmiss and (pd.isnull(arr[col][i]) or\
                pd.isnull(arr[col][j])):
                pass
            
            elif (arr[col][i] == '' or pd.isnull(arr[col][i])) \
                and (arr[col][j] == '' or pd.isnull(arr[col][j])):
                pass
            
            elif pd.isnull(arr[col][i]) or pd.isnull(arr[col][j]):
                return (0.0, arr[idcols[0]][i], arr[idcols[1]][j])
            
            else:
                dist = abs(arr[col][i] - arr[col][j])
                
                if dist > numthresh[col]:
                    return (0.0, arr[idcols[0]][i], arr[idcols[1]][j])
                
                else:
                    score += (numthresh[col] - dist) * weight[col]


    # If we passed all of the checks, return the score.
    return (score, arr[idcols[0]][i], arr[idcols[1]][j])