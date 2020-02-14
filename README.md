# FuzzyMatching
A Python program implementing fuzzy matching between two sets of records.

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
Requires the Python jellyfish, pandas, sys, and time modules.