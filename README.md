# fuzzylink

This repository contains a set of tools for matching records together when no exact key exists between the two records. These tools include fuzzy matching of records based on string or numerical characteristics (matching.py) and matching of places in records based on geographic distance (distance_matching.py).




## matching.py

This is a Python program implementing fuzzy matching between two sets of records.

It takes a Pandas DataFrame with potential duplicate rows and matches
them based on the columns you specify. It then returns two dataframes,
one with matched observations and one with unmatched observations.
It requires at least one column on which to make an exact match and allows
fuzzy matches based on any other columns.

The matched data will be collapsed (i.e. each row matched with other
rows will be collapsed into one row) based on the aggregation you specify.
If you pass a dictionary into agg of the following form:

    {'mode' : [col1, col2, col3], 'sum' : [col6], 'mean' : [col5, col7]}

it will then apply the different aggregation to each of the columns.
Requires the Python jellyfish, pandas, sys, and time modules.


### Example:

#### Original DataFrame (called df):

|fl|pid|birthdate|deathdate|serial|state|name|firstname|middlename|lastname|url|index|
|--|--|--|--|--|--|--|--|--|--|--|--|
|Fu|||||New York|Achille Funatelle|Achille||Funatelle|http://genealogytrails.com/ny/ww1soldiers.htm|24770|
|Ta|||13 Oct 1918||New York|Abernethy S Taylor|Abernethy|S|Taylor|https://www.honorstates.org/index.php?id=150576|72656|
|Ca|||09 Oct 1918|2721800.0|Vermont|Achille Capute|Achille||Capute|https://catalog.archives.gov/id/34391830|94085|
|Fu|||05 Oct 1918|1706020|New York|Achille Funatelli|Achille||Funatelli|https://catalog.archives.gov/id/34390682|104910|
|Ta|||13 Oct 1918||New York|Abernathy S Taylor|Abernathy|S|Taylor|https://catalog.archives.gov/id/34390682|135266|


#### Code:

    matched, unmatched = Match(df, ['fl'], nomismatch=['pid', 'birthdate', 'deathdate', 'serial', 'state'], 
                           fuzzy=['name', 'firstname', 'middlename', 'lastname'], strthresh={'name' : 0.85,
                                 'firstname' : 0.9, 'middlename' : 0.7, 'lastname' : 0.9}, allowmiss=True,
                           disp=0.1, agg={'mode' : cols, 'all' : ['url', 'index'], 'len' : ['name']})


This code will run over the data, blocking on the column 'fl' (first two letters of last name). It will then check the columns specified in 'nomismatch' (['pid', 'birthdate', 'deathdate', 'serial', 'state']) and eliminate those matches within the block where the columns specified contain mismatching non-empty values. After it will check those columns specified in 'fuzzy' (['name', 'firstname', 'middlename', 'lastname']) and eliminate matches whose jarowinkler score is less than that specified in 'strthresh' (in this case I specified a lower threshold for middle name since some names only have a middle initial). I also set 'allowmiss' to 'True' to allow missing values to still be matches in the jarowinkler. Lastly I aggregate using the mode for all columns except 'url' and 'index', for which I collect all values into a list, and 'name', for which I take the longest value.


#### Results:

##### Matched DataFrame:

|fl|pid|birthdate|deathdate|serial|state|name|firstname|middlename|lastname|url|index|
|--|--|--|--|--|--|--|--|--|--|--|--|
|Fu|||05 Oct 1918|1706020|New York|Achille Funatelli|Achille||Funatelli|['http://genealogytrails.com/ny/ww1soldiers.html', 'https://catalog.archives.gov/id/34390682']|[104910, 24770]|
|Ta|||13 Oct 1918||New York|Abernathy S Taylor|Abernathy|S|Taylor|['https://www.honorstates.org/index.php?id=150576', 'https://catalog.archives.gov/id/34390682']|[135266, 72656]|

##### Unmatched DataFrame:

|fl|pid|birthdate|deathdate|serial|state|name|firstname|middlename|lastname|url|index|
|--|--|--|--|--|--|--|--|--|--|--|--|
|Ca|||09 Oct 1918|2721800.0|Vermont|Achille Capute|Achille||Capute|https://catalog.archives.gov/id/34391830|94085|




## distance_matching.py

This program takes two data sets, each with latitude and longitude for a set of places (in decimal format), and gives the N closest places from the second data set for each place in the first.


### Example:

#### Original DataFrames (called data1 and data2):

data1:

|id1|place_name|lat|lon|
|--|--|--|--|
|0|La Jolla, California|32.841991|-117.273018|
|1|New York, New York|40.713051|-74.007233|
|2|Washington DC|38.907192|-77.036873|

data2:

|id2|place_name|lat|lon|
|--|--|--|--|
|0|La Jolla, California|32.841991|-117.273018|
|1|San Diego, California|32.715760|-117.163820|
|2|Albany, New York|42.651718|-73.755089|
|3|Baltimore, Maryland|39.290897|-76.610406|
|4|Washington DC|38.907192|-77.036873|


#### Code:

	matches = distance_matching.GetNClosest(data1, data2, 
                   ['id1', 'id2'], ['lat', 'lat'],
                   ['lon', 'lon'], nummatches=2,
                   chunksize=len(data1))

This takes every row in data1 and compares the distance to data2, returning the closest 2 places. The chunksize set to len(data1) means that the code will do the entire dataset at once (if data1 and/or data2 are very large this can be set to a smaller number than the full data length to do it in chunks).


#### Results:

##### Matches Crosswalk:

|id1|__matches__|
|--|--|
|0|[0, 1]|
|1|[2, 3]|
|2|[4, 3]|

The id1 column corresponds exactly to that in data1. The __matches__ column contains a list of values from id2 representing, in order closest to furthest, the N closest matches from data2.
