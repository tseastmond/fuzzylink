#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from time import time, sleep


###############################################################################
###############################################################################
###############################################################################
def _timer(progress, cores, start_time, disp):
    '''
    A function to print the progress of each process and the code overall.
    

    Parameters
    ----------
    progress : dict
        A dictionary containing the keys 'p1', 'm1', and 'total1' (where "1"
        is replaced with process number), where the values are respectively
        the total number of rows checked, the total number of matches made,
        and the overall total number of rows that will be checked for each
        process.
    cores : int
        The number of processes specified.
    start_time : float
        The time when the code started in seconds since the Epoch.
    disp : int
        The number of second to wait before updating the progress output.


    Returns
    -------
    None.
    '''
    # Ensure that each process has started matching before moving forward.
    while True:
        count = 0
        for x in range(1, cores+1):
            if 'p'+str(x) in progress.keys():
                count += 1
                
        if count == cores:
            break
        
    
    # Loop continuously. The main code will kill this process when finished.
    while True:
        # Wait the specified time and reset the values for display.
        sleep(disp)
        s = ''
        matches = 0
        total = 0
        
        # Calculate the total ellapsed time hours, then convert to minutes if
        #  under 1 hour.
        now = time()
        ellapsed = round((now-start_time)/3600, 2)
        
        if ellapsed < 1:
            ellapsed = round(60*ellapsed)
            
            if ellapsed == 1:
                e1 = 'Minute'
            else:
                e1 = 'Minutes'
        else:
            e1 = 'Hours'
        
        # Collect the information from each process.
        for x in range(1, cores+1):
            try:
                # Get the average time per iteration for the process and
                #  multiply by the number of rows left, converting to hours.
                t1 = round((((now-start_time)/progress['p'+str(x)]) *\
                            (progress['tot'+str(x)] -\
                             progress['p'+str(x)]))/3600,2)
            except:
                t1 = 9999
            
            if t1 < 1:
                t1 = round(60*t1)
                
                if t1 == 1:
                    t2 = 'Minute'
                else:
                    t2 = 'Minutes'
            else:
                t2 = 'Hours'
            
            # Append into one string for display.
            s1 = 'Process ' + str(x) + ':' + ' '*(3-len(str(x))) + '[' + \
            '='*round(40*(progress['p'+str(x)]/progress['tot'+str(x)])) +' '*\
            (40 - round(40*(progress['p'+str(x)]/progress['tot'+str(x)]))) +\
            '] ' + str(round(100*(progress['p'+str(x)]/\
                                  progress['tot'+str(x)]), 1)) + '%  -  ' +\
            'Est. ' + str(t1) + ' ' + t2 + ' Remaining\n\n'
            
            s = ''.join([s, s1])
            
            # Get the total number of matches and checked rows.
            matches += progress['m'+str(x)]
            total += progress['p'+str(x)]
        
        # Avoid division by zero if processes are slow.
        if total == 0:
            total = 1
                
        # Clear the screen and print the output. This includes (1) the process
        #  number, (2) a loading bar with associated percent complete, (3)
        #  estimated time to completion, (4) overall number of matches and rows
        #  checked, and (5) the total ellapsed time.
        os.system('clear')
        print(s + 'Total Matches:  ' + str(matches) +\
              ' of {0} (~{1}%)\n\n'.format(total, round(100*matches/total, 1))\
              + 'Elapsed Time:   ' + str(ellapsed) + ' ' + e1 + '\n\n')
