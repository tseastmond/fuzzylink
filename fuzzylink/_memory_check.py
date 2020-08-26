#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import kill
from psutil import virtual_memory
from time import sleep


###############################################################################
###############################################################################
###############################################################################
def _memory_check(processes, output):
    '''
    A function to break the program if overall memory usage is too high.
    

    Parameters
    ----------
    processes : list
        A list of all subprocesses pids generated by the main program.
    output : dict
        A dictionary shared by processes to return output.    


    Returns
    -------
    None.
    '''
    # Loop indefinitely, breaking only if memory usage is too high.
    while True:
        sleep(.1)
        
        if virtual_memory().percent > 98:
            # Communicate the error to the main program.
            output['end'] = True

            # Kill all spawned processes.
            for p in processes:
                try:
                    kill(p, 9)
                except:
                    pass
            
            break