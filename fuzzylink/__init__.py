#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__title__ = 'fuzzylink'
__version__ = '1.0.0'
__author__ = 'Tanner S Eastmond'
__contact__ = 'https://github.com/tseastmond'
__license__ = 'MIT'

from multiprocessing import Process, Manager
import os
import sys
from time import time

from jellyfish import jaro_winkler
import numpy as np
import pandas as pd

from .distance_matching import GetNClosest
from .matching import Match
