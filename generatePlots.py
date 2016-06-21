import math
import os
import sys
import numpy as np
import scipy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


def parse_data(data_file):
    d = defaultdict(list)
    reader = csv.reader(data_file)
    headers = map(str.strip, reader.next())
    for line in reader:
        for index, column in enumerate(line.split(',')):
            column = float(column.strip())
            d[headers[index]].append(column)

    return d

with open('results.txt') as f:
    data = parse_data(f)
