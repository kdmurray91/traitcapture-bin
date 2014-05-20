# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:52:40 2014

@author: chuong nguyen, chuong.nguyen@anu.edu.au

This script provides functions to read and write YAML format used by OpenCV

Current support: string, int, float, 2D numpy array
"""

from __future__ import absolute_import, division, print_function

import numpy as np

def readValueFromLineYML(line):
    name = line[:line.index(':')].strip()
    string = line[line.index(':')+1:].strip()
    if string[0] in '-+.0123456789':
        if '.' in string:
            value =  float(string)
        else:
            value = int(string)
    else:
        value = string
        
    return name, value

def readOpenCVArrayFromYML(myfile):
    line = myfile.readline().strip()
    rname, rows = readValueFromLineYML(line)
    line = myfile.readline().strip()
    cname, cols = readValueFromLineYML(line)
    line = myfile.readline().strip()
    dtname, dtype = readValueFromLineYML(line)
    line = myfile.readline().strip()
    dname, data = readValueFromLineYML(line)
    if rname != 'rows' and cname != 'cols' and dtname != 'dt' \
        and dname != 'data' and '[' in data:
        print('Error reading YML file')
    elif dtype != 'd':
        print('Unsupported data type: dt = ' + dtype)
    else:
        if ']' not in data:
            while True:
                line = myfile.readline().strip()
                data = data + line
                if ']' in line:
                    break
        data = data[data.index('[')+1: data.index(']')].split(',')
        dlist = [float(el) for el in data]
        if cols == 1:
            value = np.asarray(dlist)
        else:
            value = np.asarray(dlist).reshape([rows, cols])
    return value

def yml2dic(filename):
    with open (filename, 'r') as myfile:
        dicdata = {}
        while True:
            line = myfile.readline()
            if not line:
                break
            
            line = line.strip()
            if len(line) == 0 or line[0] == '#':
                continue
            
            if ':' in line:
                name, value = readValueFromLineYML(line)

                # if OpenCV array, do extra reading
                if isinstance(value, str) and 'opencv-matrix' in value:
                    value = readOpenCVArrayFromYML(myfile)

                # add parameters
                dicdata[name] = value

    return dicdata

def writeOpenCVArrayToYML(myfile, key, data):
    myfile.write(key + ': !!opencv-matrix\n')
    myfile.write('   rows: %d\n' %data.shape[0])
    myfile.write('   cols: %d\n' %data.shape[1])
    myfile.write('   dt: d\n')
    myfile.write('   data: [')
    datalist = []
    for i in range(data.shape[0]):
        datalist = datalist + [str(num) for num in list(data[i,:])]
    myfile.write(', '.join(datalist))
    myfile.write(']\n')
    
def dic2yml(filename, dicdata):
    with open (filename, 'w') as myfile:
        myfile.write('%YAML:1.0\n')
        for key in dicdata:
            data = dicdata[key]
            if type(data) == np.ndarray:
                writeOpenCVArrayToYML(myfile, key, data)
            elif isinstance(data, str):
                myfile.write(key+': "%s"\n' %data)
            elif isinstance(data, int):
                myfile.write(key+': %d\n' %data)
            elif isinstance(data, float):
                myfile.write(key+': %f\n' %data)
            else:
                print('Unsupported data: ', data)
                