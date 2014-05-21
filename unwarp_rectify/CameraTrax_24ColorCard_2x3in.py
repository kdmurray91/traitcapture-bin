# -*- coding: utf-8 -*-
"""
Created on Wed May 21 14:09:56 2014

@author: chuong
"""
from __future__ import absolute_import, division, print_function


import cv2
import numpy as np
from matplotlib import pylab as plt

#                            RED GRN BLU      NAME
CameraTrax_24ColorCard = [  [115, 83, 68],\
                            [196,147,127],\
                            [ 91,122,155],\
                            [ 94,108, 66],\
                            [129,128,176],\
                            [ 98,190,168],\
                            [223,124, 47],\
                            [ 72, 92,174],\
                            [194, 82, 96],\
                            [ 93, 60,103],\
                            [162,190, 62],\
                            [229,158, 41],\
                            [ 49, 66,147],\
                            [ 77,153, 71],\
                            [173, 57, 60],\
                            [241,201, 25],\
                            [190, 85,150],\
                            [  0,135,166],\
                            [242,243,245],\
                            [203,203,204],\
                            [162,163,162],\
                            [120,120,120],\
                            [ 84, 84, 84],\
                            [ 50, 50, 52]]
ColourNames = ['DrkTone', 'LtTone', 'SkyBlue', 'Tree-Grn', 'LtBlu', 'Blu-Grn', \
               'Orange', 'MedBlu', 'LtRed', 'Purple', 'Yel-Grn', 'Org-Grn', \
               'Blue', 'Green', 'Red', 'Yellow', 'Magenta', 'Cyan', \
               'White', 'LtGrey', 'Grey', 'DrkGrey', 'Charcoal', 'Black']
SquareSize = 50 # pixels                            
P24ColorCard = np.zeros([SquareSize*4, SquareSize*6, 3], dtype = np.uint8)
for i,Colour in enumerate(CameraTrax_24ColorCard):
    R,G,B = Colour
    Row = int(i/6)
    Col = i - Row*6
    P24ColorCard[Row*SquareSize:(Row+1)*SquareSize, Col*SquareSize:(Col+1)*SquareSize, 0] = R
    P24ColorCard[Row*SquareSize:(Row+1)*SquareSize, Col*SquareSize:(Col+1)*SquareSize, 1] = G
    P24ColorCard[Row*SquareSize:(Row+1)*SquareSize, Col*SquareSize:(Col+1)*SquareSize, 2] = B

P24ColorCard_BGR = P24ColorCard[:,:,::-1]
cv2.imwrite('CameraTrax_24ColorCard_2x3in.png', P24ColorCard_BGR)
P24ColorCard_BGR2 = cv2.imread('CameraTrax_24ColorCard_2x3in.png')
if (P24ColorCard_BGR2 != P24ColorCard_BGR).any():
    print('Output image is not the same as the actual image')

plt.imshow(P24ColorCard)
plt.show()
