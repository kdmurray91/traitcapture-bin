# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:15:55 2014

@author: chuong nguyen, chuong.nguyen@anu.edu.au

This script is to provide distortion correction and 
image rectification with support as generator or function.
"""
from __future__ import absolute_import, division, print_function

import os, sys, glob
from multiprocessing import Pool
import numpy as np
import cv2
import cv2yml # support input/output of OpenCV's YAML file format
from matplotlib import pylab as plt

# Supporting functions
def rotateImage(Image, Angle):
    Center=tuple(np.array(Image.shape[0:2])/2)
    RotationMatrix = cv2.getRotationMatrix2D(Center, Angle, 1.0)
    return cv2.warpAffine(Image, RotationMatrix, Image.shape[0:2], flags=cv2.INTER_LINEAR)
    
def rectifyTrayImages(Image, RectList, TrayImgSize):
    Width, Height = TrayImgSize
    RectifiedCorners = np.float32([[0,0], [0,Height], [Width,Height], [Width,0]])
    RectifiedTrayImageList = []
    for Rect in RectList:
        Corners = np.float32(Rect)
        M = cv2.getPerspectiveTransform(Corners, RectifiedCorners)
        RectifiedTrayImage = cv2.warpPerspective(Image, M,(Width, Height))
        RectifiedTrayImageList.append(RectifiedTrayImage)
    return RectifiedTrayImageList

def joinTrayImages(TrayImageList, Shape = [2,4]):
    if len(TrayImageList) == 0 or len(TrayImageList) != Shape[0]*Shape[1]:
        print('Invalid inputs: ', TrayImageList, Shape)
        return np.asarray([])
        
    TrayShape = TrayImageList[0].shape
    RectifiedImage = np.resize(np.zeros_like(TrayImageList[0]), (Shape[0]*TrayShape[0], Shape[1]*TrayShape[1], TrayShape[2]))
    for r in range(Shape[0]):
        for c in range(Shape[1]):
            rr0 = (Shape[0]-r-1)*TrayShape[0]
            rr1 = (Shape[0]-r)*TrayShape[0]
            cc0 = c*TrayShape[1]
            cc1 = (c+1)*TrayShape[1]
            RectifiedImage[rr0:rr1, cc0:cc1, :] = TrayImageList[c+r*Shape[1]]
    return RectifiedImage

def readCalibFile(CalibFile):
    print('  Read', CalibFile)
    parameters = cv2yml.yml2dic(CalibFile)
    print('  This file created on', parameters['calibration_time'])
    SquareSize = parameters['square_size']
    ImageWidth = parameters['image_width']
    ImageHeight = parameters['image_height']
    ImageSize = (ImageWidth, ImageHeight)
    CameraMatrix = parameters['camera_matrix']
    DistCoefs = parameters['distortion_coefficients']
    RVecs = parameters['RVecs']
    TVecs = parameters['TVecs']
    return ImageSize, SquareSize, CameraMatrix, DistCoefs, RVecs, TVecs

def readTrayConfigFile(ConfigFile):
    print('  Read', ConfigFile)
    dicdata = cv2yml.yml2dic(ConfigFile)
    print('  This file created on', dicdata['Date'])
    TrayPixWidth = dicdata['TrayImgWidth']
    TrayPixHeight = dicdata['TrayImgHeight']
    RectList2 = dicdata['TrayRectList'].tolist()
    RectList = []
    for i in range(0,len(RectList2),4):
        RectList.append(RectList2[i:i+4])
    return RectList, TrayPixWidth, TrayPixHeight

# Using the generator pattern (an iterable)
class UndistortionRectificationGenerator(object):
    def __init__(self, \
        FileNameList = [], \
        CameraMatrix = np.asarray([]), DistortionCoeffients = np.asarray([]), \
        ImageSize = (), RotationAngle = 0, \
        TrayRectangleList = [], TrayImgSize = [1230, 1489], TrayArrangement = [2,4]):

        self.FileNameList = FileNameList
        self.CameraMatrix = CameraMatrix
        self.DistortionCoeffients = DistortionCoeffients
        self.ImageSize = ImageSize
        self.RotationAngle = RotationAngle
        self.TrayRectangleList = TrayRectangleList
        self.TrayImgSize = TrayImgSize
        self.TrayArrangement = TrayArrangement
        self.n = len(self.FileNameList)
        self.num = 0
        
        if len(self.CameraMatrix) > 0 and len(self.DistortionCoeffients) > 0 and len(self.ImageSize) > 0:
            self.MapX, self.MapY = cv2.initUndistortRectifyMap(self.CameraMatrix, \
                self.DistortionCoeffients, None, self.CameraMatrix, self.ImageSize, cv2.CV_32FC1)
            
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n and os.path.exists(self.FileNameList[self.num]):
            print('  Read', self.FileNameList[self.num])
            Image = cv2.imread(self.FileNameList[self.num])
            self.num = self.num+1

            if len(self.CameraMatrix) > 0 and len(self.DistortionCoeffients) > 0 and len(self.ImageSize) > 0:
                ImageUndistorted = cv2.remap(Image, self.MapX, self.MapY, cv2.INTER_CUBIC)
            else:
                ImageUndistorted = Image
            
            if abs(self.RotationAngle) == 180:
                ImageUndistorted = np.rot90(np.rot90(ImageUndistorted))
            elif self.RotationAngle != 0:
                ImageUndistorted = rotateImage(ImageUndistorted, self.RotationAngle) 

            RectifiedTrayImageList = rectifyTrayImages(ImageUndistorted, self.TrayRectangleList, self.TrayImgSize)
            RectifiedImage = joinTrayImages(RectifiedTrayImageList, self.TrayArrangement)
            
            return RectifiedImage
        else:
            raise StopIteration()


# Using normal function with possible parallism 
def UndistortionRectificationFunction(Arg):
    FileNameIn, FileNameOut, MapX, MapY, RotationAngle, \
        TrayRectangleList, MedianTraySize, TrayArrangement = Arg
        
    print('  Read', FileNameIn)
    Image = cv2.imread(FileNameIn)
    if len(MapX) > 0 and len(MapY) > 0:
        ImageUndistorted = cv2.remap(Image, MapX, MapY, cv2.INTER_CUBIC)
    else:
        ImageUndistorted = Image
    
    if abs(RotationAngle) == 180:
        ImageUndistorted = np.rot90(np.rot90(ImageUndistorted))
    elif RotationAngle != 0:
        ImageUndistorted = rotateImage(ImageUndistorted, RotationAngle) 

    RectifiedTrayImageList = rectifyTrayImages(ImageUndistorted, TrayRectangleList, MedianTraySize)
    RectifiedImage = joinTrayImages(RectifiedTrayImageList, TrayArrangement)
    cv2.imwrite(FileNameOut, RectifiedImage)
    print('  Wrote', FileNameOut)

# Usage demos
def demo_parallel(ImageFileListIn, ImageFileListOut, CameraMatrix, DistortionCoeffients, \
    ImageSize, RotationAngle, TrayRectangleList, TrayImgSize, TrayArrangement):
        
    MapX, MapY = cv2.initUndistortRectifyMap(CameraMatrix, DistortionCoeffients, \
                    None, CameraMatrix, ImageSize, cv2.CV_32FC1)
    ArgList = [[ImageFileIn, ImageFileOut, MapX, MapY, RotationAngle, TrayRectangleList, TrayImgSize, TrayArrangement] \
                for ImageFileIn, ImageFileOut in zip(ImageFileListIn, ImageFileListOut)]

    # actual processing    
    ProcessPool = Pool()
    ProcessPool.map(UndistortionRectificationFunction, ArgList)

def demo_generator(ImageFileListIn, ImageFileListOut, CameraMatrix, DistortionCoeffients, \
    ImageSize, RotationAngle, TrayRectangleList, TrayImgSize, TrayArrangement):
        
    UndistRectGen = UndistortionRectificationGenerator(ImageFileListIn, \
                        CameraMatrix, DistortionCoeffients, ImageSize, \
                        RotationAngle, TrayRectangleList, TrayImgSize, TrayArrangement)
    for i,RectifiedImage in enumerate(UndistRectGen):
        plt.imshow(RectifiedImage)
        plt.show()
        cv2.imwrite(ImageFileListOut[i], RectifiedImage)
        print('  Wrote', ImageFileListOut[i])
    
if __name__ == "__main__":
    options = sys.argv[1:] # ignored for now
    
    # inputs data
    CalibFile = '/home/chuong/Data/Calibration-Images/calib_parameters.yml'
    ConfigFile = '/home/chuong/Data/GC03L-temp/corrected/TrayConfig.yml'
    InputImagePattern = '/home/chuong/Data/GC03L-temp/IMG*JPG'
    OutputFolder = '/home/chuong/Data/GC03L-temp/Rectified'
    RotationAngle = 180.0 # so that chamber door is on bottom side of images
    TrayArrangement = [2,4] # 2-rows by 4-columns tray arrangement
    
    # data preparation
    if not os.path.exists(OutputFolder):
        os.mkdir(OutputFolder)
    ImageFileListIn = sorted(glob.glob(InputImagePattern))
    ImageFileListOut = [os.path.join(OutputFolder, os.path.basename(ImageFile)) for ImageFile in ImageFileListIn]
    ImageSize, SquareSize, CameraMatrix, DistortionCoeffients, RVecs, TVecs = readCalibFile(CalibFile)
    TrayRectangleList, TrayImgWidth, TrayImgHeight = readTrayConfigFile(ConfigFile)
    
    # This can be set to a fixed value so that rectified tray image sizes 
    # are the same for different experiments
    TrayImgSize = [TrayImgWidth, TrayImgHeight] # [1230, 1489]
    
    # Demos
#    demo_parallel(ImageFileListIn, ImageFileListOut, CameraMatrix, DistortionCoeffients, 
#                  ImageSize, RotationAngle, TrayRectangleList, TrayImgSize, TrayArrangement)
    demo_generator(ImageFileListIn, ImageFileListOut, CameraMatrix, DistortionCoeffients, 
                  ImageSize, RotationAngle, TrayRectangleList, TrayImgSize, TrayArrangement)