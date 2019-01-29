'''
Created on 18/06/2015

Detects and crops the face from the image.

@author: Alexandre Yukio Yamashita
'''

from argparse import ArgumentParser
import math
import numpy

import cv2 as cv
from image import Image
import point


def convertToMatrix(X, rowOrder=True):
    '''Receives a list of vectors X and make a matrix.
    if rowOrder == True, each vector in the list will be a row of the matrix
    otherwise, each vector will be a column of the matrix
    '''
    if rowOrder:
        return numpy.vstack(X)
    else:
        return numpy.vstack(X).T

def distance(a, b):
    return numpy.linalg.norm(numpy.asarray(a) - numpy.asarray(b))

def calculateRotation(point2D, rotationMatrix2D):
    resultX = rotationMatrix2D[0][0] * point2D[0] + rotationMatrix2D[0][1] * point2D[1] + rotationMatrix2D[0][2] 
    resultY = rotationMatrix2D[1][0] * point2D[0] + rotationMatrix2D[1][1] * point2D[1] + rotationMatrix2D[1][2]
    
    return [resultX, resultY]

###################################
# Cropping and alignment parameters

DEFAULT_TARGET_SIZE = 200
DEFAULT_TARGET_EYEW_RATIO = 0.47
DEFAULT_OFFSET_X = 0.0
DEFAULT_OFFSET_Y = -0.7

# XML files for Haar cascade
DEFAULT_XML_FACE = 'config/haarcascade_frontalface_alt2.xml'
DEFAULT_XML_EYE_PAIR = 'config/haarcascade_mcs_eyepair_big.xml'
DEFAULT_XML_EYE_LEFT = 'config/haarcascade_lefteye_2splits.xml'
DEFAULT_XML_EYE_RIGHT = 'config/haarcascade_righteye_2splits.xml'

# : What is used when the image must be offset too far? 0 for black border, 1 for stretch colors
GAP_BORDER = 1

######################
# Debugging parameters

# : If True, print debug info
DEBUG = True

# : If true, will mark on the image the eyes/eyepairs which were selected to be used for calculations
MARKUSED = False

# : If true, will mark all eyes/eyepairs on the image
MARKALL = False

# : If true, don't perform the scale, offset, rotation (useful for debugging with MARKALL)
NOTRANSFORM = False

# : If true, skip individual eye/eyepair detection, and go to face detection
FORCE_FULL_FACE = False

########################
# Feature marking colors

# : The color used to mark eyepairs
EYEPAIR_COLOR = (255, 0, 0)

# : The color used to mark left eyes
LEFT_EYE_COLOR = (0, 255, 0)

# : The color used to mark right eyes
RIGHT_EYE_COLOR = (0, 0, 255)

# : The color used to mark faces
FACE_COLOR = (0, 255, 255)

# : The color used to mark eye/eyepair center points
MIDPOINT_COLOR = (100, 100, 100)

#######################################################
# Face characteristics, may need to be tweaked per face

# : An eyepair is probably valid with this width/height ratio
EYEPAIR_RATIO = 2

# : The minimum distance threshold for left/right eye. Usually just necessary to
# ensure that detected left/right eyes w/o eyepair are not the same eye
EYE_MIN_DISTANCE = 0.05

# : Conversion factor from the height of a detected face to the eyes midpoint.
# Used when falling back on face detection from eye detection
FACE_HEIGHT_TO_EYE_MID = 0.45

# : Conversion factor from the width of a face to the eye width.
# Used when falling back on face detection from eye detection
FACE_WIDTH_TO_EYE_WIDTH = 0.45

# : The minimum size detection threshold for eyepair as a fraction of the image size
EYEPAIR_MIN_SIZE = (0.15, 0.03)

# : The maximum size detection threshold for eyepair as a fraction of the image size
EYEPAIR_MAX_SIZE = (0.55, 1)

#############################################################
# You probably don't need to change anything below this point

# : Reject a left/right pair of eyes if one is larger by this factor or more
EYE_MAX_SIZE_DIFFERENCE = 2


class Point:

    def __init__(self, *args):
        if len(args) == 1:
            self.x = args[0][0]
            self.y = args[0][1]
        elif len(args) == 2:
            self.x = args[0]
            self.y = args[1]

    def dist(self, p1):
        '''
        Returns the pythagorean distance from this point to p1.
        '''
        return pow(pow(self.x - p1.x, 2) + pow(self.y - p1.y, 2), .5)

    def toTuple(self):
        return (self.x, self.y)

    def __str__(self):
        return '({0}, {1})'.format(self.x, self.y)

    def __repr(self):
        return self.__str__()


class Size:

    def __init__(self, *args):
        if len(args) == 1:
            self.w = len(args[0][0])
            self.h = len(args[0])
        elif len(args) == 2:
            self.w = args[0]
            self.h = args[1]

    def toTuple(self):
        return (self.w, self.h)

    def __str__(self):
        return '({0}, {1})'.format(self.w, self.h)

    def __repr(self):
        return self.__str__()


class Rect:

    def __init__(self, array):
        self.x = array[0]
        self.y = array[1]
        self.w = array[2]
        self.h = array[3]
        self.a = self.w * self.h
        self.center = Point(self.x + self.w / 2.0, self.y + self.h / 2.0)

    def contains(self, p):
        return self.x <= p.x <= self.x + self.w and \
            self.y <= p.y <= self.y + self.h

    def vsplit(self):
        lRect = Rect((self.x, self.y, self.w / 2.0, self.h))
        rRect = Rect((self.center.x, self.y, self.w / 2.0, self.h))
        return lRect, rRect

    def __str__(self):
        return '({0}, {1}), ({2}, {3}), w = {4}, h = {5}, a = {6}'.format(
            self.x, self.y, self.x + self.w, self.y + self.h, self.w, self.h, self.a
        )

    def __repr(self):
        return self.__str__()


class FaceImage:
    '''
    Represents an image with a face in it, and all the scaling/cropping that
    goes along with it.
    '''

    def __init__(self, image, xml_face=DEFAULT_XML_FACE, xml_eye_pair=DEFAULT_XML_EYE_PAIR, xml_eye_left=DEFAULT_XML_EYE_LEFT, xml_eye_right=DEFAULT_XML_EYE_RIGHT):
        self.image = image.data
        self.origSize = Size(self.image)
        self.log = ''
        self._finalImg = None
        self._xml_face = xml_face
        self._xml_eye_pair = xml_eye_pair
        self._xml_eye_left = xml_eye_left
        self._xml_eye_right = xml_eye_right
       
    def cropToFace(self, target_size=DEFAULT_TARGET_SIZE, eyew_ratio=DEFAULT_TARGET_EYEW_RATIO, offset_x=DEFAULT_OFFSET_X, offset_y=DEFAULT_OFFSET_Y, **kwargs):
      """ Finds the face position of the OpenCV image, scales so that the face is the 'ideal'
        size, then crops so that the face is in the center """
      EYEW_TARGET = eyew_ratio * target_size
      # : The target face midpoint coords:image ratio
      
      # Calculate middle.
      MID_X_TARGET_RATIO = .50
      MID_Y_TARGET_RATIO = .45
      correct_scale = 0.6
      
      # : The target x and y-components of the position of the midpoint of the face
      MID_X_TARGET = target_size * MID_X_TARGET_RATIO
      MID_Y_TARGET = target_size * MID_Y_TARGET_RATIO

      if NOTRANSFORM:
        return self.image
      
      eyepair = None
      lEye = rEye = None      
      
      if not FORCE_FULL_FACE:
          eyepair = self._getEyePair()
          lEye, rEye = self._getEyes(eyepair)
      
      # Find the middle of the eyes
      if lEye is not None and rEye is not None and eyepair is not None:
          eyeAngle = math.degrees(
              math.atan((rEye.center.y - lEye.center.y) / (rEye.center.x - lEye.center.x)))
          
            # Rotate
          if eyeAngle == 0:
              rotatedImage = self.image
              
          else:
              self._log('Rotating to: ' + str(eyeAngle))
              rotMatrix = cv.getRotationMatrix2D((MID_X_TARGET, MID_Y_TARGET), eyeAngle, 1)
              rotatedImage = cv.warpAffine(self.image, rotMatrix, (self.image.shape[1], self.image.shape[0]))
              
              rEyeRotated = calculateRotation([rEye.center.x, rEye.center.y], rotMatrix)
              rEye.center.x = rEyeRotated[0]
              rEye.center.y = rEyeRotated[1]
              
              lEyeRotated = calculateRotation([lEye.center.x, lEye.center.y], rotMatrix)
              lEye.center.x = lEyeRotated[0]
              lEye.center.y = lEyeRotated[1]
              
          eyewidth = rEye.center.dist(lEye.center)
          mid = Point(rEye.center.x / 2.0 + lEye.center.x / 2.0,
                      rEye.center.y / 2.0 + lEye.center.y / 2.0)
              
          self._log('', 1)
          self._log('Eye mid at: ' + str(mid) + ', should be: ' + str(Point(MID_X_TARGET, MID_Y_TARGET)), 1)     
      
          # Calculate scaling params
          scaleF = EYEW_TARGET * correct_scale / eyewidth
          scSize = Size(int(self.origSize.w * scaleF), int(self.origSize.h * scaleF))
          scMid = Point(mid.x * scaleF, mid.y * scaleF)
          self._log('Eye width: ' + str(eyewidth) + ', should be: ' + str(EYEW_TARGET), 1)
          self._log('Scale factor: ' + str(scaleF), 1)
          self._log('Pre-crop scaled size: ' + str(scSize), 1)
      
          # Scale image
          scImg = cv.resize(rotatedImage, (scSize.w, scSize.h), interpolation=cv.INTER_LANCZOS4)
      
          # Determine translation. offset: (positive leaves a top/left border, negative doesn't)
          self._log('Scaled midpoint: ' + str(scMid), 1)
          self._log('Target midpoint: ' + str(Point(MID_X_TARGET, MID_Y_TARGET)), 1)
          offset = Point(int(MID_X_TARGET - scMid.x), int(MID_Y_TARGET - scMid.y))
          self._log("offset: " + str(offset), 1)
          self._finalImg = _crop(scImg, offset, Size(target_size, target_size))
      else:
          eyeAngle = 0
      
          self._log(', falling back on face')
          face = self._getFace()
          
          if face is None:
            height = len(self.image) 
            width = len(self.image[0])
            crop_origin_x = 0
            crop_origin_y = 0
            
            if width > target_size and height > target_size:
                crop_origin_x = (width - target_size) / 2
                crop_origin_y = (height - target_size) / 2 
                
                image = Image(image=self.image)
                image = image.crop(point.Point(crop_origin_x, crop_origin_y), 
                                   point.Point(crop_origin_x + target_size -1, crop_origin_y + target_size -1))
                return image.data
            elif width == height:
                resize = int(height* 0.7)
                crop_origin_x = (width - resize) / 2
                crop_origin_y = (height - resize) / 2
                image = Image(image=self.image)
                image = image.crop(point.Point(crop_origin_x, crop_origin_y), 
                                   point.Point(crop_origin_x + resize -1, crop_origin_y + resize -1))
                
                return cv.resize(image.data, (target_size, target_size), interpolation=cv.INTER_LANCZOS4)
            else:
                if width > height:
                    resize = height 
                else: 
                    resize = width
                
                image = Image(image=self.image)
                if width > height:
                    crop_origin_x = (width - resize) / 2
                    print crop_origin_x
                    self.image = image.crop(point.Point(crop_origin_x, 0),
                                            point.Point(crop_origin_x + resize -1, resize -1))
                    return cv.resize(self.image.data, (target_size, target_size), interpolation=cv.INTER_LANCZOS4)
                else:
                    crop_origin_y = (height - resize) / 2
                    
                    self.image = image.crop(point.Point(0, crop_origin_y),
                                            point.Point(resize -1, crop_origin_y + resize -1))
                    return cv.resize(self.image.data, (target_size, target_size), interpolation=cv.INTER_LANCZOS4)                  
                
          mid = Point(face.center.x, face.h * FACE_HEIGHT_TO_EYE_MID + face.y)
          eyewidth = face.w * FACE_WIDTH_TO_EYE_WIDTH
          
          if MARKUSED or MARKALL:
              self._markPoint(mid, MIDPOINT_COLOR)
      
          self._log('', 1)
          self._log('Eye mid at: ' + str(mid) + ', should be: ' + str(Point(MID_X_TARGET, MID_Y_TARGET)), 1)     
      
          # Calculate scaling params
          scaleF = EYEW_TARGET * 1.2 * correct_scale / eyewidth
          scSize = Size(int(self.origSize.w * scaleF), int(self.origSize.h * scaleF))
          scMid = Point(mid.x * scaleF, mid.y * scaleF * 0.9)
          self._log('Eye width: ' + str(eyewidth) + ', should be: ' + str(EYEW_TARGET), 1)
          self._log('Scale factor: ' + str(scaleF), 1)
          self._log('Pre-crop scaled size: ' + str(scSize), 1)
      
          # Scale image
          scImg = cv.resize(self.image, (scSize.w, scSize.h), interpolation=cv.INTER_LANCZOS4)
      
          # Determine translation. offset: (positive leaves a top/left border, negative doesn't)
          self._log('Scaled midpoint: ' + str(scMid), 1)
          self._log('Target midpoint: ' + str(Point(MID_X_TARGET, MID_Y_TARGET)), 1)
          offset = Point(int(MID_X_TARGET - scMid.x), int(MID_Y_TARGET - scMid.y))
          self._log("offset: " + str(offset), 1)
          translatedScaledImage = _crop(scImg, offset, Size(target_size, target_size))
      
          # Rotate
          if eyeAngle == 0:
              self._finalImg = translatedScaledImage
          else:
              self._log('Rotating to: ' + str(eyeAngle))
              rotMatrix = cv.getRotationMatrix2D((MID_X_TARGET, MID_Y_TARGET), eyeAngle, 1)
              self._finalImg = cv.warpAffine(translatedScaledImage, rotMatrix, (target_size, target_size))
          
      return self._finalImg
    
    def _getEyePair(self):
        cascade = cv.CascadeClassifier(self._xml_eye_pair)
        minSize = (int(EYEPAIR_MIN_SIZE[0] * self.origSize.w),
                   int(EYEPAIR_MIN_SIZE[1] * self.origSize.h))
        maxSize = (int(EYEPAIR_MAX_SIZE[0] * self.origSize.w),
                   int(EYEPAIR_MAX_SIZE[1] * self.origSize.h))
        eyepairs = _toRects(cascade.detectMultiScale(self.image, minSize=minSize, maxSize=maxSize))

        if not eyepairs:
            return None

        for eyepair in eyepairs:
            self._log('Eyepair found: ' + str(eyepair), 1)
            if MARKALL:
                self._markRect(eyepair, EYEPAIR_COLOR)

        # Find the largest eyepair
        largest = max(eyepairs, key=lambda e: e.a)

        if largest.w / largest.h < EYEPAIR_RATIO:
            return None
        else:
            if MARKUSED:
                self._markRect(largest, EYEPAIR_COLOR)
            return largest

    def _getEyes(self, eyepair):
        lEyeCascade = cv.CascadeClassifier(self._xml_eye_left)
        rEyeCascade = cv.CascadeClassifier(self._xml_eye_right)

        lEyes = _toRects(lEyeCascade.detectMultiScale(self.image))
        rEyes = _toRects(rEyeCascade.detectMultiScale(self.image))

        # mark eyes if needed
        for eye in lEyes:
            self._log('Left eye found: ' + str(eye), 1)
            if MARKALL:
                self._markRect(eye, LEFT_EYE_COLOR)

        for eye in rEyes:
            self._log('Right eye found: ' + str(eye), 1)
            if MARKALL:
                self._markRect(eye, RIGHT_EYE_COLOR)

        if len(lEyes) == 0 or len(rEyes) == 0:
            self._log('Didn\'t find both left and right eyes')
            return (None, None)

        # Filter eye results by having centers in the correct half of the eyepair
        if eyepair:
            rightEyepair, leftEyepair = eyepair.vsplit()
            self._log(str(leftEyepair))
            self._log(str(rightEyepair))
            lEyes = filter(lambda e: leftEyepair.contains(e.center), lEyes)
            rEyes = filter(lambda e: rightEyepair.contains(e.center), rEyes)

            if (len(lEyes) == 0 or len(rEyes) == 0):
                self._log('Didn\'t find eyes in the correct half of the eyepair')
                return (None, None)

        lEye = max(lEyes, key=lambda e: e.a)
        rEye = max(rEyes, key=lambda e: e.a)

        # Throw out the eyes if they are too close
        eyeDist = lEye.center.dist(rEye.center)
        minEyeDist = EYE_MIN_DISTANCE * self.origSize.w
        if eyeDist < minEyeDist:
            self._log('Eyes too close, rejected - %d, should be %d' % (eyeDist, minEyeDist))
            return (None, None)

        # Throw out the eyes if they differ in size too much
        eyeSizeDiff = max(lEye.a, rEye.a) / min(lEye.a, rEye.a)
        if eyeSizeDiff >= EYE_MAX_SIZE_DIFFERENCE:
            self._log('Eyes too different in size, rejected - %d vs %d' % (lEye.a, rEye.a))
            return (None, None)

        if MARKUSED:
            self._markRect(lEye, LEFT_EYE_COLOR)
            self._markRect(rEye, RIGHT_EYE_COLOR)

        return (lEye, rEye)

    def _getFace(self):
        ''' Returns coordinates of the face in this image '''
        cascade = cv.CascadeClassifier(self._xml_face)
        faces = _toRects(cascade.detectMultiScale(self.image))

        for face in faces:
            self._log('Face found: ' + str(face), 1)
            if MARKALL:
                self._markRect(face, FACE_COLOR)

        if len(faces) > 0:
            bestFace = faces[0]
            for face in faces:
                bestFace = self._bestFace(bestFace, face)

            if MARKUSED:
                self._markRect(bestFace, FACE_COLOR)

            return bestFace
        else:
            return None

    def _bestFace(self, f1, f2):
        # if the sizes of these faces are within .5% of each other, take the
        # one nearest midpoint
        p = .005
        deltaP = float(abs(f1.a - f2.a)) / max(f1.a, f2.a)
        imageMidpoint = Point(self.origSize.w / 2, self.origSize.h / 2)
        if deltaP < p:
            return f1 if f1.center.dist(imageMidpoint) < f2.center.dist(imageMidpoint) else f2
        else:
            return max(f1, f2, key=lambda f: f.a)

    def _markRect(self, rect, color):
        ''' Marks the location of the given rect onto the image '''
        cv.rectangle(self.image, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), color)
        self._markPoint(rect.center, MIDPOINT_COLOR)

    def _markPoint(self, p, color):
        pointSize = 10
        cv.rectangle(
            self.image,
            (int(p.x) - pointSize / 2, int(p.y) - pointSize / 2),
            (int(p.x) + pointSize / 2, int(p.y) + pointSize / 2),
            color,
            cv.cv.CV_FILLED)

    def _log(self, msg, level=0):
        if DEBUG:
            self.log += '  ' * level + str(msg) + '\n'

def crop_face(image, size):
    fi = FaceImage(image)
    return Image(image=fi.cropToFace(size))

def _toRects(cvResults):
    return [Rect(result) for result in cvResults]

def _crop(image, offset, size):
    imageSize = Size(image)

    # If there will be a border, use CopyMakeBorder.
    # Setting ROI, no border is created and resulting image is smaller
    if offset.x > 0 or \
       offset.y > 0 or \
       offset.x + imageSize.w < size.w or \
       offset.y + imageSize.h < size.h:

        # offset may have negative values, if there will be a right/bottom border
        offsTop = offset.y
        offsBottom = -(offset.y + imageSize.h - size.h)
        offsLeft = offset.x
        offsRight = -(offset.x + imageSize.w - size.w)

        image = image[max(0, -offset.y):min(-offset.y + size.h, imageSize.h), max(0, -offset.x):min(-offset.x + size.w, imageSize.w)]
        offsTop = max(0, offsTop)
        offsBottom = max(0, offsBottom)
        offsLeft = max(0, offsLeft)
        offsRight = max(0, offsRight)

        finalImg = cv.copyMakeBorder(image, offsTop, offsBottom, offsLeft, offsRight, GAP_BORDER)

        return finalImg

    else:
        return image[-offset.y:-offset.y + size.h, -offset.x:-offset.x + size.w]

if __name__ == '__main__':
    # Parses args.
    parser = ArgumentParser(description='Align, crop face and plot image.')
    #parser.add_argument("-f", "--file_path", default="resources/Feret/00329/00329fb010_940422.jpg", help='image file path')
    #parser.add_argument("-f", "--file_path", default="resources/FRGC/04470/04470d10.jpg", help='image file path')
    parser.add_argument("-f", "--file_path", default="resources/lena.jpg", help='image file path')
    args = vars(parser.parse_args())
    
    # Load and plot image.
    image = Image(args["file_path"])
    image = crop_face(image, 200)
    image.convert_to_gray()
    image.plot()
