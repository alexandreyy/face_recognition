'''
Created on 18/06/2015

@author: Alexandre Yukio Yamashita, Flavio Nicastro
'''

import cv2
from numpy import random
import os
import sys
import time

from files import Files
from image import Image
from logger import Logger
import numpy as np

set_total_used_for_training = 0
set_total_used_for_test = 0
max_persons = 999999

class FaceRecognition:
    '''
    Face recognizer.
    '''    
    _logger = Logger()
    recognizer = None
    labels = None
    images = None
    all_images = None
    test_labels = None
    test_images = None
    directories = None
    classifier = None
    image_width = 0
    image_height = 0
    total_used_for_test = 0
    max_images_for_testing = 0
    ratio_used_for_test = 0
    total_images = 0
    total_labels = 80
       
    def __init__(self, classifier="eigenfaces"):
        self.set_classifier(classifier)
    
    def set_classifier(self, classifier="eigenfaces"):
        '''
        Set classifier for face recognition.
        '''
        if classifier == "eigenfaces":
            self._logger.log(Logger.INFO, "Creating eigenface recognizer.")
            self.recognizer = cv2.createEigenFaceRecognizer(80)
        elif classifier == "fisherfaces":
            self._logger.log(Logger.INFO, "Creating fisher face recognizer.")
            self.recognizer = cv2.createFisherFaceRecognizer(self.total_labels)
        else:
            self._logger.log(Logger.INFO, "Creating LBPH face recognizer.")
            self.recognizer = cv2.createLBPHFaceRecognizer()
        
        self.classifier = classifier
        
    def configure_database(self, database_path, max_images_loaded = 0, max_images_for_testing = 0, ratio_used_for_test=0.33):
        '''
        Load images.
        '''
        self._logger.log(Logger.INFO, "Configuring database using: " + database_path)
        
        self.directories = [os.path.join(database_path, o) for o in os.listdir(database_path) if os.path.isdir(os.path.join(database_path, o))]
        self.labels = []
        self.images = []
        self.test_labels = []
        self.test_images = []
        self.all_images = {}
        image_width = 0
        image_height = 0
        self.max_images_for_testing = max_images_for_testing
        self.ratio_used_for_test = ratio_used_for_test
        label = 0
        self.total_images = 0
        self.total_labels = len(self.directories)
        
        for directory in self.directories:
            image_files = Files(directory + "/")
            label += 1 
            
            # Person will be include in training only if it has at least two images.  
            total_image_files = len(image_files.paths)
                
            if total_image_files > 1:
                # Set image dimensions for training.
                if self.image_width == 0:
                    image = Image(image_files.paths[0])
                    
                    if image.width > 600:
                        r = 600.0 / image.data.shape[1]
                         
                        # perform the actual resizing of the image and show it
                        image = image.resize(600, int(image.data.shape[0] * r))
                        
                    self.image_width = image.width
                    self.image_height = image.height
                
                # Initialize list of images for person.
                self.all_images[label] = []
                
                # Read all images from directory and add them in recognizer.
                for image_file in image_files.paths:                    
                    image = Image(image_file)
                    
                    # Found a image with wrong dimensions, resize it. 
                    if image.width != self.image_width or image.height != self.image_height:
                        image = image.resize(self.image_width, self.image_height)
                    
                    # Convert and equalize image.
                    image.convert_to_gray()
                    image.equalize()
                    
                    # Add image.
                    self.all_images[label].append(image.data)
                    self.total_images += 1
                
                if max_images_loaded != 0 and self.total_images >= max_images_loaded and label > 2:
                    break
        
        self.shuffle()
        
    def train(self):
        '''
        Recognize person by image. 
        '''
        self._logger.log(Logger.INFO, "Training classifier.")
        self.total_labels = np.unique(self.labels).size
        self.set_classifier(self.classifier)
        
        try:
            self.recognizer.train(self.images, np.array(self.labels))
            return True
        except:
            return False
            
    def shuffle(self, iteration=0):
        '''
        Shuffle images for testing. 
        '''
        self._logger.log(Logger.INFO, "Shuffle images for testing.")
        self.labels = []
        self.images = []
        self.test_labels = []
        self.test_images = []
        labels = [label for label in self.all_images]
        random.shuffle(labels)
        
        # Set images used for traning and testing.
        if self.max_images_for_testing > 0 and self.max_images_for_testing < self.total_images:
            total_to_add = self.max_images_for_testing
        else:
            total_to_add = self.total_images
        
        new_label = 0
        total_persons = 0
        for label in labels:
            if total_persons < max_persons:
                random.shuffle(self.all_images[label])
                total_image_files = len(self.all_images[label])
                    
                # Get total of images used for training.
                total_used_for_test = int(total_image_files * self.ratio_used_for_test)
                
                if total_used_for_test < 1:
                    total_used_for_test = 1
                
                # Set total of images used for training. 
                total_used_for_training = total_image_files -total_used_for_test
                
                # Set number of total_used_for training manually
                if set_total_used_for_training > 0 and set_total_used_for_test > 0:
                    total_used_for_training = set_total_used_for_training
                    total_used_for_test = set_total_used_for_test
                
                # We need to add at least two images by label for testing.
                if total_to_add == 1:
                    total_to_add = 2
                
                if total_used_for_training + total_used_for_test <= total_image_files:
                    total_persons += 1
                    
                    for image in self.all_images[label]:
                        if total_used_for_test == 0 and total_used_for_training==0:
                            break
                        
                        if total_used_for_test < 1:
                            if total_used_for_training > 0:
                                self.images.append(image)
                                self.labels.append(new_label)
                                total_used_for_training -= 1
                            else:
                                break
                        else:
                            self.test_images.append(image)
                            self.test_labels.append(new_label)
                            total_used_for_test -= 1                
                        
                        total_to_add -= 1
                        
                        if total_to_add == 0:
                            break
                    
                if total_to_add == 0:
                    break
                
                new_label += 1
            else:
                break
            
        self.total_labels = np.unique(self.labels).size
        difference = 2*self.total_labels -1 -len(self.images)
        
        if set_total_used_for_training == 1:
            difference = 0
        
        if iteration < 100 and (difference > 0 or len(self.test_images) >= len(self.test_images) < 2 or len(self.images) < 3 or np.unique(self.labels).size == 1):
            if difference > 0:
                while difference > 0 and len(self.test_images) > 3:
                    self.images.append(self.test_images.pop())
                    self.labels.append(self.test_labels.pop())
                    self.total_labels = np.unique(self.labels).size
                    difference = 2*self.total_labels -1 -len(self.images)
            else:
                self.shuffle(iteration +1)
        
        #print len(self.images)/np.unique(self.labels).size
        #print len(self.images)
        
    def predict(self, image):
        '''
        Predict person by image. 
        '''
        return self.recognizer.predict(image)
    
    def test(self):
        '''
        Test face recognizer.
        '''
        self._logger.log(Logger.INFO, "Testing classifier.")
        total_test_images = len(self.test_images)
        total_match = 0
        
        for index_image in range(total_test_images):
            found_label = face_recognizer.predict(self.test_images[index_image])[0]
            
            if found_label == self.test_labels[index_image]:
                total_match += 1
        
        return total_match*100.0/total_test_images, total_match, total_test_images
    
if __name__ == '__main__':
    # Face recognizer configuration.
    database_paths = []
    max_iterations = []
    max_images_loaded = 0
    max_images = 0
    max_retries = 10
    set_total_used_for_training = 0
    set_total_used_for_test = 0
    max_persons = 99999999
    
    prefix = "../Faces/"
    #prefix2 = "/dev/shm/Faces/"
    prefix = ""
    #prefix2 = ""
     
    # Set databases for testing.
#     database_paths.append(prefix + "resources/alexandre/")
#     max_iterations.append(2)
#     database_paths.append(prefix + "resources/alexandre_cropped/")
#     max_iterations.append(2)
#     database_paths.append(prefix + "resources/Feret/")
#     max_iterations.append(180)
#    database_paths.append(prefix + "resources/Feret_cropped/")
#    max_iterations.append(100)
#     database_paths.append(prefix2 + "resources/FRGC/")
#     max_iterations.append(100)
#     database_paths.append(prefix + "resources/FRGC_cropped/")
#     max_iterations.append(50)
    database_paths.append(prefix + "resources/Yale/")
    max_iterations.append(330)
#    database_paths.append(prefix + "resources/Yale_cropped/")
#    max_iterations.append(100)
    
    # Set classifiers for testing.
    classifiers = []
    classifiers.append("fisherfaces")
    classifiers.append("eigenfaces")
    classifiers.append("LBPHfaces")
    history_file = open("results.txt", "w")
    history_file.close() 
    
#     total = range(1, set_total_used_for_training +1)
#     #total = range(set_total_used_for_training, set_total_used_for_training +1)
#     
#     for index_total_used_for_training in total:        
#         set_total_used_for_training = index_total_used_for_training
#         
#         write_data = "Total_used_for_training: " + str(index_total_used_for_training)
#         with open("results.txt", "a") as history_file:
#             history_file.write(write_data + "\n\n")
#         print write_data + "\n"
        
    index_database = 0
    for database_path in database_paths:
        history_result = {}
        
        for classifier in classifiers:
            history_result[classifier] = []
            
        face_recognizer = FaceRecognition()
        face_recognizer.configure_database(database_path, max_images_loaded, max_images)
            
        database_name = database_path.split("/")
        database_name = database_name[len(database_name) -2]
        elapsed_total = 0
        
        write_data = "Database: " + database_name
        with open("results.txt", "a") as history_file:
            history_file.write(write_data + "\n\n")
        print write_data + "\n"
        
        for iteration in range(max_iterations[index_database]):
            start = time.time()
            face_recognizer.shuffle()
            
            for classifier in classifiers: 
                face_recognizer.set_classifier(classifier)
                
                retries = 0
                
                while not face_recognizer.train():
                    retries +=1
                    
                    if retries > max_retries:
                        print "Max of retries achieved. Classifier will be ignored in this iteration."
                        print "Loading database again."
                        face_recognizer.configure_database(database_path, max_images_loaded, max_images)
                        break
                    
                    print "Failed to train " + classifier
                    print "Retraining with new images. Number of retries: " +  str(retries) + "\n"

                    face_recognizer.shuffle()
                
                if retries > max_retries:
                    break
                
                result = face_recognizer.test()
                
                write_data = "[" + str(iteration) + "] " + "[" + classifier + "] " + "{:5.2f}".format(result[0]) + "% - " + str(result[1]) + " of " + str(result[2])
                with open("results.txt", "a") as history_file:
                    history_file.write(write_data + "\n")
                print write_data
                history_result[classifier].append([result[0], result[1]])
            
#             print ""
#             print "Statistics: "
#             for classifier2 in classifiers: 
#                 if len(history_result[classifier2]) == 0:
#                     history_result[classifier2].append([0, 0])
#                     
#                 accuracy = [result[0] for result in history_result[classifier2]]              
#                 write_data = "[" + classifier2 + "] " + "Mean: " + "{:5.2f}".format(np.mean(accuracy)) + "% - Std: " + "{:5.2f}".format(np.std(accuracy)) + "%"
#                 with open("results.txt", "a") as history_file:
#                     history_file.write(write_data + "\n")
#                 print write_data
                    
            end = time.time()
            elapsed = end - start
            elapsed_total += elapsed
            estimate_elapse = elapsed_total * (max_iterations[index_database] -iteration -1) / (60*(iteration +1))
            
            print "[" + str(iteration) + "] Time: " + str(elapsed) + " s"
            print "[" + str(iteration) + "] Estimate time: " + str(estimate_elapse) + " min\n"
              
        with open("results.txt", "a") as history_file:
            history_file.write("\n")
        
        for classifier in classifiers: 
            if len(history_result[classifier]) == 0:
                history_result[classifier].append([0, 0])
                
            accuracy = [result[0] for result in history_result[classifier]]
            
            write_data = "[" + classifier + "] " + "Mean: " + "{:5.2f}".format(np.mean(accuracy)) + "% - Std: " + "{:5.2f}".format(np.std(accuracy)) + "%"
            with open("results.txt", "a") as history_file:
                history_file.write(write_data + "\n")
            print write_data
        
        write_data = "--------------------------------------------------------------"
        with open("results.txt", "a") as history_file:
            history_file.write(write_data + "\n")
        print write_data
        
        index_database += 1