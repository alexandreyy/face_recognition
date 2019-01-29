'''
Created on 18/06/2015

@author: Alexandre Yukio Yamashita, Flavio Nicastro
'''
import os

from crop_face import crop_face
from files import Files
from image import Image
from logger import Logger
from multiprocessing import cpu_count, Pool

def run_job(params):
    '''
    Pre process image.
    '''
    directory = params[0]
    root_path = params[1]
    before_crop_resize = params[2]
    image_size = params[3]
    
    # Read and crop image.
    image_files = Files(directory + "/")
    directory = directory.split("/")
    directory = directory[len(directory) - 1]
    
    for image_file in image_files.paths:
        image = Image(image_file)
        
        image = crop_face(image, image_size)
         
        if image.width > before_crop_resize:
            image = image.resize(before_crop_resize)
         
        # Save image.
        file_name = image_file.split("/")
        file_name = file_name[len(file_name) - 1]                
        image.save(root_path + directory + "/" + file_name)
        
class PreProcessing:
    '''
    Align and crop faces from database.
    '''    
    _logger = Logger()
    database_path = None
    directories = None
    total_jobs = 0
    root_path = None
    
    def __init__(self, database_path, total_jobs = 1):
        self.database_path = database_path
        self.total_jobs = total_jobs
        
    def run(self, size = 200, before_crop_resize = 400):
        '''
        Run pre-processment.
        '''
        # Create directories to save face images.
        self._create_directories()
        image_size = size
        before_crop_resize = before_crop_resize
        
        # Crop face and save image.
        self._logger.log(Logger.INFO, "Pre-processing images for database: " + self.database_path)
        
        # Creating process pool
        process_pool = Pool(processes=self.total_jobs)
        root_list = [self.root_path] * len(self.directories)
        crop_list = [before_crop_resize]  * len(self.directories)
        image_list = [image_size]  * len(self.directories)
        
        # Processing images.
        ret = process_pool.map_async(run_job, zip(self.directories, root_list, crop_list, image_list))
        ret.get()

        # Closing process pool
        process_pool.close()
        
    def _create_directories(self):
        '''
        Create directories to save face images.
        '''
        self._logger.log(Logger.INFO, "Creating directories for database: " + self.database_path)
        
        self.directories = [os.path.join(self.database_path, o) for o in os.listdir(self.database_path) if os.path.isdir(os.path.join(self.database_path, o))]
        splitted_path = self.directories[0].split("/")
        splitted_path_size = len(splitted_path)
        self.root_path = ""
        
        # Get root path.
        for index_directory in range(splitted_path_size):
            if index_directory < splitted_path_size - 2:
                self.root_path += splitted_path[index_directory] + "/"
            else:
                self.root_path += splitted_path[index_directory] + "_cropped/"
                
                # Create root directory if it does not exist.
                if not os.path.exists(self.root_path):
                    os.makedirs(self.root_path)
                
                break;
        
        # Create new directories.
        for index_directory in range(len(self.directories)):
            directory = self.directories[index_directory].split("/")
            directory = directory[len(directory) - 1]
            directory = self.root_path + directory

            # Create directory if it does not exist.
            if not os.path.exists(directory):
                os.makedirs(directory)

if __name__ == '__main__':
    #database_path = "resources/alexandre/"
    #database_path = "resources/Yale/"
    #database_path = "resources/Feret/"
    database_path = "resources/FRGC/"
    
    total_jobs = cpu_count() - 1
    
    # Execute pre processement.
    pre_processing = PreProcessing(database_path, total_jobs)
    pre_processing.run()
