import numpy as np
import os
import cv2
import parse

from cvision_tools import detect_face, crop_face, convert_to_gray, resize_with_pad

from sklearn.model_selection import train_test_split

CK_IMAGE_PATH = "data/ck/CK+/extended-cohn-kanade-images/cohn-kanade-images/"
CK_INFO_PATH  = "data/ck/CK+/Emotion_labels/Emotion/"

CK_INFO_FROMAT_STRING = "{}e+00\n"

class CohnKanadeDataset():
    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
    
    def read(self, test_size=0.2, valid_size=0, gray=True, image_height=100, image_width=100, image_n_channels=1 ):
        images, labels = self.read_ck_data()
        images = np.array(images)
        faces = [crop_face(image) for image in images]
        faces = [resize_with_pad(face, image_height, image_width) for face in faces]
        if gray==True:
            faces = [convert_to_gray(face) for face in faces]
            faces = np.array(faces)
            faces=faces.reshape(-1, image_height, image_width, image_n_channels)

        X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=test_size, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=42)

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test

    def read_ck_data(self):
        images = []
        labels = []
        for subject_dir in os.listdir(CK_IMAGE_PATH):
            subject_image_path = os.path.join(CK_IMAGE_PATH, subject_dir)
            subject_info_path  = os.path.join(CK_INFO_PATH, subject_dir)
            
            for folder in os.listdir(subject_info_path):
                subject_folder_image_path = os.path.join(subject_image_path, folder)
                subject_folder_info_path  = os.path.join(subject_info_path, folder)
                image_file = os.listdir(subject_folder_image_path)
                info_files = os.listdir(subject_folder_info_path)
                if info_files:
                    first_image_file = image_file[0]
                    midle_image_file = image_file[len(image_file)//2]
                    last_image_file = image_file[-1]
                    info_file = info_files[0]
                    
                    # Read first image is always neutral emotion (encoded as 0)
                    subject_image_file_path = os.path.join(subject_folder_image_path, first_image_file)
                    subject_info_file_path  = os.path.join(subject_folder_info_path, info_file)
                    image, label = self.read_ck_subject(subject_image_file_path, subject_info_file_path)
                    label = 0
                    images.append(image)
                    labels.append(label)

                    # Read middle image and emotion label
                    subject_image_file_path = os.path.join(subject_folder_image_path, midle_image_file)
                    subject_info_file_path  = os.path.join(subject_folder_info_path, info_file)
                    image, label = self.read_ck_subject(subject_image_file_path, subject_info_file_path)
                    images.append(image)
                    labels.append(label)

                    # Read last image and emotion label
                    subject_image_file_path = os.path.join(subject_folder_image_path, last_image_file)
                    subject_info_file_path  = os.path.join(subject_folder_info_path, info_file)
                    image, label = self.read_ck_subject(subject_image_file_path, subject_info_file_path)
                    images.append(image)
                    labels.append(label)
                    #print(image)
                    #print (label)
            
        return images, labels

    def read_ck_subject(self, subject_image_file_path, subject_info_file_path):
        subject_image_file_full_path = os.path.abspath(subject_image_file_path)
        subject_info_file_full_path = os.path.abspath(subject_info_file_path)
        
        #print(subject_image_file_full_path)
        #print(subject_info_file_full_path)
        image = cv2.imread(subject_image_file_full_path)
        
        with open(subject_info_file_full_path, "r") as f:
            string = f.readline()
            parsed = parse.parse(CK_INFO_FROMAT_STRING, string)
            label = int(float(parsed[0]))

        return image, label