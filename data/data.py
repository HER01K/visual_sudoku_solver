import os
import cv2
import numpy as np
from glob import glob

def loadDigitFolders(Project):
    images = []
    labels = []

    for digit in range(10):
        folderPath = os.path.join(Project, str(digit))
        if not os.path.exists(folderPath):
            continue

        for imgPath in glob(os.path.join(folderPath, "*.*")):
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (28, 28))

            img = img.astype(np.float32) / 255.0

            images.append(img)
            labels.append(digit)

    return np.array(images), np.array(labels)

def loadSudokuFolders(Project):

def loadMNISTFolders(Project):
    