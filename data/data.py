import os
import cv2
import gzip
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

def loadMNISTFolders(imagePath, labelPath):
    with gzip.open(imagePath, 'rb') as f:
        f.read(4)
        numImages = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')

        imageData = f.read()
        images = np.frombuffer(imageData, dtype=np.uint8).reshape(numImages, rows, cols)
        images = images.astype(np.float32) / 255.0

    with gzip.open(labelPath, 'rb') as f:
        f.read(4)
        numLabels = int.from_bytes(f.read(4), 'big')
        labelData = f.read()
        labels = np.frombuffer(labelData, dtype=np.uint8)

    return images, labels

def parseDatFile(datPath):
    with open(datPath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    gridRows = lines[2:11]
    grid = []
    for rowStr in gridRows:
        if ' ' in rowStr:
            rowVals = [int(x) for x in rowStr.split()]
        else:
            rowVals = [int(c) for c in rowStr if c.isdigit()]
        grid.append(rowVals)

    return np.array(grid, dtype=int)

def loadSudokoProblems(folderPath):
    dataset = []

    datFiles = glob(os.path.join(folderPath, "*.dat"))

    for datPath in datFiles:
        baseName = os.path.splitext(datPath)[0]
        imgPath = None
        if os.path.exists(baseName + '.jpg'):
            imgPath = baseName + '.jpg'
            break

        if not imgPath:
            continue

        grid = parseDatFile(datPath)

        dataset.append({
            "imagePath": imgPath,
            "grid": grid
        })

    return dataset