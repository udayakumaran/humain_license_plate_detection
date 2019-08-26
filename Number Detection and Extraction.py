import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pytesseract
import glob
import pytesseract
import requests
from PIL import Image
import matplotlib.pyplot as plt
import urllib
import random
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm

# tessaract Execute file path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# dataset Path
data = pd.read_json(r"E:\uday\humAIn\vehicle dataset\Indian_Number_plates.json", lines=True)
pd.set_option('display.max_colwidth', -1)
# print(data)


# Delete the empty column
del data['extras']

# Extract the points of the bounding boxes because thats what we want
data['points'] = data.apply(lambda row: row['annotation'][0]['points'], axis=1)

# And delete the rest of the annotation info
del data['annotation']
Images = []
Plates = []

# Training the Dataset
def downloadTraining(df):
    for index, row in df.iterrows():
        # Getting the image from the URL
        resp = urllib.request.urlopen(row[0])
        im = np.array(Image.open(resp))
        # append the image to the training input array
        Images.append(im)

        # Points of number Plate
        x_point_top = row[1][0]['x'] * im.shape[1]
        y_point_top = row[1][0]['y'] * im.shape[0]
        x_point_bot = row[1][1]['x'] * im.shape[1]
        y_point_bot = row[1][1]['y'] * im.shape[0]

        # Cut the plate from the image and use it as output
        carImage = Image.fromarray(im)
        plateImage = carImage.crop((x_point_top, y_point_top, x_point_bot, y_point_bot))
        Plates.append(plateImage)
        # grayscaled = cv2.cvtColor(plateImage, cv2.COLOR_BGR2GRAY)
        # retval, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
        # cv2.imshow('original', img)
        # cv2.imshow('threshold', threshold)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        text = pytesseract.image_to_string(plateImage, lang='eng')
        print("Plate Number is  : " + text)
        cv2.waitKey(0)


downloadTraining(data)

# Output Displaying part
# Create figure and axes
fig, ax = plt.subplots(2, 1, constrained_layout=True)

# Set title
ax[0].set_title('')
ax[1].set_title('')

# Display the images
n = int(input("Enter the car number between 0 - 237: "))

ax[0].imshow(Images[n])
ax[1].imshow(Plates[n])

plt.show()
