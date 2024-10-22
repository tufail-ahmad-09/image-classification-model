import os
import cv2
import numpy as np

real_path = 'data/Real'
generated_path = 'data/Fake'

if not os.path.exists('data'):
    os.makedirs('data')

images = []
labels = []

for filename in os.listdir(real_path):
    if filename.endswith(('.jpg','.png','.jpeg')):
        img_path = os.path.join(real_path, filename)
        img = cv2.imread(img_path)
        images.append(img)
        labels.append(0)

for filename in os.listdir(generated_path):
    if filename.endswith(('.jpg','.png','.jpeg')):
        img_path = os.path.join(generated_path, filename)
        img = cv2.imread(img_path)
        images.append(img)
        labels.append(1)
images = np.array(images)
labels = np.array(labels)

np.save('data/images.npy', images)
np.save('data/labels.npy', labels)

print("Data Prepration Complete")
