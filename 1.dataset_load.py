import os
import cv2
import pickle
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

#path to images and annotation folder
main_dir = os.getcwd()
image_dir = os.path.join(main_dir,'images')
ann_dir = os.path.join(main_dir,'annotations')

#defining labels and cataegory 
label_to_cat = {'without_mask': 0, 'with_mask': 1, 'mask_weared_incorrect': 2}
cat_to_label = {v: k for k,v in label_to_cat.items() }

#defining data set
datas = []

#gathering the labelled information from annotation files
for root, dirs, files in os.walk(ann_dir):
    for file in files:
        tree = ET.parse(os.path.join(root, file))
        data = {'path': None, 'objects': []}
        data['path'] = os.path.join(image_dir, tree.find('filename').text)
        for obj in tree.findall('object'):
            label = label_to_cat[obj.find('name').text]
            # top left co-ordinates
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            # bottom right co-ordinates
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)
            data['objects'].append([label, xmin, ymin, xmax, ymax])
        datas.append(data)

print ("Total Number of training images: ",len(datas))

#displaying sample image
index = np.random.randint(0, len(datas))
img = cv2.imread(datas[index]['path'])
for (category, xmin, ymin, xmax, ymax) in datas[index]['objects']:
    # Draw bounding boxes
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    #displaying text class
    cv2.putText(img, str(cat_to_label[category]), (xmin+2, ymin-3), cv2.FONT_HERSHEY_COMPLEX, 0.35, (255,0,0), 1)
# Show image
plt.figure(figsize=(10, 6))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


#take region of interest as data X and target as a categorical data
img_size = (100, 100)
X = []
Y = []
for data in datas:
    img = cv2.imread(data['path'])
    for (category, xmin, ymin, xmax, ymax) in data['objects']:
        roi = img[ymin : ymax, xmin : xmax]
        roi = cv2.resize(roi, (100, 100))
        data = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        target = to_categorical(category, num_classes=len(cat_to_label))
        X.append(data)
        Y.append(target)
X = np.array(X)
Y = np.array(Y)

np.save('X', X)
np.save('Y', Y)

with open('category2label.pkl', 'wb') as pf:
    pickle.dump(cat_to_label, pf)

