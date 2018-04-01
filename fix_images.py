import cv2
import os
import glob
import sys

train_path = 'testing_data'
label = sys.argv[1]

path = os.path.join(train_path, label, '*g')
files = glob.glob(path)
for fl in files:
    image = cv2.imread(fl)

    image = image[256:1024, 256:1756]
    # cv2.imshow("image", image)
    # cv2.waitKey(0)

    print(fl.split('\\')[-1])

    cv2.imwrite('testing_data/' + label + '/' + fl.split('\\')[-1], image)
