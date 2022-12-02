


import h5py as h5py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys    
from local_utils import detect_lp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from os.path import splitext, basename
from keras.models import model_from_json
import glob
from paddleocr import PaddleOCR,draw_ocr
def captureImage():
    cam = cv2.VideoCapture(0)

# title of the app
    cv2.namedWindow('python webcam screenshot app')

# let's assume the number of images gotten is 0
    img_counter = 0
    while True:
    # intializing the frame, ret
        ret, frame = cam.read()
        # if statement
        if not ret:
            print('failed to grab frame')
            break
        # the frame will show with the title of test
        cv2.imshow('Captured Image', frame)
        #to get continuous live video feed from my laptops webcam
        k  = cv2.waitKey(1)
        # if the escape key is been pressed, the app will stop
        if k%256 == 27:
            print('escape hit, closing the app')
            break
        
        elif k%256  == 32:
            img_name = f'opencv_frame_{img_counter}'
            cv2.imwrite('C:/Users/prapp/OneDrive/Desktop/NCSU/Senior-Year/ECE411/lpdr_project_code/Plate_detect_and_recognize/test_images/input.jpg', frame)
            print('screenshot taken')
            cam.release()
            cv2.destroyWindow('Captured Image')

#Use webcam to capture image and save it as test.jpg
captureImage()


wpod_net_path =  "C:/Users/prapp/OneDrive/Desktop/NCSU/Senior-Year/ECE411/lpdr_project_code/Plate_detect_and_recognize/wpod-net.json"
def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

image_paths = glob.glob("C:/Users/prapp/OneDrive/Desktop/NCSU/Senior-Year/ECE411/lpdr_project_code/Plate_detect_and_recognize/test_images/input.jpg")
#image_paths = glob.glob("C:/Users/prapp/OneDrive/Desktop/NCSU/Senior-Year/ECE411/lpdr_project_code/Plate_detect_and_recognize/test_images/IMG_5580.jpg")
print("Found %i images..."%(len(image_paths)))


from keras.utils.image_utils import save_img
def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# Obtain plate image and its coordinates from an image
test_image = image_paths[0]
LpImg,cor = get_plate(test_image)
print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])
print("Coordinate of plate(s) in image: \n", cor)

cv2.imshow("Detected Plate",LpImg[0])
cv2.waitKey(0)
cv2.destroyAllWindows()
save_img('C:/Users/prapp/OneDrive/Desktop/NCSU/Senior-Year/ECE411/lpdr_project_code/Plate_detect_and_recognize/test_images/test.jpg', LpImg[0])

def calculate_rectangle(box):
  topleftx, toplefty = box[0]
  toprightx, toprighty = box[1]
  btmleftx, btmlefty = box[3]
  btmrightx, btmrighty = box[2]
  rectangle = (toprightx - topleftx) * (btmlefty - toplefty)
  return rectangle, int(topleftx), int(toplefty), int(btmrightx), int(btmrighty)

ocr = PaddleOCR(use_angle_cls=True, lang='en') # need to run only once to download and load model into memory
img_path = 'C:/Users/prapp/OneDrive/Desktop/NCSU/Senior-Year/ECE411/lpdr_project_code/Plate_detect_and_recognize/test_images/test.jpg'
img=cv2.imread(img_path)
result = ocr.ocr(img, cls=True)
for idx in range(len(result)):
    res = result[idx]
    # for line in res:
    #     #print(line)


# draw result
from PIL import Image
result = result[0]

image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]

rectangles = []
coords = []
i = 0;
for box in boxes:
  area, lx, ly, rx, ry = calculate_rectangle(box)
  rectangles.append([area, txts[i], i])
  coords.append([[lx, ly], [rx, ry]])
  i += 1

rectangles.sort(key=lambda row: (row[0]), reverse=True)
detected_text = rectangles[0][1]
result = cv2.rectangle(img, coords[rectangles[0][2]][0], coords[rectangles[0][2]][1],(255, 0, 0), 4)
cv2.putText(result, detected_text, coords[rectangles[0][2]][0], cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 255), 2)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()


