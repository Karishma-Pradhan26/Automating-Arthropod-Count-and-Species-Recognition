import pyrebase
import streamlit as st
import time 
import cv2
import os
from PIL import Image
from collections import Counter
import pandas as pd
#from PIL import Image as PILImage
from torchvision.transforms.functional import to_pil_image
from numpy import asarray
from sahi.utils.yolov8 import download_yolov8n_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.predict import get_prediction, get_sliced_prediction, predict, visualize_object_predictions
#from IPython.display import Image as IPythonImage
import urllib.request
import numpy as np



st.set_page_config(layout="wide", page_title="Arthropods Detection APP")

st.write("## Detect Arthropods from your image")
st.write("Try uploading an image to detect the arthropods. Full quality images can be downloaded from the sidebar:grin:")
st.sidebar.write("## Upload :gear:")


# first write the file name u want to display on database and file which u want to upload
#storage.child("PXL_20240202_184304751.jpg").put("/content/drive/MyDrive/PXL_20240202_184304751.jpg")

#first write the file name u want to download then write the local name
#storage.download("PXL_20240202_184304751.jpg","downloaded.jpg")

#*************NMS Function starts*******************************

def nms(boxes, scores, iou_threshold, object_names):
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    selected_indices = []

    while sorted_indices:
        current_index = sorted_indices.pop(0)
        selected_indices.append(current_index)
        rest_boxes = [boxes[i] for i in sorted_indices]
        iou_scores = [iou(boxes[current_index], rest_box) for rest_box in rest_boxes]

        sorted_indices = [
            sorted_indices[i] for i in range(len(iou_scores)) if iou_scores[i] < iou_threshold
        ]

    selected_boxes = [boxes[i] for i in selected_indices]
    selected_object_names = [object_names[i] for i in selected_indices]
    selected_confidence_scores = [scores[i] for i in selected_indices]

    return selected_boxes, selected_object_names, selected_confidence_scores

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou


# ************************Main function starts************************* 
def process_image(image_file, confidence_threshold):
  st.write(image_file.name)
  service_account = {
  "type": "service_account",
  "project_id": "arthropods-fd246",
  "private_key_id": "2ae9b5e4194524f6b38de6a2444d00cbb6dc8fcb",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDlZlK5rVMcFy5t\nd6LW8eln2uK8laeDCLszmHvUHy6gQgUjFn6OCNJm9gmJhNpB39onavJhxkYFnxSg\nfdXNtKigZyLR2MCLAuHp+S3zYzDdqD0UbV6jXq/BQ8Tf0P2HK9Og14mDu5QJST+T\nLNE+KMDvn47ON6ChNs4NzT015Za1oUYIl0B0/uBelxCHW23arNzJxuw1wIxqbiY4\nfQizUDszbHJKzFG7IPo2MZGqSLUDrLbbcr05nYYXs6Sd2J6zm7Zsd9bt4PrEL61Z\nbb1/CaDiv92CYfALRHLaVul9LL312Vi/RSfCCQ6Gl3GsAj5QMiXhedJjJF156xHu\novgRMojxAgMBAAECggEAcau24UfoD1rTSC4IlD7yG+LECJOVmmIItSPhdDZxWbrt\nJbpfQo1PiPJwvmKE7jB4vqCJc3aWuvFdfLtt5s0W48jHYbloi1mFzXUKVni3vZZA\nke8x/n0MVRV6KDe8cS7/4kr7WrnnvIQ9yvVhx83U8MgdMMUyxL/N8dvZ8WuPNvc/\ne0c/WAuQe/0HsUuPOqGIA3OEeFP5BynkzufH1PHuJusKryyF1EypxJ7HbuVEKFWt\nMbrn8EkNPBzWvLtEcgPljFiTtqL2/oxgSIe9+FlfzLTKJyAjxAl2HY7pIs9CbkaC\nyYEppupPAhPNS1+1nUyJEAuV89zqWiL1UGNWVSOoQwKBgQD0OG7/1dprIeUFTrzq\nbOhw67/i6yAM+dBjv+E8IN8plWS6UwgFfcp2LwbbOcro8aZg+UHuqSECsCEfq4RS\n0Z+NWeHq/8IBe6LCck7MzhScd58jfpemYnVpofH9oYxFrbPZ+51DsTgIpQ7O5Nx5\nMr/1gEsWT6aBosgZ7o+FhRUymwKBgQDwduMZ0JL8ngMo2usoHwVWbefiA+BiN1WR\n/93DWdjGN3EZZb0LmnwdGpksnh8GqCjsIeSsqFtEISfoyyJqJtNTT/Rja66GLZdc\nLw7WxVcQ8bAXBjpKstaUNg6CVDYtEp/jSBuzTNe9b6XhFqwcLQ6vf3/0UIwfHUo1\nEms0VH/VYwKBgQCAEwxbDIqT/l19kBJQmja36VpjJU1UoLnhZeI3x0FhLGB7ukmN\nn69KlVTyKPe5e+j2GCG0iGuwXi+wzSEC0b3rwgRRWjkiz/TGJm4ex8BVYgJJ+fyL\nnDzY3b3rdgn7DKIyIdsSgeDR38XJSIo64Rt8IwC9vZ2jhHuN5ebMbcN1GwKBgQC0\nhJuicokV6c13azkkCn2oT6PHcZ+OO/oaO5oQoC9zRoW6Sj0sK9JGywP4kc+64Jyw\nABW6ZDz3nWhng6+XPwRgQ+K0jE8dXbaGNrGOG+Frz/SgpEU4dJ9P7nDf+DYfih7L\nsgNRjDtddCZ0HuAf2tduemnlUF6QsR1XUDd4toKs/wKBgQDQweV0ihiIEFC8b4gU\n+Jul7CBwEq/8GDWG/QoucW5fRxOYeh9/lpAWX56ckpnUx6OpNBuShPN6/OSjQ8em\nurguJLuTamHUbNtd0Rfn3bBS1fdqKY+q2nEdsU/jlSOQuggrXBN+5S2nWkJELDMe\nTuR9Fn1URbBgWqJ8Lc7+SdslcA==\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-hqngv@arthropods-fd246.iam.gserviceaccount.com",
  "client_id": "111199615214923048537",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-hqngv%40arthropods-fd246.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"}
 
  config = {
  "apiKey": "AIzaSyAJzChgGmAHIcz0Jo7zxXsZ75sP0zoB_3c",
  "authDomain": "arthropods-fd246.firebaseapp.com",
  "projectId": "arthropods-fd246",
  "storageBucket": "arthropods-fd246.appspot.com",
  "messagingSenderId": "335666008622",
  "appId": "1:335666008622:web:7ebe04ff7f98e8771d85bf",
  "measurementId": "G-BM9VQ53BRH",
  "serviceAccount": service_account,
  "databaseURL": "https://arthropods-fd246-default-rtdb.firebaseio.com/"};

  # ***********Storing image***********************************
  firebase = pyrebase.initialize_app(config)
  storage = firebase.storage()
  image_bytes = image_file.read()
  storage.child(image_file.name).put(image_bytes)
  #***************Fetching image************************************
  image_url = storage.child(image_file.name).get_url(None)
  model_url = storage.child("best.pt").get_url(None)

   # ************Now, you can use Image for further processing using SAHI******
  detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_url,
        confidence_threshold=confidence_threshold,
        device='cpu'
    )

  result = get_sliced_prediction(
        image_url,
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.3,
        overlap_width_ratio=0.3
    )

  boxes = [[pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy] for pred in result.object_prediction_list]
  scores = [pred.score.value for pred in result.object_prediction_list]
  names  = [pred.category.name for pred in result.object_prediction_list]
  selected_boxes, selected_names, selected_scores = nms(boxes, scores, iou_threshold,names)
  
  
  # Fetch the image from Firebase Storage
  image_bytes = urllib.request.urlopen(image_url).read()
  nparr = np.frombuffer(image_bytes, np.uint8)
  image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

  # Draw bounding boxes and labels on the image
  for box, object_name, confidence in zip(selected_boxes, selected_names, selected_scores):
    x_min, y_min, x_max, y_max = map(int, box)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    font_scale = 2.0
    text = f"{object_name}: {confidence:.2f}"
    cv2.putText(image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
    
  # Encode the result image as bytes
  _, result_image_bytes = cv2.imencode('.jpg', image)

  # Upload the result image bytes to Firebase Storage
  img_name = str(confidence_threshold)+image_file.name
  storage.child("result_with_nms.jpg"+img_name).put(result_image_bytes.tobytes())

  # Get the URL of the result image
  result_image_url = storage.child("result_with_nms.jpg"+img_name).get_url(None)
  
  #Table creation
  object_prediction_list = result.object_prediction_list
  class_map = {
    'MA': 'Melon Aphid',
    'NT': 'Nesidiocorius Tenius',
    'OI': 'Orius Insidiosus',
    'WFT': 'Western Flower Thrips',
    'WF': 'White Fly',
    'TS': 'Two Spotted Spidermite'}
  class_counts = {class_name: 0 for class_name in class_map.keys()}
  for prediction in object_prediction_list:
    class_name = prediction.category.name
    if class_name in class_counts:
      class_counts[class_name] += 1
      
  data = {
    'Abbreviation': [],
    'Full Name': [],
    'Count': []}

  for class_name, count in class_counts.items():
    data['Abbreviation'].append(class_name)
    data['Full Name'].append(class_map.get(class_name, class_name))
    data['Count'].append(count)

  df = pd.DataFrame(data)
  table_data = {"Abbreviation": df['Abbreviation'].tolist(), "Full Name": df['Full Name'].tolist(), "Count": df['Count'].tolist()}
    
    
  #Display of images and table
  col1.write("Original Image :camera:")
  col1.image(image_url)
  col2.write("## Class Counts")
  col2.write(df.set_index('Abbreviation', drop=True))
  col3.write("Detected Image :bug:")
  col3.image(result_image_url)




#************Main Funtion Ends************************************

#***************Buttons and sliders initiated***********************
MAX_FILE_SIZE = 5 * 1024 * 1024 
col1, col2, col3 = st.columns(3)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.05, 0.05,key="confidence_slider")
iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 0.9, 0.1, 0.1, key="iou_slider")
#iou_threshold=0.10


# ************************Button Condition start******************************


if st.button('Detect Arthropods'):
    
    
    if my_upload is not None:
        if my_upload.size > MAX_FILE_SIZE:
             st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
        else:
            with st.spinner(text='In progress'):
              process_image(image_file=my_upload,confidence_threshold=confidence_threshold)
            with st.spinner(text='In progress'):
                time.sleep(3)
                st.success('Done')

                bar = st.progress(50)
                time.sleep(3)
                bar.progress(100)
                st.success('Success message')
    else:
        st.error('Please upload an image')
ColorMinMax = st.markdown(
    '''<style> 
    div.stSlider > div[data-baseweb="slider"] > div[data-testid="stTickBar"] > div {
        background: rgb(1 1 1 / 0%);
    } 
    </style>''', unsafe_allow_html=True)

Slider_Cursor = st.markdown(
    '''<style> 
    div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"] {
        background-color: #1E90FF; 
        box-shadow: rgba(14, 38, 74, 0.2) 0px 0px 0px 0.2rem;
    } 
    </style>''', unsafe_allow_html=True)

Slider_Number = st.markdown(
    '''<style> 
    div.stSlider > div[data-baseweb="slider"] > div > div > div > div {
        color: #000000; 
    } 
    </style>''', unsafe_allow_html=True)

col = f'''<style> 
    div.stSlider > div[data-baseweb="slider"] > div > div {{
        background: linear-gradient(to right, #1E90FF 0%, #1E90FF {confidence_threshold }%, rgba(151, 166, 195, 0.25) {confidence_threshold }%, rgba(151, 166, 195, 0.25) 100%);
    }} 
    </style>'''

ColorSlider = st.markdown(col, unsafe_allow_html=True)

# ************************Slidder Condition start******************************

if (confidence_threshold > 0.05 and confidence_threshold < 1.00) and (iou_threshold>0 and iou_threshold<1.00):

    if my_upload is not None:
          if my_upload.size > MAX_FILE_SIZE:
              st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
          else:
              with st.spinner(text='In progress'):
                process_image(image_file=my_upload,confidence_threshold=confidence_threshold)
              with st.spinner(text='In progress'):
                  time.sleep(3)
                  st.success('Done')

                  bar = st.progress(50)
                  time.sleep(3)
                  bar.progress(100)
                  st.success('Success message')
