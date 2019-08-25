import numpy as np
import cv2
import urllib
import pytesseract
import pandas as pd
import re
from tqdm import tqdm

file = open('Indian_Number_plates.json', 'r')
serial_no = 0
final_data = []

count_empty = 0

for each in tqdm(file):
    # for each URL or content in json file
    if serial_no != 107:
        length = len(each) # getting the length of each string
        a = each[:length-16] # removing extras=null as it is same in all string
        a += '}' # concatinating '}' as it is removed
        data = eval(a) # converting to dictionary
        url = data["content"] # storing URL of image
        annotation = data['annotation'] # returns a list of important co-ordinates of plates, image height and width

        # getting the co-ordinates of plates
        x1 = annotation[0]['points'][0]['x']
        y1 = annotation[0]['points'][0]['y']
        x2 = annotation[0]['points'][1]['x']
        y2 = annotation[0]['points'][1]['y']

        # getting the height and width
        image_height = annotation[0]['imageHeight']
        image_width = annotation[0]['imageWidth']

        #getting the image from url
        url_response = urllib.request.urlopen(url)
        img_array = np.array(bytearray(url_response.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, -1)

        # getting the exact required image using co-ordinates
        x1 =x1*image_width 
        y1=y1*image_height
        x2=x2*image_width 
        y2=y2*image_height
        img_rgb= img[int(y1) : int(y2) , int(x1) : int(x2)]
        
        #converting it to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        #checking for different contrast value 
        img_gray_contrast1 = ((img_gray/255)**.1)*255
        img_gray_contrast2 = ((img_gray/255)**.2)*255
        img_gray_contrast3 = ((img_gray/255)**.4)*255
        img_gray_contrast4 = ((img_gray/255)**2.5)*255
        img_gray_contrast5 = ((img_gray/255)**10)*255
        
        # Configuration for tesseract
        config = ('-l eng --oem 1 --psm 3')
        
        # Run tesseract OCR on grayscale_image_contrast5
        text_contrast4 = pytesseract.image_to_string(img_gray_contrast4, config=config)
        text_contrast4 = re.sub(r'\W+', '', text_contrast4) # removing special characters (keeping only AlphaNumeric values)
        text_update = re.sub('[^A-Z0-9]', '', text_contrast4) # taking only Capital Alphabets and numbers
        if len(text_update) == 0 or len(text_update) == 1:
            # Run tesseract OCR on rgb_image
            text_rgb = pytesseract.image_to_string(img_rgb, config=config)
            text_rgb = re.sub(r'\W+', '', text_rgb) # removing special characters (keeping only AlphaNumeric values)
            text_update = re.sub('[^A-Z0-9]', '', text_rgb) # taking only Capital Alphabets and numbers
            if len(text_update) == 0 or len(text_update) == 1:
                # Run tesseract OCR on grayscale_image
                text_gray = pytesseract.image_to_string(img_gray, config=config)
                text_gray = re.sub(r'\W+', '', text_gray) # removing special characters (keeping only AlphaNumeric values)
                text_update = re.sub('[^A-Z0-9]', '', text_gray) # taking only Capital Alphabets and numbers
                if len(text_update) == 0 or len(text_update) == 1:        
                    # Run tesseract OCR on grayscale_image_contrast1
                    text_contrast1 = pytesseract.image_to_string(img_gray_contrast1, config=config)
                    text_contrast1 = re.sub(r'\W+', '', text_contrast1) # removing special characters (keeping only AlphaNumeric values)
                    text_update = re.sub('[^A-Z0-9]', '', text_contrast1) # taking only Capital Alphabets and numbers
                    if len(text_update) == 0 or len(text_update) == 1:
                        # Run tesseract OCR on grayscale_image_contrast2
                        text_contrast2 = pytesseract.image_to_string(img_gray_contrast2, config=config)
                        text_contrast2 = re.sub(r'\W+', '', text_contrast2) # removing special characters (keeping only AlphaNumeric values)
                        text_update = re.sub('[^A-Z0-9]', '', text_contrast2) # taking only Capital Alphabets and numbers
                        if len(text_update) == 0:
                            # Run tesseract OCR on grayscale_image_contrast3
                            text_contrast3 = pytesseract.image_to_string(img_gray_contrast3, config=config)
                            text_contrast3 = re.sub(r'\W+', '', text_contrast3) # removing special characters (keeping only AlphaNumeric values)
                            text_update = re.sub('[^A-Z0-9]', '', text_contrast3) # taking only Capital Alphabets and numbers
                            if len(text_update) == 0:
                                # Run tesseract OCR on grayscale_image_contrast5
                                text_contrast5 = pytesseract.image_to_string(img_gray_contrast5, config=config)
                                text_contrast5 = re.sub(r'\W+', '', text_contrast5) # removing special characters (keeping only AlphaNumeric values)
                                text_update = re.sub('[^A-Z0-9]', '', text_contrast5) # taking only Capital Alphabets and numbers

                            
        # if still empty then count it
        if len(text_update) == 0:
            count_empty += 1
                
        # storing all Data in a 2D list.
        serial_no += 1
        final_data.append([serial_no,text_update])
    else:
        serial_no += 1
        final_data.append([serial_no,'NULL']) # for gif image
        continue
#Data is stored in CSV file
df = pd.DataFrame(final_data, columns = ['Serial_no','Number_plate'])
df.to_csv('result.csv')
print("No. of Datapoints that my model didn't recognize",count_empty)
print("done")