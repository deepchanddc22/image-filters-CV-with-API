from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np
import requests
import os

app = FastAPI()

class ImageUrl(BaseModel):
    url: str

class ImageProcessingRequest(BaseModel):
    url: str
    filters: List[str]

def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Could not download the image")
    image = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def apply_cool_filter(image):
    cool_filter = np.array([[1.1, 0, 0],
                            [0, 1.1, 0],
                            [0, 0, 0.9]])
    cool_image = cv2.transform(image, cool_filter)
    cool_image = np.clip(cool_image, 0, 255)
    return cool_image.astype(np.uint8)

def apply_warm_filter(image):
    warm_filter = np.array([[0.9, 0, 0],
                            [0, 0.9, 0],
                            [0, 0, 1.1]])
    warm_image = cv2.transform(image, warm_filter)
    warm_image = np.clip(warm_image, 0, 255)
    return warm_image.astype(np.uint8)

def apply_bw_filter(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bw_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return bw_image

def apply_blur_filter(image):
    blur_image = cv2.GaussianBlur(image, (35, 35), 0)
    return blur_image

def save_image(image, folder_name, file_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    path = os.path.join(folder_name, file_name)
    cv2.imwrite(path, image)
    return path

# @app.post("/process-image/")
# def process_image(image_url: ImageUrl):
#     try:
#         original_img = download_image(image_url.url)
#     except HTTPException as e:
#         raise e

#     cool_image = apply_cool_filter(original_img)
#     warm_image = apply_warm_filter(original_img)
#     bw_image = apply_bw_filter(original_img)
#     blur_image = apply_blur_filter(original_img)

#     paths = {
#         "original": save_image(original_img, 'original_images', 'original.jpg'),
#         "cool": save_image(cool_image, 'cool_images', 'cool.jpg'),
#         "warm": save_image(warm_image, 'warm_images', 'warm.jpg'),
#         "bw": save_image(bw_image, 'bw_images', 'bw.jpg'),
#         "blur": save_image(blur_image, 'blur_images', 'blur.jpg')
#     }

#     return {"message": "Images processed successfully.", "paths": paths}

@app.post("/process-filters/")
def process_filters(request: ImageProcessingRequest):
    try:
        original_img = download_image(request.url)
    except HTTPException as e:
        raise e

    paths = {"original": save_image(original_img, 'original_images', 'original.jpg')}
    
    for filter_name in request.filters:
        if filter_name == "cool":
            filtered_image = apply_cool_filter(original_img)
        elif filter_name == "warm":
            filtered_image = apply_warm_filter(original_img)
        elif filter_name == "bw":
            filtered_image = apply_bw_filter(original_img)
        elif filter_name == "blur":
            filtered_image = apply_blur_filter(original_img)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown filter: {filter_name}")
        
        paths[filter_name] = save_image(filtered_image, f'{filter_name}_images', f'{filter_name}.jpg')

    return {"message": "Selected filters applied and images processed successfully.", "paths": paths}

# To run the app use:
# uvicorn main:app --reload
