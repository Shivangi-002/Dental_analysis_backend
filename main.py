from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import uvicorn
from fastapi.responses import FileResponse
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"
    output_filename_lesion = f"lesion_{file.filename}"
    output_filename_restoration = f"restoration_{file.filename}"
    output_filename_implant = f"implant_{file.filename}"

    output_path_lesion = f"{PROCESSED_FOLDER}/{output_filename_lesion}"
    output_path_restoration = f"{PROCESSED_FOLDER}/{output_filename_restoration}"
    output_path_implant = f"{PROCESSED_FOLDER}/{output_filename_implant}"

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image
    process_dental_xray(file_path, output_path_lesion, output_path_restoration, output_path_implant)

    # Return all processed images to the frontend
    return {
        "message": "Processing complete",
        "lesion_image": f"/processed/{output_filename_lesion}",
        "restoration_image": f"/processed/{output_filename_restoration}",
        "implant_image": f"/processed/{output_filename_implant}"
    }

@app.get("/processed/{filename}")
async def get_processed_image(filename: str):
    file_path = f"{PROCESSED_FOLDER}/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}

def process_dental_xray(image_path, lesion_output, restoration_output, implant_output):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_resized = cv2.resize(image, (500, 500))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_image = clahe.apply(image_resized)

    blurred = cv2.GaussianBlur(enhanced_image, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Convert image to color for visualization
    lesion_output_img = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
    restoration_output_img = lesion_output_img.copy()
    implant_output_img = lesion_output_img.copy()

    # ==== LESION DETECTION (Caries) ====
    cavity_mask = cv2.inRange(image_resized, 0, 60)  # Select darker regions
    cavity_contours, _ = cv2.findContours(cavity_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in cavity_contours:
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if 10 < cw < 50 and 10 < ch < 50:
            cv2.rectangle(lesion_output_img, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 2)
            cv2.putText(lesion_output_img, "Lesion", (cx, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ==== RESTORATION DETECTION ====
    restoration_mask = cv2.inRange(image_resized, 250, 255)
    restoration_contours, _ = cv2.findContours(restoration_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in restoration_contours:
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if 10 < rw < 50 and 10 < rh < 50:
            cv2.rectangle(restoration_output_img, (rx, ry), (rx + rw, ry + rh), (255, 255, 0), 2)
            cv2.putText(restoration_output_img, "Restoration", (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # ==== IMPLANT DETECTION ====
    implant_mask = cv2.inRange(enhanced_image, 240, 255)
    implant_contours, _ = cv2.findContours(implant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in implant_contours:
        ix, iy, iw, ih = cv2.boundingRect(cnt)
        aspect_ratio = ih / iw
        if 4 <= iw <= 80 and 8 <= ih <= 150 and aspect_ratio >= 2:
            cv2.rectangle(implant_output_img, (ix, iy), (ix + iw, iy + ih), (0, 255, 255), 2)
            cv2.putText(implant_output_img, "Implant", (ix, iy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Save processed images
    cv2.imwrite(lesion_output, lesion_output_img)
    cv2.imwrite(restoration_output, restoration_output_img)
    cv2.imwrite(implant_output, implant_output_img)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
