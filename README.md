# 🦷 Dental X-ray Analysis - Backend (FastAPI)

This is the backend service for **Dental X-ray Analysis**, which processes X-ray images uploaded by users, detects **lesions (caries), restorations (fillings), and implants**, and returns the analyzed images to the frontend.

---

## **🚀 Features**
- Upload a **dental X-ray image** for analysis
- **Process the image** using OpenCV (image segmentation, thresholding, etc.)
- **Detect and label:**
    - 🦷 **Lesions (Caries)**
    - ⚒️ **Restorations (Fillings)**
    - 🏗 **Implants**
- Return processed images to the frontend for display.

---

## **🛠️ Tech Stack**
- **FastAPI** - Lightweight Python web framework
- **Uvicorn** - ASGI server for FastAPI
- **OpenCV** - Image processing
- **NumPy** - Numerical operations
- **Python-Multipart** - Handling file uploads
- **CORS Middleware** - Allow frontend communication

---

## **📌 Local Deployment Steps**

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/YOUR_USERNAME/dental-analysis-backend.git
cd dental-analysis-backend
uvicorn main:app --reload
```
