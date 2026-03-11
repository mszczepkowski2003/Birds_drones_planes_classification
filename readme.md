# Flying Objects Classification: Birds, Drones & Planes 🦅 🛸 ✈️

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?style=flat-square&logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker)

**[Kaggle Dataset](https://www.kaggle.com/datasets/maryamlsgumel/drone-detection-dataset/data)**


## 📌 Project Overview
This project is an **end-to-end deep learning solution** designed to classify images of flying objects into three categories: **Birds, Drones, and Planes**. It serves as a robust foundation for applications like anti-spy drone detection or ornithological activity monitoring.

The system features a complete machine learning pipeline for data processing and model training, coupled with a high-performance **FastAPI backend** for real-time inference.

---

## 🚀 Model & Data Pipeline

The core classification model leverages **MobileNetV3Large**, built using **TensorFlow/Keras**. This lightweight yet powerful architecture ensures an optimal balance between accuracy and inference latency.

- **Data Strategy:** Trained on an imbalanced dataset split strategically into **70% Training / 15% Testing / 15% Validation**.
- **Controlling the overfitting & Enrichment:** Extensive **Data Augmentation** was applied to the training set to prevent overfitting, artificially expand the dataset, and address class imbalance.
- **Results:** The final model achieves **~95% weighted accuracy** on both the test and validation sets, demonstrating strong generalization.

---

## 🏗️ System Architecture

The codebase champions modularity and separation of concerns. The ML model is fully decoupled from the API layer, ensuring the system is maintainable and scalable.

- **Backend / Inference Bridge:** **FastAPI** acts as the high-speed bridge between the predictive model and potential frontend applications.
- **Containerization:** The entire inference API is **Dockerized**, guaranteeing consistent execution across environments and ensuring it is ready for immediate deployment on platforms like Render.

---

## 📂 Directory Structure

```text
├── app/                  # FastAPI main application and API endpoints
│   ├── api.py            # Main application router
│   ├── api_helpers.py    # Inference utilities and helper functions
│   └── config.py         # API Configuration settings
├── docker-compose.yaml   # Docker Compose config for running the application
├── dockerfile            # Docker configuration for containerizing the API
├── notebooks/            # Jupyter Notebooks for exploratory data analysis and prototyping
├── requirements.txt      # API dependencies (optimized for the Docker image)
└── src/                  # Core Machine Learning source code (training, data ops)
    ├── data_ops/         # Data loading and processing scripts
    └── model/            # Model architecture, training loops, and validation
```

## Instructions on running the project via docker-compose

1. Build the docker image
```bash
docker-compose build
```

2. Run the docker container
```bash
docker-compose up
```

3. Getting the prediction: 
 - for single image prediction use the endpoint http://localhost:8000/predict (You can use the Swagger UI at http://localhost:8000/docs to test it)
 - for batch prediction use the endpoint http://localhost:8000/predict-batch (Swagger UI doesn't support batch prediction, you can test it using curl)
    - Curl command for predict-batch 
    ```bash
    curl -X POST "http://localhost:8000/predict-batch" \
         -H "accept: application/json" \
         -H "Content-Type: multipart/form-data" \
         -F "files=@/path/to/image1.jpg" \
         -F "files=@/path/to/image2.jpg" \
         -F "files=@/path/to/image3.jpg"
    ```

---

## 🗺️ Roadmap & Future Plans

- [ ] **Cloud Deployment:** Deploy the FastAPI backend via Render.
- [ ] **Interactive UI:** Connect the backend to a responsive **Streamlit** application for seamless user testing.
- [ ] **Advanced Computer Vision:** Evolve the current classification pipeline into a YOLO-style multiple-object detection system capable of locating and classifying several objects within a single image.

---
*Built as a showcase of end-to-end ML engineering best practices.*