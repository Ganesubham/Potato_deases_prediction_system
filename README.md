# 🥔 Potato Disease Prediction System 2025

A robust deep learning-based system to detect and classify potato leaf diseases—**Early Blight**, **Late Blight**, and **Healthy**—using Convolutional Neural Networks and Ensemble Learning. Built with accessibility in mind for farmers, this system is deployable on both web and mobile platforms.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Model Development](#model-development)
- [Deployment](#deployment)
---

## 🌱 Project Overview

Potato is a key crop in India and highly vulnerable to fungal diseases like Early and Late Blight. These can reduce yields by over 30%. Our AI-based system aims to:
- Automatically detect diseases from leaf images.
- Classify them as **Healthy**, **Early Blight**, or **Late Blight**.
- Provide results in real-time using a web/mobile interface.

---

## 🧰 Tech Stack

### 🖥️ Backend
- Python
- TensorFlow & Keras (for model building)
- Django + Django REST Framework (API for predictions)

### 💻 Frontend
- Streamlit (for interactive UI)

### 🧪 Libraries
- NumPy, Pandas, Matplotlib
- Scikit-learn (for classical models & evaluation)

---

## 📂 Dataset

- **Classes**: Early Blight (1000), Late Blight (1000), Healthy (152)
- **Format**: Images stored in subfolders per class
- **Preprocessing**: 
  - Resize to 256x256
  - Normalize pixel values to [0, 1]
  - Augmentation: flipping, rotation, zoom, brightness

---

## 🏗️ Project Architecture

1. **Data Wrangling**
2. **Feature Engineering**
3. **Model Training**
   - Classical Models: KNN, SVM (OvR), Random Forest, Decision Tree
   - Deep Learning Models: VGG19, ResNet50, EfficientNetB7
4. **Ensemble Learning**
   - Soft Voting & Weighted Voting
5. **System Deployment**
   - Backend (Django)
   - Frontend (Streamlit)

---

## 🧠 Model Development

### 🧪 Classical ML
- K-Nearest Neighbors (KNN)
- Support Vector Machine (One-vs-Rest)
- Decision Tree
- Random Forest

> **Limitation**: Low accuracy due to limited feature extraction capabilities.

### 🤖 Deep Learning
- VGG19, ResNet50, EfficientNetB7
- Combined using **Soft Voting** & **Weighted Voting** Ensemble

> **Improvement**: Higher generalization and classification accuracy.

---

## 🚀 Deployment

- **Backend**: RESTful API via Django
- **Frontend**: Streamlit interface for image upload & prediction
- **Real-time**: Upload an image → Get prediction + confidence score
