# AI-Powered Real-Time Human Activity Recognition System

## Overview

An advanced real-time Human Activity Recognition (HAR) system built using AI, computer vision, deep learning, FastAPI, and React.js. The system detects and classifies human activities in real time using pose estimation and sequence-based activity prediction.

This project combines modern AI pipelines, real-time video processing, pose landmark extraction, temporal sequence modeling, and full-stack deployment architecture to create a production-style intelligent activity recognition platform.

---

# Features

* Real-time human activity detection
* Pose estimation using MediaPipe
* Sequence-based activity recognition
* Deep learning activity classification
* FastAPI backend architecture
* React.js frontend dashboard
* Real-time webcam integration
* Confidence score prediction
* REST API integration
* Scalable modular architecture
* Activity prediction visualization
* Real-time frame processing pipeline
* Full-stack AI system design

---

# Supported Activities

* Walking
* Running
* Sitting
* Standing
* Jumping
* Squatting
* Push-Ups
* Idle Detection

---

# Tech Stack

## Frontend

* React.js
* TypeScript
* Vite
* Tailwind CSS

## Backend

* FastAPI
* Python
* OpenCV
* MediaPipe
* NumPy
* Scikit-learn

## AI / Machine Learning

* Deep Learning Sequence Prediction
* Pose Landmark Extraction
* Temporal Activity Classification

---

# System Architecture

```text
Camera Feed
     ↓
OpenCV Frame Capture
     ↓
MediaPipe Pose Estimation
     ↓
Pose Landmark Extraction
     ↓
Sequence Buffer Creation
     ↓
Activity Classification Model
     ↓
FastAPI Backend
     ↓
React Frontend Dashboard
```

---

# Project Structure

```text
HAR PROJECT/
│
├── backend/
│   ├── app/
│   │   ├── ml/
│   │   ├── routes/
│   │   ├── services/
│   │   └── main.py
│   │
│   ├── camera/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.tsx
│   │   └── main.tsx
│   │
│   ├── package.json
│   └── vite.config.ts
│
├── screenshots/
├── README.md
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/tenejsahukar-source/Human-Activity-Recognition-System.git
cd Human-Activity-Recognition-System
```

---

# Backend Setup

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Backend Server

```bash
uvicorn app.main:app --reload
```

Backend runs on:

```text
http://127.0.0.1:8000
```

---

# Frontend Setup

## Navigate to Frontend

```bash
cd frontend
```

## Install Dependencies

```bash
npm install
```

## Run Frontend

```bash
npm run dev
```

Frontend runs on:

```text
http://localhost:5173
```

---

# API Endpoints

## Health Check

```http
GET /
```

## Activity Prediction

```http
POST /predict
```

## Camera Streaming

```http
GET /camera
```

---

# Machine Learning Pipeline

## Workflow

1. Capture video frames
2. Extract body pose landmarks
3. Generate temporal frame sequences
4. Process sequences through activity classifier
5. Predict human activity
6. Return confidence score
7. Display result in frontend dashboard

---

# Real-Time Pose Estimation

The project uses MediaPipe Pose for extracting real-time skeletal landmarks from webcam frames.

Extracted features include:

* Body joint coordinates
* Motion patterns
* Pose alignment
* Temporal movement sequences

---

# Future Improvements

* CNN + BiLSTM + Attention architecture
* Multi-person activity detection
* Sensor fusion integration
* Fall detection system
* Workout repetition counter
* Real-time analytics dashboard
* Activity timeline tracking
* Transformer-based HAR models
* Cloud deployment
* Mobile integration

---

# Performance Goals

* Low-latency real-time inference
* High accuracy activity prediction
* Scalable backend architecture
* Optimized frame processing
* Stable pose tracking

---

# Screenshots

## Webcam Activity Detection

Add screenshot here:

```text
screenshots/demo1.png
```

## Pose Skeleton Visualization

Add screenshot here:

```text
screenshots/demo2.png
```

## Activity Prediction Dashboard

Add screenshot here:

```text
screenshots/demo3.png
```

---

# Deployment

## Backend

* FastAPI
* Docker
* Render / Railway / AWS

## Frontend

* Vercel
* Netlify

---

# Resume Project Highlights

* Built an end-to-end real-time AI-powered Human Activity Recognition system.
* Implemented pose estimation and sequence-based activity classification.
* Designed scalable FastAPI backend architecture.
* Developed responsive React.js frontend dashboard.
* Integrated real-time webcam streaming and activity prediction.

---

# Author

## Tenej Sahukar

B.Tech Computer Science Engineering

GitHub:
[https://github.com/tenejsahukar-source](https://github.com/tenejsahukar-source)

---

# License

This project is licensed under the MIT License.

---

# Star The Repository

If you found this project useful, consider giving it a star on GitHub.
