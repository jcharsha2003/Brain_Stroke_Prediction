# 🧠 Brain Stroke Prediction Web Application

**Live App**: [BSP.com](#)  
**GitHub Repo**: [GitHub Link](#)

---

## 🚀 Overview

The **Brain Stroke Prediction (BSP)** web application is a machine learning-powered tool designed to assess the likelihood of a person having a stroke. With a simple, interactive interface, the app takes user input related to health, lifestyle, and personal metrics, and returns a real-time stroke risk prediction.

This project was built using a dataset of **8,600 rows and 34 features**. By selecting 20 core input features and generating the remaining ones using **KMeans Clustering** and **K-Nearest Neighbors (KNN)**, the app constructs a full feature set for prediction. It then uses a **pre-trained XGBoost model** with **96% accuracy** to produce stroke risk predictions.

---

## 🧠 What This Project Does

- Accepts user input via a user-friendly web interface
  - 📋 Personal Information
  - 🍔 Lifestyle Habits
  - 🏥 Health Metrics
- Applies **KMeans Clustering** to segment similar user profiles
- Uses **KNN** to predict and complete missing health features
- Feeds the complete feature vector into a trained **XGBoost** model
- Provides users with a **real-time stroke risk prediction**

> 🔒 **Note**: This app is not intended for medical diagnosis, but as a health awareness and prediction tool.

---

## 🛠️ Technologies Used

| Category        | Tools & Libraries                        |
|----------------|------------------------------------------|
| 💻 Frontend     | Streamlit (Python-based web framework)   |
| 🤖 Machine Learning | KMeans, KNN, XGBoost (96% accuracy)     |
| 📊 Data Handling | Pandas, NumPy                            |
| ☁️ Deployment   | Streamlit Community Cloud                |

---

## 📦 Installation & Usage

Follow these steps to set up the project locally on your system.

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>

## To Install Required Packages and To run our Application 
   ```bash 
   pip install -r requirements.txt
   streamlit run main.py
   ```

   

