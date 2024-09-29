# forest-fire-deteection
Based on the content of the project documentation on **"Forest Fire Detection using CNN-RF and CNN-XGBoost Machine Learning Algorithms"**, here’s a draft of the README file for your GitHub repository:

---

# **Forest Fire Detection using CNN-RF and CNN-XGBoost**

### **Project Team:**
- Akkamwar Amulya (20J21A0501)
- Annamgari Sai Kiran (20J21A0503)
- Kamutala Chandra Siddartha (20J21A0531)
- Kommineni Sai Manu (20J21A0537)

### **Supervisor:**
- Dr. T. Prabhakaran, HOD, Department of Computer Science & Engineering, Joginpally B.R. Engineering College

## **Project Overview**
This project presents a solution for early detection of forest fires using a combination of **Convolutional Neural Networks (CNN)** and two powerful machine learning algorithms: **Random Forest (RF)** and **Extreme Gradient Boosting (XGBoost)**. The primary goal is to enhance detection accuracy by leveraging deep learning and ensemble learning techniques.

## **Table of Contents**
1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Implementation Details](#implementation-details)
4. [Algorithms Used](#algorithms-used)
5. [Project Modules](#project-modules)
6. [Dataset](#dataset)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Results](#results)
9. [Conclusion](#conclusion)
10. [Future Enhancements](#future-enhancements)
11. [References](#references)

## **Introduction**
Forest fires pose a major threat to both ecosystems and human lives. Early detection systems using satellite or UAV-based imaging can help mitigate this risk. Our project utilizes CNNs for feature extraction from forest fire images and combines them with RF and XGBoost models for classification and prediction of fire-prone areas.

## **System Architecture**
The system architecture consists of the following steps:
1. **Data Collection**: Collect forest fire image data from satellite sensors or UAVs.
2. **Pre-processing**: Enhance image quality by removing noise and adjusting image sizes.
3. **Feature Extraction**: Use CNN to extract image features like fire patterns, smoke, and intensity.
4. **Classification**: Classify the images using Random Forest and XGBoost models.
5. **Prediction**: Predict the likelihood of a fire occurrence based on the classifier output.

## **Implementation Details**
The system was implemented using **Python** with the following libraries:
- **TensorFlow/Keras** for the CNN model.
- **Scikit-learn** for Random Forest and XGBoost models.
- **OpenCV** for image pre-processing.

### **System Requirements**
- **Software**: Python 3.7, Anaconda, TensorFlow/Keras, OpenCV, scikit-learn, XGBoost.
- **Hardware**: 
  - CPU: Intel Core i5 or higher
  - RAM: 8GB or more
  - GPU (optional for faster training): Nvidia GTX 1050 or higher

## **Algorithms Used**
- **CNN-RF**: Uses a CNN to extract image features, which are then classified using a Random Forest model.
- **CNN-XGBoost**: Uses CNN-extracted features for classification through the XGBoost algorithm to achieve higher accuracy.

## **Project Modules**
1. **Data Selection and Pre-processing**: Involves selecting fire/non-fire images and pre-processing them for model training.
2. **Model Training**: Training CNN with RF and XGBoost models.
3. **Prediction**: Predict the presence of fire in test images using the trained models.
4. **Evaluation**: Evaluate model accuracy and visualize performance metrics like accuracy, recall, and F1-score.

## **Dataset**
The dataset includes satellite images or UAV imagery with both fire and non-fire labels. Images are resized and processed before being fed into the CNN model.

## **Evaluation Metrics**
- **Accuracy**: Percentage of correct predictions.
- **Precision, Recall, F1-Score**: To measure the performance of fire detection, especially in imbalanced datasets.
- **AUC-ROC Curve**: Measures the model's ability to distinguish between fire and non-fire images.

## **Results**
The model achieved high accuracy in detecting forest fires using the CNN-RF and CNN-XGBoost models, demonstrating the effectiveness of combining deep learning and ensemble methods.

## **Conclusion**
The system provides an efficient way to detect forest fires early, helping in minimizing damage. The combination of CNN for feature extraction and RF/XGBoost for classification offers a robust solution with high accuracy.

## **Future Enhancements**
- **Integration with IoT Sensors**: Use real-time sensor data for enhanced predictions.
- **Multi-Modal Data**: Incorporate satellite data and weather forecasts for improved accuracy.
- **Real-time Monitoring**: Develop a real-time fire detection system using live camera feeds and streaming analytics.

## **References**
1. Jamil Ahmad et al., "Efficient Deep CNN-Based Fire Detection and Localization in Video Surveillance Applications", IEEE Transactions, 2018.
2. Yanhong Chen et al., "UAV Image-Based Forest Fire Detection Approach Using CNN", 2019.
3. Khan Muhammad et al., "Spatio-Temporal Flame Modeling for Automatic Fire Detection", IEEE Transactions, 2013.

---

Let me know if you need any changes or additional sections!
