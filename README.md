# Enhanced Sonar Sensor Accuracy using Machine Learning

## Overview
This project focuses on improving sonar sensor accuracy through advanced machine learning techniques, specifically addressing the challenge of distinguishing between actual objects and small holes. By implementing a hybrid approach combining multiple clustering algorithms and a 1D Convolutional Neural Network (CNN), the system achieves an enhanced ability to process and interpret sonar sensor data.

## Features
### Data Processing & Analysis
- Feature Engineering
 - Distance calculation
 - Velocity derivation
 - Acceleration computation
 - Jerk analysis
 - Rolling mean calculations
 - Angular measurements
- Automated outlier detection and removal
- Signal smoothing using Savitzky-Golay filter
- Comprehensive data normalization

### Machine Learning Implementation
- Multiple Clustering Algorithms:
 - KMeans
 - Gaussian Mixture Models (GMM)
 - DBSCAN
 - Agglomerative Clustering
 - Spectral Clustering
 - Ensemble clustering integration
- 1D Convolutional Neural Network:
 - Custom architecture for sensor data
 - Real-time prediction capabilities
 - Optimized performance

### Performance Evaluation
- Clustering Metrics
 - Silhouette Score
 - Calinski-Harabasz Index
 - Davies-Bouldin Index
- Error Analysis
 - Mean Absolute Error (MAE)
 - Mean Squared Error (MSE)
 - Root Mean Squared Error (RMSE)
 - R-squared value

### Visualization Suite
- Raw vs Processed Data Comparison
- Clustering Results Analysis
- Error Distribution
- Model Training Progress
- Real-time Performance Metrics

## Requirements
```bash
pandas==2.0.0
numpy==1.24.0
scikit-learn==1.2.2
tensorflow==2.12.0
matplotlib==3.7.1
seaborn==0.12.2
scipy==1.10.1
joblib==1.2.0
