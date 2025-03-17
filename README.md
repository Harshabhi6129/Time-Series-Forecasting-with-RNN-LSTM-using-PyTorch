# **Time-Series Forecasting using RNNs and LSTMs**

## **Overview**
This repository contains a Jupyter Notebook that demonstrates **Time-Series Forecasting** using **Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models**. The project aims to build a robust forecasting model using **PyTorch** and evaluate its performance based on industry-standard metrics.

The objective is to develop a deep learning model capable of forecasting time-dependent data patterns with an accuracy of at least **75%** on the test set.

## **Dataset Description**
The dataset used in this project is the **Air Quality Data Set**, sourced from the **UCI Machine Learning Repository**.

- **Dataset Source:** UCI Machine Learning Repository  
- **Link:** [Air Quality Data Set](https://archive.ics.uci.edu/ml/datasets/Air+Quality)  
- **Description:** The dataset includes hourly readings of air pollutants and meteorological data collected from an urban monitoring station. The goal is to predict future air quality based on past observations.

## **Project Structure**
The Jupyter Notebook is organized into the following key sections:

### **1. Data Exploration and Preprocessing**
- Load the dataset using **Pandas**
- Display key statistics:
  - Number of samples (time points)
  - Number of features
  - Mean, standard deviation, minimum, and maximum values for each feature
- Identify missing values and handle them using imputation techniques
- Scale and normalize the dataset using **StandardScaler** from **scikit-learn**

### **2. Time-Series Data Preparation**
- Convert the dataset into a supervised learning problem using time-windowing techniques
- Split the dataset into training and testing sets for model evaluation
- Format data for deep learning models using **PyTorch**'s TensorDataset and DataLoader

### **3. Building the RNN & LSTM Models**
- Implement an **RNN (Recurrent Neural Network)** for time-series forecasting
- Implement an **LSTM (Long Short-Term Memory)** model with multiple layers
- Define model architecture including:
  - Input layer
  - Hidden layers with recurrent units
  - Fully connected output layer
- Use **torchinfo.summary()** to display the model's structure

### **4. Training the Models**
- Define a loss function (**Mean Squared Error - MSE**)
- Use **Adam optimizer** for gradient descent
- Train the models using **PyTorch**
- Monitor training performance using **TensorBoard**

### **5. Model Evaluation & Results**
- Evaluate the trained models on the test set using key performance metrics:
  - **Mean Absolute Error (MAE)**
  - **Mean Squared Error (MSE)**
  - **R-squared Score (R²)**
- Visualize actual vs. predicted values using **Matplotlib & Seaborn**
- Discuss model performance and areas for improvement

## **Installation & Dependencies**
### **Requirements**
To run the Jupyter Notebook, install the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn torch torchinfo scikit-learn tensorboard
```
Alternatively, install dependencies from the **requirements.txt** file:
```bash
pip install -r requirements.txt
```

### **Cloning the Repository**
```bash
git clone https://github.com/your-username/time-series-forecasting-rnn.git
cd time-series-forecasting-rnn
```

### **Running the Notebook**
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook air_quality_analysis.ipynb
   ```
2. Execute the cells sequentially to process the dataset, build models, and generate results.

## **Results & Discussion**
- The trained **LSTM model** achieves a **test accuracy of over 75%**.
- Predicted values closely match actual values, demonstrating strong forecasting ability.
- Visualizations provide insight into model performance and areas for improvement.


## **License**
This project is licensed under the **MIT License**.

---

Feel free to contribute to this project by suggesting improvements, adding new forecasting models, or improving the dataset preprocessing pipeline!

