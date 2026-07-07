# Wine Classification using K-Nearest Neighbors (KNN)

## 📌 Overview
This repository contains a structured, step-by-step machine learning project that classifies different varieties of wine based on their chemical composition. The project utilizes the **K-Nearest Neighbors (KNN)** algorithm and focuses heavily on crucial ML concepts such as feature scaling, hyperparameter tuning (finding the optimal value of K), and model evaluation.

## 📊 The Dataset
The project utilizes the `WinePredictor.csv` dataset. It includes various chemical constituents found in wine grown in a specific region in Italy, derived from three different cultivars (classes). Features include attributes like Alcohol, Malic acid, Ash, Alcalinity of ash, Magnesium, Total phenols, Flavanoids, and more.

## 🚀 Incremental Approach & Project Structure
The code is divided into five scripts, demonstrating a clear evolution from basic data handling to an optimized, finalized model:

* **`WineClassifierKNN1.py` (Data Loading):** The foundational step. Loads the CSV data using Pandas and performs basic data cleaning by dropping any empty rows to ensure data integrity.
* **`WineClassifierKNN2.py` (Feature Separation):** Builds upon the first script by isolating the independent variables (chemical features) into `X` and the dependent target variable (`Class`) into `Y`.
* **`WineClassifierKNNModel.py` (Scaling & Tuning):** Introduces critical machine learning steps. It splits the data into training and testing sets, applies `StandardScaler` (essential for distance-based algorithms like KNN), and iterates through K values from 1 to 20 to calculate varying accuracy scores.
* **`WineClassifierKNNModelVisualization.py` (Visualization):** Enhances the tuning process by utilizing `matplotlib` to plot the K values against their corresponding accuracy scores, making it visually intuitive to identify the optimal K.
* **`WineClassifierKNNModelVisualizationFinal.py` (The Final Pipeline):** The culmination of the project. It builds the final KNN model using the best identified value of K, and evaluates its performance using a confusion matrix and a comprehensive classification report.

## 🛠️ Technologies Used
* **Language:** Python 3
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn (`KNeighborsClassifier`, `StandardScaler`, `train_test_split`)
* **Visualization:** Matplotlib

## ⚙️ How to Run
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed: `pip install pandas scikit-learn matplotlib`.
3. Make sure the `WinePredictor.csv` dataset is in the same directory as the scripts.
4. Run the scripts sequentially to observe the pipeline evolution. To view the hyperparameter tuning graph, run:
   ```bash
   python WineClassifierKNNModelVisualization.py
