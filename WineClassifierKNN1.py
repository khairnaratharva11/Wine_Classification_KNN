import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def MarvellousClassifier(Datapath):
    Border = "-"*100
    # Step 1 : Load the dataset from CSV file
    print(Border)
    print("Step 1 : Load the dataset from CSV file")
    print(Border)

    df = pd.read_csv(Datapath)
    print("Some Entries from the Data are : ")
    print(df.head())

    # Step 2 : Clean the dataset by removing empty rows
    print(Border)
    print("Step 2 : Clean the dataset by removing empty rows")
    print(Border)

    df.dropna(inplace=True)
    print("Total Records : ",df.shape[0])
    print("Total Columns : ", df.shape[1])




def main():
    Border = "-"*100
    print(Border)
    print("Wine Classifierr using KNN")
    print(Border)

    MarvellousClassifier("WinePredictor.csv")


if __name__ == "__main__":
    main()