import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report

def ASKClassifier(datapath):
    border = "-"*60

    #-----------------------------------------------------------
    # Step 1 : Load the Dataset from the CSV file
    #-----------------------------------------------------------

    print(border)
    print("Step 1 : Load the Dataset from CSV file ")
    print(border)

    df = pd.read_csv(datapath)
    print("Some Entries from the Data are : ")
    print(df.head())

    #-----------------------------------------------------------
    # Step 2 : Clean the Dataset from Removing empty rows
    #-----------------------------------------------------------

    print(border)
    print("Step 2 : Clean the Dataset from Removing empty rows")
    print(border)

    df.dropna(inplace=True)

    print("Total records : ",df.shape[0])
    print("Total columns : ",df.shape[1])

    print(border)

    #-----------------------------------------------------------
    # Step 3 : Separate Independent and Dependent Variable
    #-----------------------------------------------------------

    print(border)
    print("Step 3 : Separate Independent and Dependent Variable")
    print(border)

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of X : ",X.shape)
    print("Shape of Y : ",Y.shape)  

    print(border)
    print("Input Columns : ",X.columns.tolist())
    print("Output Column : Class")
    print(border)  
            

def main():
    
    border = "-"*60

    print(border)
    print("Wine Classifier using KNN")
    print(border)

    ASKClassifier("WinePredictor.csv")

  
    

if __name__ == "__main__":
    main()