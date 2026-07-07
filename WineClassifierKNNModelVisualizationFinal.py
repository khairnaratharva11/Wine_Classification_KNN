import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix, classification_report

def ASKClassifier(datapath):
    Border = "-"*60

    #-----------------------------------------------------------
    # Step 01 : Load the Dataset from CSV file
    #-----------------------------------------------------------

    print(Border)
    print("Step 01 : Load the Dataset from CSV file ")
    print(Border)

    df = pd.read_csv(datapath)
    print("Some Entries from the Data are : ")
    print(df.head())

    #-----------------------------------------------------------
    # Step 02 : Clean the Dataset from Removing empty rows
    #-----------------------------------------------------------

    print(Border)
    print("Step 02 : Clean the Dataset from Removing empty rows")
    print(Border)

    df.dropna(inplace=True)

    print("Total records : ",df.shape[0])
    print("Total columns : ",df.shape[1])

    print(Border)

    #-----------------------------------------------------------
    # Step 03 : Separate Independent and Dependent Variable
    #-----------------------------------------------------------

    print(Border)
    print("Step 03 : Separate Independent and Dependent Variable")
    print(Border)

    X = df.drop(columns=['Class'])
    Y = df['Class']

    print("Shape of X : ",X.shape)
    print("Shape of Y : ",Y.shape)  

    print(Border)
    print("Input Columns : ",X.columns.tolist())
    print("Output Column : Class")
    print(Border)

    # Step 04 : Split the dataset for training and testing
    #-----------------------------------------------------------

    print(Border)
    print("Step 04 : Split the dataset for training and testing  ")
    print(Border)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42, stratify=Y) 

    print(Border)
    print("Information of training and testing data")
    print("X_train shape : ",X_train.shape)
    print("X_test shape : ",X_test.shape)
    print("Y_train shape : ",Y_train.shape)
    print("Y_test shape : ",Y_test.shape)
    
    #-----------------------------------------------------------
    # Step 05 : Feature Scaling
    #-----------------------------------------------------------

    print(Border)
    print("Step 05 : Feature Scaling")
    print(Border)

    scalar = StandardScaler()
    # Independent Variable Scaling
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    print("Feature scaling is done :)")
    
    #-----------------------------------------------------------
    # Step 06 : Explore the multiple values of k
    #          Hyperparameter tuning(k)
    #-----------------------------------------------------------

    print(Border)
    print("Step 06 : Explore the multiple values of k")
    print(Border)

    accuracy_scores = []
    K_values = range(1,21)
 
    for k in K_values:
        model = KNeighborsClassifier(n_neighbors= k)
        model.fit(X_train_scaled,Y_train)
        Y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(Y_test,Y_pred)
        accuracy_scores.append(accuracy)

    print(Border)
    print("Accuracy report of all K values from 1 to 20 :")
    for value in accuracy_scores:
        print(value)
    print(Border)

    #-----------------------------------------------------------
    # Step 07 : Plot graph of K VS Accuracy
    #-----------------------------------------------------------

    print(Border)
    print("Step 07 : Plot graph of K VS Accuracy")
    print(Border)

    plt.figure(figsize=(8,5))
    plt.plot(K_values, accuracy_scores, marker = 'o')
    plt.title("K Values VS Accuracy")
    plt.xlabel("Value of K")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(list(K_values))
    plt.show()


    #-----------------------------------------------------------
    # Step 08 : Find Best Value of K
    #-----------------------------------------------------------

    print(Border)
    print("Step 08 : Find Best Value of K")
    print(Border)

    best_k = list(K_values)[accuracy_scores.index(max(accuracy_scores))]
    print("Best value of K is : ", best_k)

    #-----------------------------------------------------------
    # Step 09 : Build Final model using best value of k
    #-----------------------------------------------------------

    print(Border)
    print("Step 09 : Build Final model using best value of k")
    print(Border)

    final_model = KNeighborsClassifier(n_neighbors= best_k)
    final_model.fit(X_train_scaled,Y_train)
    Y_pred = final_model.predict(X_test_scaled)


    #-----------------------------------------------------------
    # Step 10 : Calculate final accuracy
    #-----------------------------------------------------------

    print(Border)
    print("Step 10 : Calculate final accuracy")
    print(Border)

    accuracy = accuracy_score(Y_test,Y_pred)

    print("Accuracy of model is : ",accuracy*100)


    #-----------------------------------------------------------
    # Step 11 : Display Confusion Matrix
    #-----------------------------------------------------------

    print(Border)
    print("Step 11 : Display Confusion Matrix")
    print(Border)

    cm = confusion_matrix(Y_test,Y_pred)

    print(cm)


    #-----------------------------------------------------------
    # Step 12 : Display classification report
    #-----------------------------------------------------------

    print(Border)
    print("Step 12 : Display classification report")
    print(Border)

    print(classification_report(Y_test,Y_pred))

def main():
    
    Border = "-"*60

    print(Border)
    print("Wine Classifier using KNN")
    print(Border)

    ASKClassifier("WinePredictor.csv")

if __name__ == "__main__":
    main()