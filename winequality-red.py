# import libraries

import pandas as pd
import numpy as np
import scipy

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report


def read_eda(filename):
    wine_df = pd.read_csv(filename)
    print(wine_df.shape)
    print(wine_df.head())
    print(wine_df.info())
    print(wine_df.columns)
    
    return wine_df
    
def eda(df):
    
    plot_columns =list(wine_df.columns)
    plot_columns.remove('quality')
    
    for col in plot_columns:
        
        sns.barplot(x='quality',y=col,data = df)
        plt.title(f'{col} vs quality')
        plt.show()
        
        sns.scatterplot(x='quality',y=col,data = df)
        plt.show()
    
    sns.countplot(df['quality'])
    plt.show()
    

def model(wine_df):
    y = wine_df['quality']
    X = wine_df.drop('quality',axis =1)
    
    train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = 0.3,
                                                     shuffle = True,stratify = y)
    
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.fit_transform(test_X)
    
    # applying various models
    # 1. Random forest classifier
    
    rf = RandomForestClassifier()
    rf.fit(train_X_scaled,train_y)
    pred_rf = rf.predict(test_X_scaled)
    print(classification_report(test_y,pred_rf))
    
    # 2. SVC
    
    svc= SVC()
    svc.fit(train_X_scaled,train_y)
    svc_pred = svc.predict(test_X_scaled)
    print(classification_report(test_y, svc_pred))
    
    # 3 Decesion tree classifier
    
    tree = DecisionTreeClassifier()
    tree.fit(train_X,train_y)
    pred_tree = tree.predict(test_X)
    print(classification_report(test_y, pred_tree))
    
    
if __name__=="__main__":
    
    wine_df = read_eda('winequality-red.csv')
    #eda(wine_df)
    
    # modifying the problem as a binary classification problem
    bins = (2,5,8)
    group_name = ['bad','good']
    wine_df['quality'] = pd.cut(wine_df['quality'], bins = bins,
                                labels=group_name)
    label_encoder = LabelEncoder()
    wine_df['quality'] = label_encoder.fit_transform(wine_df['quality'])
    sns.countplot(wine_df['quality'])
    plt.show()
    
    model(wine_df)
    
    
    