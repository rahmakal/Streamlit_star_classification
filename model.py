import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import pickle
df=pd.read_csv("data\Stars.csv")
df=pd.get_dummies(df,columns=["Color","Spectral_Class"])
x=df.drop('Type',axis=1)
y=df['Type']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train= scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())

svm_model = SVC(random_state=42)
svm_model.fit(x_train, y_train)
y_pred_svm = svm_model.predict(x_test)
print("SVM Model Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Model Classification Report:\n", classification_report(y_test, y_pred_svm))
with open('star_model.pkl','wb') as file:
    pickle.dump(svm_model,file)
