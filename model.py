import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
df=pd.read_csv(r"C:\Users\Rahma Kallel\Desktop\data\Stars.csv")
color_mapping = {
    'Blue-white': 'Blue-White',
    'Blue White': 'Blue-White',
    'Blue white': 'Blue-White',
    'Blue-White': 'Blue-White',
    'white blue ': 'Blue-White',
    'yellow-white':'White',
    'Yellowish White':'White',
    'Whitish':'White',
    'yellowish':'White',
    'Yellowish':'White',
    'white':'White',
    'Yellow':'White',
    'Pale yellow orange':'Orange',
    'White-Yellow':'White',
    'Orange-Red':'Orange'
    # Add more mappings as needed
}

# Replace similar values in the 'Color' column
df['Color'] = df['Color'].replace(color_mapping)
df.to_csv('stars.csv', index=False)
#print(df.describe(),Counter(df["Color"]),Counter(df["Spectral_Class"]))
'''distinct_color_type_pairs = df.groupby(['Color', 'Type']).size().reset_index(name='Count')

# Create a pivot table for better visualization
pivot_table = distinct_color_type_pairs.pivot('Color', 'Type', 'Count').fillna(0)

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='viridis', annot=True, fmt='g', cbar_kws={'label': 'Count'})
plt.title('Distinct Pairs of (Color, Type)')
plt.xlabel('Type')
plt.ylabel('Color')
plt.show()'''
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

# Instantiate the SVM model
svm_model = SVC(random_state=42)

# Train the model
svm_model.fit(x_train, y_train)

# Make predictions on the testing set
y_pred_svm = svm_model.predict(x_test)

# Evaluate the SVM model
print("SVM Model Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Model Classification Report:\n", classification_report(y_test, y_pred_svm))
with open('star_model.pkl','wb') as file:
    pickle.dump(svm_model,file)
