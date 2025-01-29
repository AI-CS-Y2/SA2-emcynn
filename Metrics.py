import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score, davies_bouldin_score, roc_curve, auc, silhouette_samples, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Dataset initialization (extracted from the excel file)
data = {
    "EMIRATE": [
        "Abu Dhabi", "Abu Dhabi", "Abu Dhabi", "Abu Dhabi", "Abu Dhabi", "Abu Dhabi",
        "Dubai", "Dubai", "Dubai", "Dubai", "Dubai", "Dubai",
        "Sharjah", "Sharjah", "Sharjah", "Sharjah", "Sharjah", "Sharjah",
        "Ajman", "Ajman", "Ajman", "Ajman", "Ajman", "Ajman",
        "Umm Al Quwain", "Umm Al Quwain", "Umm Al Quwain", "Umm Al Quwain", "Umm Al Quwain", "Umm Al Quwain",
        "Fujairah", "Fujairah", "Fujairah", "Fujairah", "Fujairah", "Fujairah",
        "Ras Al Khaimah", "Ras Al Khaimah", "Ras Al Khaimah", "Ras Al Khaimah", "Ras Al Khaimah", "Ras Al Khaimah"
    ],
    "CAUSE": [
        "Failure to Yield", "Reckless and Hazardous Driving", "Traffic Signal and Lane Violations", "Distracted or Inattentive Driving",
        "Intoxicated Driving", "Environmental and Mechanical Risks",
        "Failure to Yield", "Reckless and Hazardous Driving", "Traffic Signal and Lane Violations", "Intoxicated Driving",
        "Distracted or Inattentive Driving", "Environmental and Mechanical Risks",
        "Failure to Yield", "Distracted or Inattentive Driving", "Reckless and Hazardous Driving", "Intoxicated Driving",
        "Environmental and Mechanical Risks", "Traffic Signal and Lane Violations",
        "Distracted or Inattentive Driving", "Failure to Yield", "Reckless and Hazardous Driving", "Traffic Signal and Lane Violations",
        "Intoxicated Driving", "Environmental and Mechanical Risks",
        "Distracted or Inattentive Driving", "Failure to Yield", "Intoxicated Driving", "Reckless and Hazardous Driving",
        "Traffic Signal and Lane Violations", "Environmental and Mechanical Risks",
        "Failure to Yield", "Reckless and Hazardous Driving", "Distracted or Inattentive Driving", "Traffic Signal and Lane Violations",
        "Intoxicated Driving", "Environmental and Mechanical Risks",
        "Failure to Yield", "Distracted or Inattentive Driving", "Reckless and Hazardous Driving", "Traffic Signal and Lane Violations",
        "Intoxicated Driving", "Environmental and Mechanical Risks"
    ],
    "FREQUENCY": [
        963, 330, 283, 260, 87, 17,
        735, 355, 95, 82, 70, 8,
        248, 178, 64, 14, 11, 5,
        91, 48, 28, 3, 1, 0,
        25, 11, 2, 1, 0, 0,
        44, 39, 33, 6, 4, 2,
        56, 79, 25, 13, 5, 3
    ]
}

df = pd.DataFrame(data)

# Encode categorical features
label_encoder = LabelEncoder()
df['EMIRATE'] = label_encoder.fit_transform(df['EMIRATE'])
df['CAUSE'] = label_encoder.fit_transform(df['CAUSE'])

# Define severity levels (0: Low, 1: Medium, 2: High in labels)
df['SEVERITY'] = pd.cut(df['FREQUENCY'], bins=[-1, 5, 91, float('inf')], labels=[0, 1, 2])  

# Features and target
X = df[['EMIRATE', 'CAUSE', 'FREQUENCY']]
y = df['SEVERITY']

# Train-test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# SUPERVISED ML MODEL ( RANDOM FOREST CLASSIFIER )
# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
# Confusion Matrix
y_pred = rf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovr')

# ROC Curve for each class
plt.figure(figsize=(8, 6))
for i in range(3):  # 3 classes: Low, Medium, High
    fpr, tpr, _ = roc_curve(y_test == i, rf.predict_proba(X_test)[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()

# UNSUPERVISED ML MODEL ( K-MEANS CLUSTERING )
# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to find optimal clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Visualize Silhouette Plot
cluster_labels = kmeans.labels_
silhouette_vals = silhouette_samples(X_scaled, cluster_labels)

plt.figure(figsize=(8, 6))
y_lower, y_upper = 0, 0
for i in range(4):  # 3 clusters
    cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
    plt.text(-0.05, (y_lower + y_upper) / 2, str(i))
    y_lower += len(cluster_silhouette_vals)

plt.xlabel('Silhouette Coefficient')
plt.ylabel('Cluster')
plt.title('Silhouette Plot - K-Means')
plt.show()

# Optimal clusters (3 for severity levels)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Evaluate clustering performance
silhouette = silhouette_score(X_scaled, kmeans.labels_)
davies_bouldin = davies_bouldin_score(X_scaled, kmeans.labels_)


# COMPARE RESULTS 
# Metrics Table
comparison = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Silhouette Score', 'Davies-Bouldin Index'],
    'Random Forest': [
        round(accuracy, 3),  # Accuracy
        round(precision, 3),  # Precision
        round(recall, 3),  # Recall
        round(f1, 3),  # F1 Score
        round(roc_auc, 3),  # ROC-AUC
        None,  # Silhouette Score (not applicable)
        None  # Davies-Bouldin Index (not applicable)
    ],
    'K-Means': [
        None,  # Accuracy (not applicable)
        None,  # Precision (not applicable)
        None,  # Recall (not applicable)
        None,  # F1 Score (not applicable)
        None,  # ROC-AUC (not applicable)
        round(silhouette, 3),  # Silhouette Score
        round(davies_bouldin, 3)  # Davies-Bouldin Index
    ]
}

df_comparison = pd.DataFrame(comparison)
print(df_comparison)

# Algorithm Graphs 
metrics = ['Accuracy', 'Silhouette Score']
values = [accuracy, silhouette]

plt.bar(metrics, values, color=['blue', 'orange'])
plt.ylabel('Score')
plt.title('Model Comparison: Random Forest vs. K-Means')
plt.ylim(0, 1)
plt.show()