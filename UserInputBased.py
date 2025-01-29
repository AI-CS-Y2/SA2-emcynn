import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
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
label_encoder_emirate = LabelEncoder()
label_encoder_cause = LabelEncoder()

df['EMIRATE'] = label_encoder_emirate.fit_transform(df['EMIRATE'])
df['CAUSE'] = label_encoder_cause.fit_transform(df['CAUSE'])

# Define severity levels (0: Low, 1: Medium, 2: High)
df['SEVERITY'] = pd.cut(df['FREQUENCY'], bins=[-1, 5, 91, float('inf')], labels=[0, 1, 2])  

# Features and target
X = df[['EMIRATE', 'CAUSE', 'FREQUENCY']]
y = df['SEVERITY']

# Train-test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Standardize features for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# USER INPUT
# Options for causes of accidents
cause_options = {
    "A": "Failure to Yield",
    "B": "Reckless and Hazardous Driving",
    "C": "Traffic Signal and Lane Violations",
    "D": "Distracted or Inattentive Driving",
    "E": "Intoxicated Driving",
    "F": "Environmental and Mechanical Risks"
}

# Ask user input
print("\nENTER ACCIDENT DATA FOR PREDICTION")
emirate_input = input("Enter the Emirate: ")

# Show options for causes
print("\nSelect the Cause of Accident from the options below:")
for key, value in cause_options.items():
    print(f"{key}. {value}")

cause_choice = input("\nEnter your choice (A-F): ").upper()
while cause_choice not in cause_options:
    print("Invalid choice. Please select a valid option.")
    cause_choice = input("Enter your choice (A-F): ").upper()

cause_input = cause_options[cause_choice]

frequency_input = int(input("Number of cases: "))

# Display the user inputs
print("\nROAD ACCIDENT DETAILS:")
print(f"Emirate: {emirate_input}")
print(f"Cause of Accident: {cause_input}")
print(f"Number of Cases: {frequency_input}")

# Encode user input
emirate_encoded = label_encoder_emirate.transform([emirate_input])[0]
cause_encoded = label_encoder_cause.transform([cause_input])[0]

# Create input data
user_data = pd.DataFrame({
    'EMIRATE': [emirate_encoded],
    'CAUSE': [cause_encoded],
    'FREQUENCY': [frequency_input]
})

# PREDICT ACCIDENT SEVERITY 
# Random Forest Prediction
severity_prediction = rf.predict(user_data)
severity_label = ['Low', 'Medium', 'High']
print(f"\nPredicted Severity Level (Random Forest): {severity_label[int(severity_prediction[0])]}")

# K-Means Clustering
user_data_scaled = scaler.transform(user_data)
cluster_label = kmeans.predict(user_data_scaled)

# Define mapping from cluster labels to severity levels
cluster_label_mapping = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# Convert numerical cluster label to severity string
severity_label = cluster_label_mapping[cluster_label[0]]

print(f"Cluster Label (K-Means): {severity_label}")