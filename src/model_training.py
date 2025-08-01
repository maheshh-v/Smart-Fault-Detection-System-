import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    import joblib
except ImportError as e:
    print(f"Missing package: {e}")
    print("Please install required packages: pip install scikit-learn matplotlib seaborn")
    exit(1)
import os

# Create models directory
os.makedirs('models', exist_ok=True)

print("🚀 Smart Fault Detection System - Model Training")
print("=" * 50)

# Load data
df = pd.read_csv('data/transformer_sensor_data.csv')
print(f"📊 Dataset loaded: {df.shape}")
print(f"📈 Fault distribution:")
print(df['fault_status'].value_counts())

# Feature engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Add time features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek

# Add rolling averages
window = 24
for col in ['voltage_kv', 'current_a', 'oil_temperature_c', 'vibration_mm_s']:
    df[f'{col}_rolling_mean'] = df[col].rolling(window=window, min_periods=1).mean()

# Add derived features
df['temp_difference'] = df['oil_temperature_c'] - df['ambient_temperature_c']
df['power_factor'] = df['voltage_kv'] * df['current_a'] / 1000

print(f"🔧 Feature engineering complete: {df.shape}")

# Prepare features
feature_cols = [col for col in df.columns if col not in ['timestamp', 'fault_status']]
X = df[feature_cols]
y = df['fault_status']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# Train Random Forest model
print("\n🤖 Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📊 Accuracy: {accuracy:.4f}")

# Classification report
target_names = label_encoder.classes_
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 Top 10 Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   • {row['feature']}: {row['importance']:.4f}")

# Save model and preprocessing objects
joblib.dump(rf_model, 'models/fault_detection_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')
joblib.dump(feature_cols, 'models/feature_names.pkl')

print("\n💾 Model saved successfully!")
print("   • models/fault_detection_model.pkl")
print("   • models/scaler.pkl")
print("   • models/label_encoder.pkl")
print("   • models/feature_names.pkl")

# Create confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Fault Detection Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎉 Analysis complete! Ready for dashboard deployment.")
