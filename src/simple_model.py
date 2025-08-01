import pandas as pd
import numpy as np
import os

print("🚀 Smart Fault Detection System - Simple Model Training")
print("=" * 60)

# Create models directory
os.makedirs('models', exist_ok=True)

# Load data
try:
    df = pd.read_csv('data/transformer_sensor_data.csv')
    print(f"✅ Dataset loaded successfully: {df.shape}")
except FileNotFoundError:
    print("❌ Dataset not found. Please run generate_dataset.py first.")
    exit(1)

print(f"\n📊 Dataset Overview:")
print(f"   • Total samples: {len(df)}")
print(f"   • Features: {df.shape[1] - 1}")
print(f"   • Time period: {df['timestamp'].min()} to {df['timestamp'].max()}")

print(f"\n📈 Fault Status Distribution:")
fault_counts = df['fault_status'].value_counts()
for status, count in fault_counts.items():
    percentage = (count / len(df)) * 100
    print(f"   • {status}: {count} samples ({percentage:.1f}%)")

# Basic feature engineering
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Add simple derived features
df['temp_difference'] = df['oil_temperature_c'] - df['ambient_temperature_c']
df['power_factor'] = df['voltage_kv'] * df['current_a'] / 1000
df['efficiency_ratio'] = df['load_factor_percent'] / (df['oil_temperature_c'] + 1)

print(f"\n🔧 Feature engineering completed!")
print(f"   • Added derived features: temp_difference, power_factor, efficiency_ratio")

# Simple rule-based model for demonstration
def simple_fault_detector(row):
    """Simple rule-based fault detection"""
    score = 0
    
    # Voltage check
    if row['voltage_kv'] < 220:
        score += 2
    elif row['voltage_kv'] < 225:
        score += 1
    
    # Current check
    if row['current_a'] > 200:
        score += 2
    elif row['current_a'] > 180:
        score += 1
    
    # Temperature check
    if row['oil_temperature_c'] > 85:
        score += 2
    elif row['oil_temperature_c'] > 75:
        score += 1
    
    # Vibration check
    if row['vibration_mm_s'] > 1.2:
        score += 2
    elif row['vibration_mm_s'] > 0.8:
        score += 1
    
    # Oil level check
    if row['oil_level_percent'] < 85:
        score += 2
    elif row['oil_level_percent'] < 90:
        score += 1
    
    # Classify based on score
    if score >= 4:
        return 'Critical'
    elif score >= 2:
        return 'Warning'
    else:
        return 'Normal'

# Apply simple model
df['predicted_status'] = df.apply(simple_fault_detector, axis=1)

# Calculate accuracy
correct_predictions = (df['fault_status'] == df['predicted_status']).sum()
accuracy = correct_predictions / len(df)

print(f"\n🤖 Simple Rule-Based Model Results:")
print(f"   • Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"   • Correct predictions: {correct_predictions}/{len(df)}")

# Confusion matrix (manual calculation)
print(f"\n📊 Prediction Results:")
for actual in ['Normal', 'Warning', 'Critical']:
    for predicted in ['Normal', 'Warning', 'Critical']:
        count = len(df[(df['fault_status'] == actual) & (df['predicted_status'] == predicted)])
        if count > 0:
            print(f"   • Actual {actual} → Predicted {predicted}: {count}")

# Save model parameters (simple dictionary)
model_params = {
    'model_type': 'rule_based',
    'accuracy': accuracy,
    'features_used': ['voltage_kv', 'current_a', 'oil_temperature_c', 'vibration_mm_s', 'oil_level_percent'],
    'thresholds': {
        'voltage_critical': 220,
        'voltage_warning': 225,
        'current_critical': 200,
        'current_warning': 180,
        'temp_critical': 85,
        'temp_warning': 75,
        'vibration_critical': 1.2,
        'vibration_warning': 0.8,
        'oil_level_critical': 85,
        'oil_level_warning': 90
    }
}

# Save as simple text file
with open('models/model_info.txt', 'w') as f:
    f.write("Smart Fault Detection System - Model Information\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model Type: {model_params['model_type']}\n")
    f.write(f"Accuracy: {model_params['accuracy']:.4f}\n")
    f.write(f"Training Data: {len(df)} samples\n")
    f.write(f"Features Used: {', '.join(model_params['features_used'])}\n\n")
    f.write("Thresholds:\n")
    for key, value in model_params['thresholds'].items():
        f.write(f"  {key}: {value}\n")

# Save processed data
df.to_csv('data/processed_transformer_data.csv', index=False)

print(f"\n💾 Files saved:")
print(f"   • models/model_info.txt")
print(f"   • data/processed_transformer_data.csv")

print(f"\n🎉 Model training completed successfully!")
print(f"   • Ready for dashboard deployment")
print(f"   • Model achieves {accuracy*100:.1f}% accuracy")
print(f"   • Suitable for demonstration and further development")

# Create a simple prediction function file
prediction_code = '''
def predict_fault_status(voltage, current, oil_temp, vibration, oil_level):
    """
    Simple fault prediction function
    Returns: fault_status, confidence_score
    """
    score = 0
    
    # Apply thresholds
    if voltage < 220: score += 2
    elif voltage < 225: score += 1
    
    if current > 200: score += 2
    elif current > 180: score += 1
    
    if oil_temp > 85: score += 2
    elif oil_temp > 75: score += 1
    
    if vibration > 1.2: score += 2
    elif vibration > 0.8: score += 1
    
    if oil_level < 85: score += 2
    elif oil_level < 90: score += 1
    
    # Classify
    if score >= 4:
        return 'Critical', min(score/6, 1.0)
    elif score >= 2:
        return 'Warning', min(score/4, 1.0)
    else:
        return 'Normal', max(1 - score/2, 0.5)

# Example usage:
# status, confidence = predict_fault_status(230, 150, 65, 0.5, 95)
# print(f"Status: {status}, Confidence: {confidence:.2f}")
'''

with open('models/prediction_function.py', 'w') as f:
    f.write(prediction_code)

print(f"   • models/prediction_function.py (for easy integration)")
