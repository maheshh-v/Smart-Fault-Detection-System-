import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_transformer_data(num_samples=5000):
    """
    Generate realistic transformer sensor data for fault detection
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + timedelta(hours=i) for i in range(num_samples)]
    
    data = []
    
    for i, timestamp in enumerate(timestamps):
        # Determine fault status (80% normal, 15% warning, 5% critical)
        fault_prob = random.random()
        if fault_prob < 0.80:
            fault_status = 'Normal'
        elif fault_prob < 0.95:
            fault_status = 'Warning'
        else:
            fault_status = 'Critical'
        
        # Generate realistic sensor readings based on fault status
        if fault_status == 'Normal':
            voltage = np.random.normal(230, 5)  # Normal voltage around 230kV
            current = np.random.normal(150, 20)  # Normal current
            oil_temp = np.random.normal(65, 8)   # Normal oil temperature
            vibration = np.random.normal(0.5, 0.1)  # Low vibration
            oil_level = np.random.normal(95, 3)   # Good oil level
            load_factor = np.random.normal(75, 15)  # Normal load
            ambient_temp = np.random.normal(25, 10)  # Ambient temperature
            
        elif fault_status == 'Warning':
            voltage = np.random.normal(225, 8)   # Slightly low voltage
            current = np.random.normal(180, 25)  # Higher current
            oil_temp = np.random.normal(75, 10)  # Higher oil temperature
            vibration = np.random.normal(0.8, 0.2)  # Increased vibration
            oil_level = np.random.normal(88, 5)   # Lower oil level
            load_factor = np.random.normal(85, 20)  # Higher load
            ambient_temp = np.random.normal(30, 12)  # Higher ambient temp
            
        else:  # Critical
            voltage = np.random.normal(210, 12)  # Low voltage
            current = np.random.normal(220, 30)  # High current
            oil_temp = np.random.normal(90, 15)  # High oil temperature
            vibration = np.random.normal(1.5, 0.3)  # High vibration
            oil_level = np.random.normal(80, 8)   # Low oil level
            load_factor = np.random.normal(95, 25)  # Very high load
            ambient_temp = np.random.normal(35, 15)  # High ambient temp
        
        # Add some realistic constraints and noise
        voltage = max(200, min(250, voltage))
        current = max(50, min(300, current))
        oil_temp = max(40, min(120, oil_temp))
        vibration = max(0.1, min(3.0, vibration))
        oil_level = max(70, min(100, oil_level))
        load_factor = max(30, min(120, load_factor))
        ambient_temp = max(-10, min(50, ambient_temp))
        
        data.append({
            'timestamp': timestamp,
            'voltage_kv': round(voltage, 2),
            'current_a': round(current, 1),
            'oil_temperature_c': round(oil_temp, 1),
            'vibration_mm_s': round(vibration, 2),
            'oil_level_percent': round(oil_level, 1),
            'load_factor_percent': round(load_factor, 1),
            'ambient_temperature_c': round(ambient_temp, 1),
            'fault_status': fault_status
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate the dataset
    df = generate_transformer_data(5000)
    
    # Save to CSV
    import os
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/transformer_sensor_data.csv', index=False)
    print(f"Generated dataset with {len(df)} samples")
    print(f"Fault distribution:")
    print(df['fault_status'].value_counts())
    print(f"Dataset saved to 'data/transformer_sensor_data.csv'")
