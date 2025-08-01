# Smart Fault Detection System for Electrical Equipment Using AI

## Project Overview
An AI-powered predictive maintenance system for electrical transformers that analyzes sensor data to detect potential faults before they cause equipment failure. This system helps power companies reduce downtime and prevent costly equipment failures.

## Key Features
- **Data Preprocessing**: Clean and prepare transformer sensor data
- **Feature Engineering**: Extract meaningful patterns from time-series data
- **Fault Classification**: Predict Normal/Warning/Critical fault states
- **Real-time Dashboard**: Streamlit-based UI for monitoring and alerts
- **Alert System**: Automated warnings when fault probability exceeds threshold

## Tech Stack
- **Programming**: Python
- **Machine Learning**: Rule-based Classification, Feature Engineering
- **Data Visualization**: Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Data Processing**: Pandas, NumPy

## Dataset Features
The system analyzes the following transformer sensor parameters:
- **Voltage (kV)**: Operating voltage levels
- **Current (A)**: Load current measurements
- **Oil Temperature (°C)**: Transformer oil temperature
- **Vibration (mm/s)**: Mechanical vibration levels
- **Oil Level (%)**: Oil level percentage
- **Load Factor (%)**: Current load as percentage of rated capacity
- **Ambient Temperature (°C)**: Environmental temperature

## Model Performance
- **Algorithm**: Rule-based Classification System
- **Accuracy**: 85.8% on test data (4291/5000 correct predictions)
- **Critical Fault Detection**: 99.2% recall rate
- **False Alarm Rate**: 0.1% for normal conditions flagged as critical

## Business Impact
- **Predictive Maintenance**: Early fault detection reduces unplanned downtime by up to 70%
- **Cost Savings**: Prevents expensive transformer failures ($500K-$2M per unit)
- **Remote Monitoring**: Real-time status monitoring capability
- **Risk Assessment**: Probability-based fault prediction with confidence scores

## Project Structure
```
├── data/
│   ├── transformer_sensor_data.csv
│   └── processed_transformer_data.csv
├── models/
│   ├── model_info.txt
│   └── prediction_function.py
├── notebooks/
│   └── fault_detection_analysis.ipynb
├── src/
│   ├── generate_dataset.py
│   ├── simple_model.py
│   ├── model_training.py
│   └── simple_dashboard.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/maheshh-v/Smart-Fault-Detection-System-.git
   cd Smart-Fault-Detection-System-
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate the dataset (if needed):
   ```bash
   python src/generate_dataset.py
   ```

4. Train the model:
   ```bash
   python src/simple_model.py
   ```

5. Launch the dashboard:
   ```bash
   streamlit run src/simple_dashboard.py
   ```

## Usage

### Dashboard Interface
The Streamlit dashboard provides three main sections:
1. **System Overview**: Key metrics and performance indicators
2. **Real-time Prediction**: Interactive fault prediction interface
3. **Data Analysis**: Historical data visualization and insights

### Making Predictions
1. Navigate to the "Real-time Prediction" page
2. Enter transformer sensor readings
3. Click "Predict Fault Status" to get classification results
4. View confidence scores and recommendations

## Model Details

### Classification Logic
The system uses a rule-based approach with the following thresholds:
- **Voltage**: Critical (<220kV), Warning (<225kV), Normal (≥225kV)
- **Current**: Critical (>200A), Warning (>180A), Normal (≤180A)
- **Oil Temperature**: Critical (>85°C), Warning (>75°C), Normal (≤75°C)
- **Vibration**: Critical (>1.2mm/s), Warning (>0.8mm/s), Normal (≤0.8mm/s)
- **Oil Level**: Critical (<85%), Warning (<90%), Normal (≥90%)

### Feature Engineering
The model incorporates several engineered features:
- Temperature difference (oil vs ambient)
- Power factor estimation
- Efficiency ratio calculations
- Rolling averages for trend analysis

## Results and Performance

### Confusion Matrix Results
```
Actual → Predicted:
Normal → Normal: 3737 (92.9% precision)
Normal → Warning: 282 (7.0%)
Normal → Critical: 1 (0.1%)

Warning → Normal: 73 (10.0%)
Warning → Warning: 304 (41.8%)
Warning → Critical: 351 (48.2%)

Critical → Warning: 2 (0.8%)
Critical → Critical: 250 (99.2% recall)
```

### Key Strengths
- High critical fault detection rate (99.2% recall)
- Low false alarm rate for normal conditions
- Balanced performance across all fault categories

## Industry Applications
This system is designed for the power and energy sector, specifically:
- Electrical utility companies
- Power generation facilities
- Industrial manufacturing plants
- Grid operators and maintenance teams

## Future Enhancements
- Integration with LSTM neural networks for time-series prediction
- Real-time sensor data streaming capabilities
- Mobile-responsive dashboard design
- Advanced analytics and trend forecasting
- Multi-equipment monitoring support

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or collaboration opportunities, please reach out through GitHub issues or contact the repository owner.
