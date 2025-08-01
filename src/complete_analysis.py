#!/usr/bin/env python3
"""
Smart Fault Detection System - Complete Analysis Script
Author: Mahesh
Date: 2025-01-01

This script performs comprehensive analysis of transformer sensor data
for fault detection using machine learning techniques.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_curve, auc
)
import joblib
import os

def load_and_explore_data():
    """Load and perform initial exploration of the dataset"""
    print("=" * 60)
    print("🔍 LOADING AND EXPLORING TRANSFORMER SENSOR DATA")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv('data/transformer_sensor_data.csv')
    
    print(f"📊 Dataset Shape: {df.shape}")
    print(f"📅 Time Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\n📈 Fault Status Distribution:")
    fault_counts = df['fault_status'].value_counts()
    for status, count in fault_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   • {status}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n🔍 Missing Values: {df.isnull().sum().sum()}")
    print(f"📊 Data Types: {df.dtypes.nunique()} unique types")
    
    return df

def feature_engineering(df):
    """Perform feature engineering on the dataset"""
    print("\n" + "=" * 60)
    print("🔧 FEATURE ENGINEERING")
    print("=" * 60)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Rolling averages (24-hour window)
    window_size = 24
    sensor_cols = ['voltage_kv', 'current_a', 'oil_temperature_c', 'vibration_mm_s']
    
    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std()
        df[f'{col}_change_rate'] = df[col].diff().fillna(0)
    
    # Derived features
    df['temp_difference'] = df['oil_temperature_c'] - df['ambient_temperature_c']
    df['power_estimation'] = df['voltage_kv'] * df['current_a'] * df['load_factor_percent'] / 100
    df['efficiency_indicator'] = df['load_factor_percent'] / (df['oil_temperature_c'] - df['ambient_temperature_c'] + 1)
    
    print(f"✅ Feature Engineering Complete!")
    print(f"📊 New Dataset Shape: {df.shape}")
    print(f"🔧 Features Added: {df.shape[1] - 9} new features")
    
    return df

def create_visualizations(df):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 60)
    print("📊 CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Fault Status Distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    fault_counts = df['fault_status'].value_counts()
    colors = ['#2ecc71', '#f39c12', '#e74c3c']  # Green, Orange, Red
    
    # Bar plot
    axes[0].bar(fault_counts.index, fault_counts.values, color=colors)
    axes[0].set_title('Fault Status Distribution (Count)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(fault_counts.values):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    axes[1].pie(fault_counts.values, labels=fault_counts.index, autopct='%1.1f%%', 
               colors=colors, startangle=90)
    axes[1].set_title('Fault Status Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('models/fault_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Sensor readings by fault status
    sensor_cols = ['voltage_kv', 'current_a', 'oil_temperature_c', 'vibration_mm_s', 
                   'oil_level_percent', 'load_factor_percent']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, col in enumerate(sensor_cols):
        sns.boxplot(data=df, x='fault_status', y=col, ax=axes[i], 
                    order=['Normal', 'Warning', 'Critical'])
        axes[i].set_title(f'{col.replace("_", " ").title()}', fontweight='bold')
        axes[i].set_xlabel('Fault Status')
        
        # Add mean values
        means = df.groupby('fault_status')[col].mean()
        for j, status in enumerate(['Normal', 'Warning', 'Critical']):
            if status in means.index:
                axes[i].text(j, means[status], f'{means[status]:.1f}', 
                            ha='center', va='bottom', fontweight='bold', 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('models/sensor_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Correlation heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                center=0, square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Sensor Data Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('models/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations created and saved!")
    
    return df

def train_models(df):
    """Train and evaluate machine learning models"""
    print("\n" + "=" * 60)
    print("🤖 MACHINE LEARNING MODEL TRAINING")
    print("=" * 60)
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'fault_status']]
    X = df[feature_cols]
    y = df['fault_status']
    
    print(f"📊 Feature Matrix Shape: {X.shape}")
    print(f"📊 Target Vector Shape: {y.shape}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"📊 Training Set: {X_train.shape[0]} samples")
    print(f"📊 Test Set: {X_test.shape[0]} samples")
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    }
    
    model_results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {name}...")
        
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"✅ {name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Select best model
    best_model_name = 'Random Forest'
    best_model = model_results[best_model_name]['model']
    best_predictions = model_results[best_model_name]['predictions']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 Final Accuracy: {model_results[best_model_name]['accuracy']:.4f}")
    
    # Classification report
    print("\n📋 Detailed Classification Report:")
    target_names = label_encoder.classes_
    print(classification_report(y_test, best_predictions, target_names=target_names))
    
    # Feature importance
    if best_model_name == 'Random Forest':
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🔍 Top 10 Most Important Features:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"   • {row['feature']}: {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(15)
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, best_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Fault Detection Model', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model and preprocessing objects
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/fault_detection_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(feature_cols, 'models/feature_names.pkl')
    
    print("\n✅ Model and preprocessing objects saved!")
    
    return best_model, scaler, label_encoder, feature_cols, model_results

def main():
    """Main analysis pipeline"""
    print("🚀 SMART FAULT DETECTION SYSTEM - COMPLETE ANALYSIS")
    print("=" * 80)
    print(f"⏰ Analysis Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Create visualizations
    df = create_visualizations(df)
    
    # Train models
    model, scaler, label_encoder, feature_cols, results = train_models(df)
    
    print("\n" + "=" * 80)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 80)
    print("📊 Key Results:")
    print(f"   • Best Model: Random Forest")
    print(f"   • Accuracy: {results['Random Forest']['accuracy']:.4f}")
    print(f"   • F1-Score: {results['Random Forest']['f1_score']:.4f}")
    print(f"   • Features Used: {len(feature_cols)}")
    print(f"   • Dataset Size: {len(df)} samples")
    print("\n📁 Files Generated:")
    print("   • models/fault_detection_model.pkl")
    print("   • models/scaler.pkl")
    print("   • models/label_encoder.pkl")
    print("   • models/feature_names.pkl")
    print("   • Visualization plots saved as PNG files")
    print(f"\n⏰ Analysis Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
