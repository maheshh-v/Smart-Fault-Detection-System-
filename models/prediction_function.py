
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
