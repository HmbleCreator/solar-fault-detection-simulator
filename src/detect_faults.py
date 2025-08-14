"""
Solar Panel Fault Detection System
Uses physics-informed rules and statistical methods to detect various fault types
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.physics_rules import SolarPhysicsRules

class SolarFaultDetector:
    """Advanced fault detection system using physics-informed rules"""
    
    def __init__(self):
        self.physics_engine = SolarPhysicsRules()
        
    def detect_all_faults(self, data):
        """Detect all types of faults in the dataset"""
        
        results = []
        
        for idx, row in data.iterrows():
            
            # Get physics-based anomalies
            anomalies = self.physics_engine.detect_anomalies(
                row['voltage'], row['current'], row['power'], 
                row['irradiance'], row['temperature']
            )
            
            # Calculate performance metrics
            efficiency = self.physics_engine.calculate_efficiency(
                row['power'], row['irradiance'], row['temperature']
            )
            
            mpp_efficiency = self.physics_engine.calculate_mpp_tracking_efficiency(
                row['voltage'], row['current'], row['irradiance'], row['temperature']
            )
            
            # Estimate shading loss
            shading_loss = self.physics_engine.estimate_shading_loss(
                row['irradiance'], row['irradiance'] * 1.1  # Assume 10% higher expected
            )
            
            # Determine fault type based on patterns
            fault_type = self.classify_fault(
                row['voltage'], row['current'], row['power'],
                row['irradiance'], row['temperature'],
                anomalies, efficiency
            )
            
            # Calculate fault severity
            severity = self.calculate_fault_severity(anomalies, efficiency)
            
            # Calculate theoretical power for reference
            theoretical_power = self.physics_engine.calculate_theoretical_power(
                row['irradiance'], row['temperature']
            )
            
            results.append({
                'timestamp': row['timestamp'],
                'voltage': row['voltage'],
                'current': row['current'],
                'power': row['power'],
                'theoretical_power': theoretical_power,
                'irradiance': row['irradiance'],
                'temperature': row['temperature'],
                'efficiency': efficiency,
                'mpp_efficiency': mpp_efficiency,
                'shading_loss': shading_loss,
                'voltage_fault': anomalies['voltage_fault'],
                'current_fault': anomalies['current_fault'],
                'power_fault': anomalies['power_fault'],
                'efficiency_low': anomalies['efficiency_low'],
                'voltage_deviation': anomalies['voltage_deviation'],
                'current_deviation': anomalies['current_deviation'],
                'power_deviation': anomalies['power_deviation'],
                'detected_fault': fault_type,
                'fault_severity': severity
            })
        
        return pd.DataFrame(results)
    
    def classify_fault(self, voltage, current, power, irradiance, temperature, anomalies, efficiency):
        """Classify the type of fault based on patterns and physics rules"""
        
        # Calculate theoretical values
        theoretical_power = self.physics_engine.calculate_theoretical_power(irradiance, temperature)
        theoretical_voltage = self.physics_engine.calculate_theoretical_voltage(irradiance, temperature)
        theoretical_current = self.physics_engine.calculate_theoretical_current(irradiance, temperature)
        
        # Voltage-to-current ratio for pattern analysis
        v_ratio = voltage / current if current > 0 else 0
        v_ratio_theoretical = theoretical_voltage / theoretical_current if theoretical_current > 0 else 0
        
        # Power factor analysis
        power_factor = power / (voltage * current) if (voltage * current) > 0 else 0
        
        # Fault classification logic
        if efficiency < 60:
            if anomalies['voltage_deviation'] > 0.3 and anomalies['current_deviation'] < 0.1:
                return 'Hotspot'
            elif anomalies['voltage_deviation'] > 0.2 and anomalies['current_deviation'] > 0.2:
                return 'Bypass_Diode_Failure'
            elif anomalies['current_deviation'] > 0.3:
                return 'Shading'
            else:
                return 'Degradation'
        elif efficiency < 80:
            if anomalies['voltage_deviation'] > 0.15:
                return 'Soiling'
            elif anomalies['current_deviation'] > 0.2:
                return 'Partial_Shading'
            else:
                return 'Performance_Degradation'
        elif anomalies['voltage_fault'] or anomalies['current_fault'] or anomalies['power_fault']:
            return 'Minor_Anomaly'
        else:
            return 'Normal'
    
    def calculate_fault_severity(self, anomalies, efficiency):
        """Calculate the severity level of detected faults"""
        
        # Base severity on efficiency drop
        efficiency_severity = max(0, (100 - efficiency) / 40)  # 0-1 scale
        
        # Add severity from deviations
        deviation_severity = max(
            anomalies['voltage_deviation'],
            anomalies['current_deviation'],
            anomalies['power_deviation']
        )
        
        # Combined severity score
        combined_severity = min(1.0, (efficiency_severity + deviation_severity) / 2)
        
        # Convert to severity levels
        if combined_severity < 0.2:
            return 'Low'
        elif combined_severity < 0.5:
            return 'Medium'
        elif combined_severity < 0.8:
            return 'High'
        else:
            return 'Critical'
    
    def generate_fault_summary(self, detected_data):
        """Generate summary statistics for detected faults"""
        
        summary = {
            'total_samples': len(detected_data),
            'normal_samples': len(detected_data[detected_data['detected_fault'] == 'Normal']),
            'fault_samples': len(detected_data[detected_data['detected_fault'] != 'Normal']),
            'fault_types': detected_data['detected_fault'].value_counts().to_dict(),
            'severity_distribution': detected_data['fault_severity'].value_counts().to_dict(),
            'average_efficiency': detected_data['efficiency'].mean(),
            'efficiency_std': detected_data['efficiency'].std(),
            'critical_faults': len(detected_data[detected_data['fault_severity'] == 'Critical']),
            'high_severity_faults': len(detected_data[detected_data['fault_severity'] == 'High'])
        }
        
        return summary
    
    def generate_alerts(self, detected_data):
        """Generate actionable alerts for critical faults"""
        
        alerts = []
        
        # Get recent critical faults
        critical_faults = detected_data[detected_data['fault_severity'] == 'Critical']
        
        for _, fault in critical_faults.iterrows():
            alert = {
                'timestamp': fault['timestamp'],
                'fault_type': fault['detected_fault'],
                'efficiency': fault['efficiency'],
                'power_loss': max(0, fault['power'] * (1 - fault['efficiency']/100)),
                'recommendation': self.get_recommendation(fault['detected_fault'])
            }
            alerts.append(alert)
        
        return alerts
    
    def get_recommendation(self, fault_type):
        """Provide recommendations based on fault type"""
        
        recommendations = {
            'Hotspot': 'Immediate inspection required. Check for hot spots and potential fire hazards.',
            'Bypass_Diode_Failure': 'Check bypass diodes and replace if necessary. Monitor for string mismatch.',
            'Shading': 'Inspect for shading sources. Consider panel repositioning or trimming vegetation.',
            'Soiling': 'Schedule cleaning. Check for dust accumulation or debris.',
            'Degradation': 'Perform IV curve analysis. Consider panel replacement if degradation >20%.',
            'Partial_Shading': 'Identify and remove shading sources. Check for bypass diode functionality.',
            'Performance_Degradation': 'Conduct comprehensive system inspection including connections and inverter.',
            'Minor_Anomaly': 'Monitor closely. Check for loose connections or measurement errors.'
        }
        
        return recommendations.get(fault_type, 'Perform detailed system inspection.')

if __name__ == "__main__":
    # Example usage
    from simulate import SolarDataSimulator
    
    # Generate sample data
    simulator = SolarDataSimulator()
    data = simulator.generate_dataset(days=7)
    
    # Detect faults
    detector = SolarFaultDetector()
    detected_data = detector.detect_all_faults(data)
    
    # Print summary
    summary = detector.generate_fault_summary(detected_data)
    print("Fault Detection Summary:")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Fault samples: {summary['fault_samples']}")
    print(f"Average efficiency: {summary['average_efficiency']:.2f}%")
    print("\nFault types:")
    for fault, count in summary['fault_types'].items():
        print(f"  {fault}: {count}")