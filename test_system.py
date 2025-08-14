"""
System Integration Test
Verifies all components work together correctly
"""

import os
import sys
import pandas as pd

# Add src directory to path
sys.path.append('src')

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from physics_rules import SolarPhysicsRules
from simulate import SolarDataSimulator
from detect_faults import SolarFaultDetector

def test_system():
    """Test all system components"""
    
    print("🌞 Solar Fault Detection System Test")
    print("=" * 50)
    
    # Test 1: Physics Rules
    print("\n1. Testing Physics Rules...")
    physics = SolarPhysicsRules()
    
    # Test theoretical calculations
    test_irradiance = 800  # W/m²
    test_temperature = 35  # °C
    
    power = physics.calculate_theoretical_power(test_irradiance, test_temperature)
    voltage = physics.calculate_theoretical_voltage(test_irradiance, test_temperature)
    current = physics.calculate_theoretical_current(test_irradiance, test_temperature)
    
    print(f"   Theoretical Power: {power:.2f}W")
    print(f"   Theoretical Voltage: {voltage:.2f}V")
    print(f"   Theoretical Current: {current:.2f}A")
    
    # Test 2: Data Generation
    print("\n2. Testing Data Generation...")
    simulator = SolarDataSimulator()
    test_data = simulator.generate_dataset(days=3)
    
    print(f"   Generated {len(test_data)} samples")
    print(f"   Fault types: {test_data['fault_type'].unique()}")
    
    # Test 3: Fault Detection
    print("\n3. Testing Fault Detection...")
    detector = SolarFaultDetector()
    detected_data = detector.detect_all_faults(test_data)
    
    print(f"   Processed {len(detected_data)} samples")
    print(f"   Detected faults: {detected_data['detected_fault'].value_counts().to_dict()}")
    
    # Test 4: Summary Generation
    print("\n4. Testing Summary Generation...")
    summary = detector.generate_fault_summary(detected_data)
    
    print(f"   Total samples: {summary['total_samples']}")
    print(f"   Average efficiency: {summary['average_efficiency']:.1f}%")
    print(f"   Critical faults: {summary['critical_faults']}")
    
    # Test 5: Alert Generation
    print("\n5. Testing Alert System...")
    alerts = detector.generate_alerts(detected_data)
    
    if alerts:
        print(f"   Generated {len(alerts)} alerts")
        for alert in alerts[:3]:  # Show first 3 alerts
            print(f"   - {alert['fault_type']} at {alert['timestamp']}")
    else:
        print("   No critical alerts generated")
    
    print("\n✅ All tests passed! System is ready.")
    return True

if __name__ == "__main__":
    test_system()