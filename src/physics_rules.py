"""
Physics-Informed Rules for Solar Panel Performance
Implements PINN-style logic with temperature coefficients and Ohm's law
"""

import numpy as np

class SolarPhysicsRules:
    """Physics-informed rules for solar panel behavior based on real PV characteristics"""
    
    def __init__(self, 
                 rated_power=320,  # W
                 rated_voltage=37.0,  # V
                 rated_current=8.6,  # A
                 temp_coeff_voltage=-0.0035,  # %/°C
                 temp_coeff_current=0.0006,  # %/°C
                 temp_coeff_power=-0.004,  # %/°C
                 irradiance_std=1000):  # W/m²
        
        self.rated_power = rated_power
        self.rated_voltage = rated_voltage
        self.rated_current = rated_current
        self.temp_coeff_voltage = temp_coeff_voltage
        self.temp_coeff_current = temp_coeff_current
        self.temp_coeff_power = temp_coeff_power
        self.irradiance_std = irradiance_std
    
    def calculate_theoretical_voltage(self, irradiance, temperature):
        """Calculate theoretical voltage based on irradiance and temperature"""
        # Voltage decreases with temperature (negative coefficient)
        temp_effect = 1 + (self.temp_coeff_voltage * (temperature - 25))
        # Voltage scales logarithmically with irradiance
        irradiance_effect = np.log(irradiance / self.irradiance_std + 1) / np.log(2)
        
        theoretical_voltage = self.rated_voltage * temp_effect * irradiance_effect
        return max(0, theoretical_voltage)
    
    def calculate_theoretical_current(self, irradiance, temperature):
        """Calculate theoretical current based on irradiance and temperature"""
        # Current increases slightly with temperature (positive coefficient)
        temp_effect = 1 + (self.temp_coeff_current * (temperature - 25))
        # Current scales linearly with irradiance
        irradiance_effect = irradiance / self.irradiance_std
        
        theoretical_current = self.rated_current * temp_effect * irradiance_effect
        return max(0, theoretical_current)
    
    def calculate_theoretical_power(self, irradiance, temperature):
        """Calculate theoretical power based on physics rules"""
        voltage = self.calculate_theoretical_voltage(irradiance, temperature)
        current = self.calculate_theoretical_current(irradiance, temperature)
        
        # Power coefficient effect
        temp_power_effect = 1 + (self.temp_coeff_power * (temperature - 25))
        
        theoretical_power = voltage * current * temp_power_effect
        return max(0, theoretical_power)
    
    def calculate_efficiency(self, actual_power, irradiance, temperature):
        """Calculate panel efficiency with realistic degradation patterns"""
        theoretical_power = self.calculate_theoretical_power(irradiance, temperature)
        if theoretical_power <= 0:
            return np.nan  # Sensor offline indicator
        
        # Simulate realistic degradation with gradual drops
        base_efficiency = (actual_power / theoretical_power) * 100
        
        # Add realistic bounds (5-100%) with sensor noise
        noise = np.random.normal(0, 2)  # ±2% sensor noise
        efficiency = base_efficiency + noise
        
        # Ensure realistic bounds but allow for sensor variations
        if efficiency < 5:
            return max(5.0, efficiency)  # Minimum realistic efficiency
        elif efficiency > 105:
            return min(105.0, efficiency)  # Allow slight over-reading
        else:
            return efficiency
    
    def detect_anomalies(self, voltage, current, power, irradiance, temperature):
        """Detect anomalies using physics-informed rules"""
        theoretical_voltage = self.calculate_theoretical_voltage(irradiance, temperature)
        theoretical_current = self.calculate_theoretical_current(irradiance, temperature)
        theoretical_power = self.calculate_theoretical_power(irradiance, temperature)
        
        # Calculate deviations from expected values
        voltage_deviation = abs(voltage - theoretical_voltage) / theoretical_voltage if theoretical_voltage > 0 else 0
        current_deviation = abs(current - theoretical_current) / theoretical_current if theoretical_current > 0 else 0
        power_deviation = abs(power - theoretical_power) / theoretical_power if theoretical_power > 0 else 0
        
        # Define thresholds for fault detection
        voltage_threshold = 0.15  # 15% deviation
        current_threshold = 0.20  # 20% deviation
        power_threshold = 0.25    # 25% deviation
        
        anomalies = {
            'voltage_fault': voltage_deviation > voltage_threshold,
            'current_fault': current_deviation > current_threshold,
            'power_fault': power_deviation > power_threshold,
            'efficiency_low': self.calculate_efficiency(power, irradiance, temperature) < 80,
            'voltage_deviation': voltage_deviation,
            'current_deviation': current_deviation,
            'power_deviation': power_deviation
        }
        
        return anomalies
    
    def estimate_shading_loss(self, irradiance, expected_irradiance):
        """Estimate shading loss based on irradiance mismatch"""
        if expected_irradiance > 0:
            shading_ratio = (expected_irradiance - irradiance) / expected_irradiance
            return max(0, min(1, shading_ratio))
        return 0
    
    def calculate_mpp_tracking_efficiency(self, voltage, current, irradiance, temperature):
        """Calculate how well the system is tracking the maximum power point"""
        theoretical_power = self.calculate_theoretical_power(irradiance, temperature)
        actual_power = voltage * current
        
        if theoretical_power > 0:
            return (actual_power / theoretical_power) * 100
        return 0