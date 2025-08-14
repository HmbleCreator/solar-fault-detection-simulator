"""
Solar Panel Data Simulator
Generates realistic solar panel performance data with fault scenarios
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class SolarDataSimulator:
    """Simulates solar panel performance data with realistic patterns and faults"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Solar panel parameters
        self.rated_power = 320  # W
        self.rated_voltage = 37.0  # V
        self.rated_current = 8.6  # A
        
        # Environmental parameters
        self.base_temp = 25  # °C
        self.temp_range = 15  # °C variation
        self.irradiance_base = 1000  # W/m²
        
    def generate_daily_profile(self, days=7, samples_per_hour=4):
        """Generate daily solar irradiance and temperature profiles"""
        total_samples = days * 24 * samples_per_hour
        
        # Time array
        time = pd.date_range(start='2024-01-01', periods=total_samples, 
                           freq=f'{60//samples_per_hour}min')
        
        # Solar irradiance profile (bell curve during daylight hours)
        irradiance = np.zeros(total_samples)
        temperature = np.zeros(total_samples)
        
        for i, t in enumerate(time):
            hour = t.hour + t.minute / 60
            
            # Daylight hours (6 AM to 6 PM)
            if 6 <= hour <= 18:
                # Bell curve for irradiance
                peak_hour = 12
                std_dev = 3
                irradiance[i] = self.irradiance_base * np.exp(-0.5 * ((hour - peak_hour) / std_dev) ** 2)
                
                # Temperature follows irradiance with some lag
                base_temp = 25 + (irradiance[i] / self.irradiance_base) * 10
                temperature[i] = base_temp + np.random.normal(0, 2)
            else:
                irradiance[i] = 0
                temperature[i] = 20 + np.random.normal(0, 1)
        
        return time, irradiance, temperature
    
    def calculate_theoretical_output(self, irradiance, temperature, panel_age=0):
        """Calculate theoretical output based on environmental conditions and panel age"""
        # Temperature coefficients
        temp_coeff_voltage = -0.0035
        temp_coeff_current = 0.0006
        temp_coeff_power = -0.004
        
        # Age-based degradation (0.5% per year typical)
        age_degradation = 1 - (panel_age * 0.005)
        
        # Adjust for temperature and irradiance
        temp_factor = 1 + temp_coeff_power * (temperature - 25)
        irradiance_factor = irradiance / 1000
        
        # Ensure realistic bounds using numpy vectorized operations
        temp_factor = np.clip(temp_factor, 0.5, 1.2)
        
        theoretical_power = self.rated_power * temp_factor * irradiance_factor * age_degradation
        theoretical_voltage = self.rated_voltage * (1 + temp_coeff_voltage * (temperature - 25)) * np.sqrt(age_degradation)
        theoretical_current = self.rated_current * temp_coeff_current * (temperature - 25) * irradiance_factor * age_degradation
        
        # Ensure positive values
        theoretical_power = np.maximum(0, theoretical_power)
        theoretical_voltage = np.maximum(0, theoretical_voltage)
        theoretical_current = np.maximum(0, theoretical_current)
        
        return theoretical_power, theoretical_voltage, theoretical_current
    
    def add_faults(self, power, voltage, current, fault_type='none', severity=0.5):
        """Add realistic fault patterns to the data"""
        
        if fault_type == 'none':
            return power, voltage, current, 'Normal'
        
        elif fault_type == 'shading':
            # Partial shading reduces irradiance and creates hot spots
            shading_factor = 1 - severity * 0.3  # 0-30% reduction
            power *= shading_factor
            voltage *= (0.8 + 0.2 * (1 - severity))  # Voltage drop
            current *= shading_factor
            return power, voltage, current, f'Shading ({severity*100:.0f}%)'
        
        elif fault_type == 'degradation':
            # Long-term degradation reduces efficiency
            degradation_factor = 1 - severity * 0.2  # 0-20% reduction
            power *= degradation_factor
            voltage *= 0.95  # Slight voltage reduction
            current *= degradation_factor
            return power, voltage, current, f'Degradation ({severity*100:.0f}%)'
        
        elif fault_type == 'soiling':
            # Dust accumulation reduces light absorption
            soiling_factor = 1 - severity * 0.25  # 0-25% reduction
            power *= soiling_factor
            voltage *= 0.9  # Voltage drop
            current *= soiling_factor
            return power, voltage, current, f'Soiling ({severity*100:.0f}%)'
        
        elif fault_type == 'hotspot':
            # Hotspot heating creates localized damage
            hotspot_factor = 1 - severity * 0.4  # 0-40% reduction
            power *= hotspot_factor
            voltage *= (0.7 + 0.3 * (1 - severity))  # Significant voltage drop
            current *= 0.8  # Current reduction
            return power, voltage, current, f'Hotspot ({severity*100:.0f}%)'
        
        elif fault_type == 'bypass_diode_failure':
            # Bypass diode failure causes string mismatch
            bypass_factor = 1 - severity * 0.35  # 0-35% reduction
            power *= bypass_factor
            voltage *= 0.6  # Major voltage drop
            current *= (0.8 + 0.2 * (1 - severity))
            return power, voltage, current, f'Bypass Diode ({severity*100:.0f}%)'
        
        else:
            return power, voltage, current, 'Normal'
    
    def add_noise(self, data, noise_level=0.02):
        """Add measurement noise to simulate real-world conditions"""
        return data * (1 + np.random.normal(0, noise_level, len(data)))
    
    def generate_dataset(self, days=30, fault_probability=0.15):
        """Generate complete dataset with various fault scenarios"""
        
        # Generate base environmental data
        time, irradiance, temperature = self.generate_daily_profile(days)
        
        # Calculate theoretical outputs
        theoretical_power, theoretical_voltage, theoretical_current = self.calculate_theoretical_output(
            irradiance, temperature)
        
        # Initialize arrays for actual measurements
        actual_power = theoretical_power.copy()
        actual_voltage = theoretical_voltage.copy()
        actual_current = theoretical_current.copy()
        fault_flags = ['Normal'] * len(time)
        
        # Define fault types and their probabilities
        fault_types = ['shading', 'degradation', 'soiling', 'hotspot', 'bypass_diode_failure']
        
        # Apply faults randomly
        for i in range(len(time)):
            if irradiance[i] > 100:  # Only apply faults during daylight
                if np.random.random() < fault_probability:
                    fault_type = np.random.choice(fault_types)
                    severity = np.random.uniform(0.3, 0.8)
                    
                    actual_power[i], actual_voltage[i], actual_current[i], fault_flags[i] = self.add_faults(
                        actual_power[i], actual_voltage[i], actual_current[i], fault_type, severity)
        
        # Add measurement noise
        actual_power = self.add_noise(actual_power)
        actual_voltage = self.add_noise(actual_voltage)
        actual_current = self.add_noise(actual_current)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': time,
            'irradiance': irradiance,
            'temperature': temperature,
            'voltage': actual_voltage,
            'current': actual_current,
            'power': actual_power,
            'theoretical_power': theoretical_power,
            'fault_type': fault_flags
        })
        
        return data
    
    def save_sample_data(self, filename='sample_solar_data.csv', days=30):
        """Generate and save sample data to CSV file"""
        data = self.generate_dataset(days)
        data.to_csv(filename, index=False)
        print(f"Generated {len(data)} samples and saved to {filename}")
        return data

if __name__ == "__main__":
    simulator = SolarDataSimulator()
    data = simulator.save_sample_data('../data/sample_solar_data.csv')
    print(f"Dataset shape: {data.shape}")
    print(f"Fault distribution:\n{data['fault_type'].value_counts()}")