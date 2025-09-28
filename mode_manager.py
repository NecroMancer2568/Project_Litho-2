# mode_manager.py - Helper for managing prediction modes and parameters

import json
import os
from typing import Dict, Any, Optional


class ModeManager:
    """Manages prediction modes and simulation parameters."""

    def __init__(self, config_file='mode_config.json'):
        self.config_file = config_file
        self.default_config = {
            'mode': 'SIMULATE',
            'simulate_params': {
                'rainfall': 15.0,
                'temperature': 18.0,
                'vibration': 1.5,
                'pore_pressure': 85.0,
                'displacement': 0.35
            }
        }
        self.load_config()

    def load_config(self):
        """Load configuration from file or create default."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except (json.JSONDecodeError, KeyError):
                self.config = self.default_config.copy()
                self.save_config()
        else:
            self.config = self.default_config.copy()
            self.save_config()

    def save_config(self):
        """Save current configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get_mode(self) -> str:
        """Get current prediction mode."""
        return self.config.get('mode', 'SIMULATE')

    def set_mode(self, mode: str):
        """Set prediction mode."""
        if mode in ['LIVE', 'SIMULATE']:
            self.config['mode'] = mode
            self.save_config()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'LIVE' or 'SIMULATE'")

    def get_simulate_params(self) -> Dict[str, float]:
        """Get current simulation parameters."""
        return self.config.get('simulate_params', self.default_config['simulate_params'])

    def set_simulate_params(self, params: Dict[str, float]):
        """Set simulation parameters."""
        # Validate parameters
        valid_params = ['rainfall', 'temperature', 'vibration', 'pore_pressure', 'displacement']
        for key in params:
            if key not in valid_params:
                raise ValueError(f"Invalid parameter: {key}")

        # Update configuration
        if 'simulate_params' not in self.config:
            self.config['simulate_params'] = {}

        self.config['simulate_params'].update(params)
        self.save_config()

    def get_preset(self, preset_name: str) -> Dict[str, float]:
        """Get predefined parameter presets."""
        presets = {
            'low_risk': {
                'rainfall': 5.0,
                'temperature': 25.0,
                'vibration': 0.2,
                'pore_pressure': 55.0,
                'displacement': 0.05
            },
            'medium_risk': {
                'rainfall': 25.0,
                'temperature': 15.0,
                'vibration': 1.0,
                'pore_pressure': 100.0,
                'displacement': 0.5
            },
            'high_risk': {
                'rainfall': 50.0,
                'temperature': 5.0,
                'vibration': 3.0,
                'pore_pressure': 150.0,
                'displacement': 1.5
            },
            'extreme_risk': {
                'rainfall': 80.0,
                'temperature': 2.0,
                'vibration': 4.5,
                'pore_pressure': 200.0,
                'displacement': 2.0
            }
        }
        return presets.get(preset_name, {})

    def apply_preset(self, preset_name: str):
        """Apply a predefined parameter preset."""
        preset = self.get_preset(preset_name)
        if preset:
            self.set_simulate_params(preset)
        else:
            raise ValueError(f"Unknown preset: {preset_name}")

    def validate_params(self, params: Dict[str, float]) -> list:
        """Validate simulation parameters and return any errors."""
        validations = {
            'rainfall': (0, 200, "mm"),
            'temperature': (-20, 60, "°C"),
            'vibration': (0, 10, "magnitude"),
            'pore_pressure': (40, 300, "kPa"),
            'displacement': (0, 5, "mm/day")
        }

        errors = []
        for param, value in params.items():
            if param in validations:
                min_val, max_val, unit = validations[param]
                if not (min_val <= value <= max_val):
                    errors.append(f"{param}: {value} {unit} is outside valid range [{min_val}-{max_val}]")

        return errors

    def get_config_summary(self) -> str:
        """Get a human-readable summary of current configuration."""
        mode = self.get_mode()
        summary = f"Mode: {mode}\n"

        if mode == 'SIMULATE':
            params = self.get_simulate_params()
            summary += "Simulation Parameters:\n"
            summary += f"  • Rainfall (72hr): {params.get('rainfall', 'N/A')} mm\n"
            summary += f"  • Temperature: {params.get('temperature', 'N/A')} °C\n"
            summary += f"  • Peak Vibration: {params.get('vibration', 'N/A')} magnitude\n"
            summary += f"  • Pore Pressure: {params.get('pore_pressure', 'N/A')} kPa\n"
            summary += f"  • Displacement: {params.get('displacement', 'N/A')} mm/day\n"

        return summary


# Global instance for easy access
mode_manager = ModeManager()


# Convenience functions
def get_current_mode():
    return mode_manager.get_mode()


def set_current_mode(mode):
    mode_manager.set_mode(mode)


def get_simulation_parameters():
    return mode_manager.get_simulate_params()


def set_simulation_parameters(**kwargs):
    mode_manager.set_simulate_params(kwargs)


def apply_risk_preset(preset_name):
    mode_manager.apply_preset(preset_name)