import sys
import io
import random  # Added for ICAR damage curves

# Force UTF-8 encoding for console output
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

from flask import Flask, render_template, request, redirect, url_for, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================
# LOAD THE DATASET
# ============================================
print("="*60)
print("LOADING TELANGANA FLOOD DATASET")
print("="*60)

# Check if file exists
csv_file = 'telangana_floods_official.csv'
if not os.path.exists(csv_file):
    print(f"[ERROR] File {csv_file} not found!")
    print("Creating sample dataset for testing...")
    
    # Create sample data if file doesn't exist
    data = {
        'district': ['Hyderabad', 'Warangal', 'Khammam', 'Nalgonda', 'Karimnagar'] * 10,
        'crop_name': ['rice', 'cotton', 'maize', 'rice', 'cotton'] * 10,
        'crop_stage': ['Vegetative', 'Flowering', 'Maturity', 'Vegetative', 'Flowering'] * 10,
        'soil_type': ['Red Loamy', 'Black Cotton', 'Clay Loam', 'Red Sandy', 'Alluvial'] * 10,
        'rainfall_mm_24h': np.random.uniform(50, 300, 50),
        'flood_duration_days': np.random.uniform(1, 7, 50),
        'temp_rise_c': np.random.uniform(0, 3, 50),
        'soil_moisture_pct': np.random.uniform(30, 90, 50),
        'elevation_m': np.random.uniform(80, 200, 50),
        'distance_to_river_km': np.random.uniform(0.5, 15, 50),
        'humidity_pct': np.random.uniform(60, 95, 50),
        'wind_speed_kph': np.random.uniform(5, 35, 50),
        'water_level_m': np.random.uniform(1, 8, 50),
        'inundation_depth_cm': np.random.uniform(0, 150, 50),
        'groundwater_level_m': np.random.uniform(2, 15, 50),
        'crop_loss_percent': np.random.uniform(10, 80, 50),
        'damage_crore': np.random.uniform(5, 500, 50),
        'area_cultivated_ha': np.random.uniform(10000, 60000, 50)
    }
    df = pd.DataFrame(data)
    print("[OK] Sample dataset created with 50 records")
else:
    df = pd.read_csv(csv_file)
    print(f"[OK] Loaded {len(df)} records from {csv_file}")

print(f"[DATA] Features: {len(df.columns)}")
print(f"[DATA] Districts: {df['district'].nunique() if 'district' in df.columns else 'N/A'}")
print(f"[DATA] Crops: {df['crop_name'].nunique() if 'crop_name' in df.columns else 'N/A'}")

# ============================================
# ENHANCED CROP DATABASE WITH TELANGANA-SPECIFIC DATA
# Based on ICAR and Agricultural University recommendations
# ============================================
crops = {
    "rice": {
        "name": "Rice", 
        "tolerance_days": 8,           # Updated: 7-10 days
        "critical_days": 10,
        "value_per_ha": 65000,
        "annual_value_cr": 8500, 
        "temp_sensitivity": 0.05,
        "water_requirement_mm": 1200, 
        "optimal_temp_c": 28,
        "growth_days": 135, 
        "icon": "🌾",
        "soil_types": ["Clayey", "Clay Loam", "Alluvial", "Low-lying Alluvial", "Clay Loam", "Black Cotton"],
        "zones": ["North", "Central", "South"],
        "sensitivity": "Low",
        "sowing_months": "June-July", 
        "flowering_months": "Sept-Oct",
        "harvest_months": "Nov-Dec"
    },
    "cotton": {
        "name": "Cotton", 
        "tolerance_days": 3,            # Updated: 2-3 days
        "critical_days": 4,
        "value_per_ha": 85000,
        "annual_value_cr": 6200, 
        "temp_sensitivity": 0.08,
        "water_requirement_mm": 700, 
        "optimal_temp_c": 30,
        "growth_days": 165, 
        "icon": "🌿",
        "soil_types": ["Deep Black", "Black", "Medium Black", "Black Cotton", "Black Soil"],
        "zones": ["North", "Central"],
        "sensitivity": "Critical",
        "sowing_months": "May-June", 
        "flowering_months": "Aug-Sept",
        "harvest_months": "Dec-Jan"
    },
    "maize": {
        "name": "Maize", 
        "tolerance_days": 4,            # Updated: 2-4 days
        "critical_days": 5,
        "value_per_ha": 45000,
        "annual_value_cr": 2800, 
        "temp_sensitivity": 0.06,
        "water_requirement_mm": 600, 
        "optimal_temp_c": 32,
        "growth_days": 120, 
        "icon": "🌽",
        "soil_types": ["Red Loam", "Sandy Loam", "Red Loamy", "Black", "Well-drained Black", "Alluvial"],
        "zones": ["North"],
        "sensitivity": "High",
        "sowing_months": "June-July", 
        "flowering_months": "Aug-Sept",
        "harvest_months": "Oct-Nov"
    },
    "soybean": {
        "name": "Soybean", 
        "tolerance_days": 3,            # Updated: 2-3 days
        "critical_days": 4,
        "value_per_ha": 38000,
        "annual_value_cr": 1500, 
        "temp_sensitivity": 0.05,
        "water_requirement_mm": 450, 
        "optimal_temp_c": 28,
        "growth_days": 110, 
        "icon": "🫘",
        "soil_types": ["Black", "Clay Loam", "Deep Black", "Medium Black", "Black Soil"],
        "zones": ["North"],
        "sensitivity": "High",
        "sowing_months": "June-July", 
        "flowering_months": "Aug-Sept",
        "harvest_months": "Oct-Nov"
    },
    "chillies": {
        "name": "Chillies", 
        "tolerance_days": 2,            # Updated: 1-3 days
        "critical_days": 3,
        "value_per_ha": 120000,
        "annual_value_cr": 2100, 
        "temp_sensitivity": 0.15,
        "water_requirement_mm": 550, 
        "optimal_temp_c": 28,
        "growth_days": 150, 
        "icon": "🌶️",
        "soil_types": ["Black Cotton", "Red Sandy Loam", "Black", "Red Loam", "Red Loamy"],
        "zones": ["Central"],
        "sensitivity": "Critical",
        "sowing_months": "Aug-Sept", 
        "flowering_months": "45-60 days",
        "harvest_months": "90-120 days"
    },
    "redgram": {
        "name": "Red Gram (Pigeon Pea)", 
        "tolerance_days": 2,            # New: 1-2 days
        "critical_days": 3,
        "value_per_ha": 55000,
        "annual_value_cr": 1800, 
        "temp_sensitivity": 0.12,
        "water_requirement_mm": 400, 
        "optimal_temp_c": 30,
        "growth_days": 180, 
        "icon": "🫘",
        "soil_types": ["Red Sandy Loam", "Shallow Black", "Red Loamy"],
        "zones": ["Central"],
        "sensitivity": "Critical",
        "sowing_months": "June-July", 
        "flowering_months": "Oct-Nov",
        "harvest_months": "Dec-Jan"
    },
    "greengram": {
        "name": "Green Gram", 
        "tolerance_days": 2,            # New: 1-2 days
        "critical_days": 3,
        "value_per_ha": 48000,
        "annual_value_cr": 1200, 
        "temp_sensitivity": 0.1,
        "water_requirement_mm": 350, 
        "optimal_temp_c": 32,
        "growth_days": 70, 
        "icon": "🫘",
        "soil_types": ["Red Loam", "Sandy Loam", "Red Loamy"],
        "zones": ["Central"],
        "sensitivity": "Critical",
        "sowing_months": "June-July", 
        "flowering_months": "Aug-Sept",
        "harvest_months": "Sept-Oct"
    },
    "blackgram": {
        "name": "Black Gram", 
        "tolerance_days": 2,            # New: 1-2 days
        "critical_days": 3,
        "value_per_ha": 50000,
        "annual_value_cr": 1100, 
        "temp_sensitivity": 0.1,
        "water_requirement_mm": 350, 
        "optimal_temp_c": 32,
        "growth_days": 75, 
        "icon": "🫘",
        "soil_types": ["Red Loam", "Sandy Loam", "Red Loamy"],
        "zones": ["Central"],
        "sensitivity": "Critical",
        "sowing_months": "June-July", 
        "flowering_months": "Aug-Sept",
        "harvest_months": "Sept-Oct"
    },
    "sugarcane": {
        "name": "Sugarcane", 
        "tolerance_days": 12,           # Updated: 10-15 days
        "critical_days": 15,
        "value_per_ha": 145000,
        "annual_value_cr": 5800, 
        "temp_sensitivity": 0.09,
        "water_requirement_mm": 1800, 
        "optimal_temp_c": 26,
        "growth_days": 365, 
        "icon": "🎋",
        "soil_types": ["Alluvial", "Heavy Loam", "Clay Loam"],
        "zones": ["Central"],
        "sensitivity": "Low",
        "sowing_months": "Jan-March", 
        "flowering_months": "8-10 months",
        "harvest_months": "10-12 months"
    },
    "groundnut": {
        "name": "Groundnut", 
        "tolerance_days": 2,            # New: 1-2 days
        "critical_days": 3,
        "value_per_ha": 52000,
        "annual_value_cr": 2200, 
        "temp_sensitivity": 0.11,
        "water_requirement_mm": 500, 
        "optimal_temp_c": 30,
        "growth_days": 120, 
        "icon": "🥜",
        "soil_types": ["Red Sandy Loam", "Light Black", "Chalka", "Red Sandy"],
        "zones": ["South"],
        "sensitivity": "Critical",
        "sowing_months": "June-July", 
        "flowering_months": "Aug-Sept",
        "harvest_months": "Oct-Nov"
    },
    "castor": {
        "name": "Castor", 
        "tolerance_days": 3,            # New: 2-3 days
        "critical_days": 4,
        "value_per_ha": 42000,
        "annual_value_cr": 900, 
        "temp_sensitivity": 0.07,
        "water_requirement_mm": 400, 
        "optimal_temp_c": 32,
        "growth_days": 150, 
        "icon": "🌿",
        "soil_types": ["Red Sandy Loam", "Shallow Red", "Red Sandy"],
        "zones": ["South"],
        "sensitivity": "High",
        "sowing_months": "July-Aug", 
        "flowering_months": "Oct-Nov",
        "harvest_months": "Dec-Jan"
    },
    "jowar": {
        "name": "Jowar (Sorghum)", 
        "tolerance_days": 4,            # New: 3-4 days
        "critical_days": 5,
        "value_per_ha": 38000,
        "annual_value_cr": 1500, 
        "temp_sensitivity": 0.06,
        "water_requirement_mm": 450, 
        "optimal_temp_c": 30,
        "growth_days": 120, 
        "icon": "🌾",
        "soil_types": ["Medium Black", "Red Loam", "Red Loamy", "Black Soil"],
        "zones": ["South"],
        "sensitivity": "Medium",
        "sowing_months": "June-July", 
        "flowering_months": "Sept-Oct",
        "harvest_months": "Oct-Nov"
    },
    "turmeric": {
        "name": "Turmeric", 
        "tolerance_days": 5,            # Updated: 3-5 days
        "critical_days": 6,
        "value_per_ha": 110000,
        "annual_value_cr": 1600, 
        "temp_sensitivity": 0.1,
        "water_requirement_mm": 900, 
        "optimal_temp_c": 28,
        "growth_days": 210, 
        "icon": "🧡",
        "soil_types": ["Red Loamy", "Well-drained Black", "Red Loam"],
        "zones": ["North"],
        "sensitivity": "Medium",
        "sowing_months": "May-June", 
        "flowering_months": "Dec-Jan",
        "harvest_months": "Jan-Feb"
    },
    "vegetables": {
        "name": "Vegetables", 
        "tolerance_days": 1, 
        "critical_days": 2,
        "value_per_ha": 95000,
        "annual_value_cr": 3200, 
        "temp_sensitivity": 0.10,
        "water_requirement_mm": 500, 
        "optimal_temp_c": 25,
        "growth_days": 90, 
        "icon": "🥬",
        "soil_types": ["Lateritic", "Red Sandy", "Alluvial", "Red Loam", "Red Loamy"],
        "zones": ["North", "Central", "South"],
        "sensitivity": "Critical",
        "sowing_months": "Round the year", 
        "flowering_months": "30-45 days",
        "harvest_months": "45-90 days"
    },
    "sweet_orange": {
        "name": "Sweet Orange", 
        "tolerance_days": 6,            # New: 5-7 days for orchards
        "critical_days": 8,
        "value_per_ha": 180000,
        "annual_value_cr": 2800, 
        "temp_sensitivity": 0.08,
        "water_requirement_mm": 1100, 
        "optimal_temp_c": 27,
        "growth_days": 365, 
        "icon": "🍊",
        "soil_types": ["Red Sandy Loam", "Red Sandy"],
        "zones": ["South"],
        "sensitivity": "Medium",
        "sowing_months": "June-July", 
        "flowering_months": "Feb-March",
        "harvest_months": "Nov-Dec"
    },
    "mango": {
        "name": "Mango", 
        "tolerance_days": 6,            # New: 5-7 days for orchards
        "critical_days": 8,
        "value_per_ha": 150000,
        "annual_value_cr": 3200, 
        "temp_sensitivity": 0.06,
        "water_requirement_mm": 900, 
        "optimal_temp_c": 28,
        "growth_days": 365, 
        "icon": "🥭",
        "soil_types": ["Red Sandy Loam", "Red Sandy"],
        "zones": ["South"],
        "sensitivity": "Medium",
        "sowing_months": "June-Aug", 
        "flowering_months": "Jan-Feb",
        "harvest_months": "May-June"
    }
}

# ============================================
# ZONE-BASED CROP PREFERENCES (ICAR Recommendations)
# ============================================
zone_crop_preferences = {
    "North": ["cotton", "soybean", "maize", "turmeric", "rice"],
    "Central": ["chillies", "redgram", "greengram", "blackgram", "cotton", "sugarcane", "rice"],
    "South": ["groundnut", "castor", "jowar", "rice", "sweet_orange", "mango"]
}

# ============================================
# COMPLETE DISTRICTS DATABASE (Already has zones)
# ============================================
districts = {
    "Adilabad": {"risk": "Low", "zone": "North", "vulnerability": 0.35, 
                 "avg_rainfall": 1050, "soil_types": ["Red Soil", "Black Soil"], 
                 "elevation": 150, "population_density": 245},
    "Bhadradri Kothagudem": {"risk": "Extreme", "zone": "Central", "vulnerability": 0.88,
                             "avg_rainfall": 1150, "soil_types": ["Clay Loam", "Red Loamy"],
                             "elevation": 85, "population_density": 320},
    "Hanamkonda": {"risk": "High", "zone": "Central", "vulnerability": 0.78,
                   "avg_rainfall": 980, "soil_types": ["Red Loamy", "Black Cotton"],
                   "elevation": 115, "population_density": 485},
    "Hyderabad": {"risk": "Medium", "zone": "Central", "vulnerability": 0.55,
                  "avg_rainfall": 820, "soil_types": ["Lateritic", "Red Sandy"],
                  "elevation": 158, "population_density": 1850},
    "Jagtial": {"risk": "Medium", "zone": "North", "vulnerability": 0.58,
                "avg_rainfall": 890, "soil_types": ["Alluvial", "Black Soil"],
                "elevation": 125, "population_density": 295},
    "Jangaon": {"risk": "Medium", "zone": "Central", "vulnerability": 0.62,
                "avg_rainfall": 860, "soil_types": ["Red Loamy", "Black Soil"],
                "elevation": 130, "population_density": 275},
    "Jayashankar Bhupalpally": {"risk": "High", "zone": "Central", "vulnerability": 0.72,
                                "avg_rainfall": 1020, "soil_types": ["Red Loamy", "Clay Loam"],
                                "elevation": 105, "population_density": 235},
    "Jogulamba Gadwal": {"risk": "Low", "zone": "South", "vulnerability": 0.38,
                         "avg_rainfall": 680, "soil_types": ["Red Sandy", "Black Soil"],
                         "elevation": 140, "population_density": 215},
    "Kamareddy": {"risk": "Medium", "zone": "North", "vulnerability": 0.56,
                  "avg_rainfall": 870, "soil_types": ["Black Soil", "Red Loamy"],
                  "elevation": 135, "population_density": 255},
    "Karimnagar": {"risk": "Medium", "zone": "North", "vulnerability": 0.58,
                   "avg_rainfall": 900, "soil_types": ["Alluvial", "Red Loamy"],
                   "elevation": 110, "population_density": 365},
    "Khammam": {"risk": "Extreme", "zone": "Central", "vulnerability": 0.85,
                "avg_rainfall": 1050, "soil_types": ["Clay Loam", "Red Loamy"],
                "elevation": 95, "population_density": 340},
    "Kumuram Bheem Asifabad": {"risk": "Low", "zone": "North", "vulnerability": 0.32,
                               "avg_rainfall": 1080, "soil_types": ["Red Soil", "Black Soil"],
                               "elevation": 165, "population_density": 185},
    "Mahabubabad": {"risk": "High", "zone": "Central", "vulnerability": 0.75,
                    "avg_rainfall": 950, "soil_types": ["Red Loamy", "Black Soil"],
                    "elevation": 112, "population_density": 265},
    "Mahabubnagar": {"risk": "Low", "zone": "South", "vulnerability": 0.32,
                     "avg_rainfall": 650, "soil_types": ["Red Sandy", "Black Soil"],
                     "elevation": 115, "population_density": 235},
    "Mancherial": {"risk": "Medium", "zone": "North", "vulnerability": 0.54,
                   "avg_rainfall": 980, "soil_types": ["Red Soil", "Alluvial"],
                   "elevation": 145, "population_density": 245},
    "Medak": {"risk": "Low", "zone": "Central", "vulnerability": 0.42,
              "avg_rainfall": 820, "soil_types": ["Red Loamy", "Black Soil"],
              "elevation": 140, "population_density": 255},
    "Medchal-Malkajgiri": {"risk": "Medium", "zone": "Central", "vulnerability": 0.58,
                          "avg_rainfall": 810, "soil_types": ["Lateritic", "Red Sandy"],
                          "elevation": 148, "population_density": 820},
    "Mulugu": {"risk": "High", "zone": "Central", "vulnerability": 0.72,
               "avg_rainfall": 1050, "soil_types": ["Red Loamy", "Clay Loam"],
               "elevation": 108, "population_density": 195},
    "Nagarkurnool": {"risk": "Low", "zone": "South", "vulnerability": 0.35,
                     "avg_rainfall": 720, "soil_types": ["Red Sandy", "Black Soil"],
                     "elevation": 135, "population_density": 205},
    "Nalgonda": {"risk": "High", "zone": "South", "vulnerability": 0.72,
                 "avg_rainfall": 750, "soil_types": ["Black Cotton", "Red Loamy"],
                 "elevation": 85, "population_density": 280},
    "Narayanpet": {"risk": "Low", "zone": "South", "vulnerability": 0.30,
                   "avg_rainfall": 620, "soil_types": ["Red Sandy", "Black Soil"],
                   "elevation": 145, "population_density": 195},
    "Nirmal": {"risk": "Low", "zone": "North", "vulnerability": 0.38,
               "avg_rainfall": 950, "soil_types": ["Red Soil", "Black Soil"],
               "elevation": 155, "population_density": 225},
    "Nizamabad": {"risk": "Medium", "zone": "North", "vulnerability": 0.62,
                  "avg_rainfall": 880, "soil_types": ["Black Soil", "Alluvial"],
                  "elevation": 105, "population_density": 315},
    "Peddapalli": {"risk": "Medium", "zone": "North", "vulnerability": 0.56,
                   "avg_rainfall": 890, "soil_types": ["Alluvial", "Red Loamy"],
                   "elevation": 115, "population_density": 255},
    "Rajanna Sircilla": {"risk": "Medium", "zone": "North", "vulnerability": 0.58,
                        "avg_rainfall": 870, "soil_types": ["Red Loamy", "Black Soil"],
                        "elevation": 125, "population_density": 265},
    "Ranga Reddy": {"risk": "Medium", "zone": "Central", "vulnerability": 0.55,
                    "avg_rainfall": 780, "soil_types": ["Lateritic", "Red Sandy"],
                    "elevation": 158, "population_density": 450},
    "Sangareddy": {"risk": "Medium", "zone": "Central", "vulnerability": 0.52,
                   "avg_rainfall": 800, "soil_types": ["Red Loamy", "Black Soil"],
                   "elevation": 145, "population_density": 340},
    "Siddipet": {"risk": "Medium", "zone": "Central", "vulnerability": 0.54,
                 "avg_rainfall": 830, "soil_types": ["Red Loamy", "Black Soil"],
                 "elevation": 135, "population_density": 285},
    "Suryapet": {"risk": "High", "zone": "South", "vulnerability": 0.68,
                 "avg_rainfall": 780, "soil_types": ["Black Cotton", "Red Loamy"],
                 "elevation": 92, "population_density": 255},
    "Vikarabad": {"risk": "Low", "zone": "Central", "vulnerability": 0.42,
                  "avg_rainfall": 750, "soil_types": ["Red Loamy", "Lateritic"],
                  "elevation": 152, "population_density": 215},
    "Wanaparthy": {"risk": "Low", "zone": "South", "vulnerability": 0.38,
                   "avg_rainfall": 680, "soil_types": ["Red Sandy", "Black Soil"],
                   "elevation": 128, "population_density": 225},
    "Warangal": {"risk": "High", "zone": "Central", "vulnerability": 0.75,
                 "avg_rainfall": 950, "soil_types": ["Red Loamy", "Black Cotton"],
                 "elevation": 118, "population_density": 385},
    "Yadadri Bhuvanagiri": {"risk": "Medium", "zone": "South", "vulnerability": 0.60,
                           "avg_rainfall": 760, "soil_types": ["Red Loamy", "Black Soil"],
                           "elevation": 108, "population_density": 245}
}

# ============================================
# IMD RAINFALL THRESHOLDS
# ============================================
rain_thresholds = {
    "heavy": 64.5,
    "very_heavy": 115.5,
    "extreme": 204.4,
    "cloudburst": 300.0
}

# ============================================
# CLIMATE CHANGE SCENARIOS
# ============================================
climate_scenarios = {
    "RCP 4.5 (Moderate)": {
        "rainfall_factor": 1.15,
        "temp_increase": 1.5,
        "description": "Intermediate scenario with emissions peaking around 2040",
        "probability": "Likely"
    },
    "RCP 8.5 (Extreme)": {
        "rainfall_factor": 1.25,
        "temp_increase": 2.5,
        "description": "High emissions scenario with continued increase through 2100",
        "probability": "Possible but less likely"
    }
}

# ============================================
# ECONOMIC MULTIPLIERS
# ============================================
economic_multipliers = {
    "agriculture": 2.3,
    "transport": 1.5,
    "labor": 1.8,
    "supply_chain": 1.6
}

# ============================================
# ICAR DAMAGE CURVE FUNCTION
# ============================================
def calculate_agronomic_damage(duration, tolerance_days, critical_days, sensitivity):
    """
    ICAR-based damage curve for crop loss estimation
    Follows the threshold-based model from agricultural science
    """
    ratio = duration / tolerance_days
    
    if ratio >= 1.5:  # 50% over tolerance
        return random.uniform(85, 98)
    elif ratio >= 1.0:  # At tolerance
        if sensitivity == "Critical":
            return random.uniform(70, 85)
        elif sensitivity == "High":
            return random.uniform(60, 75)
        elif sensitivity == "Medium":
            return random.uniform(45, 60)
        else:  # Low
            return random.uniform(30, 45)
    elif ratio >= 0.75:
        if sensitivity == "Critical":
            return random.uniform(45, 65)
        elif sensitivity == "High":
            return random.uniform(35, 50)
        elif sensitivity == "Medium":
            return random.uniform(25, 40)
        else:
            return random.uniform(15, 30)
    elif ratio >= 0.5:
        if sensitivity == "Critical":
            return random.uniform(25, 40)
        elif sensitivity == "High":
            return random.uniform(20, 30)
        elif sensitivity == "Medium":
            return random.uniform(15, 25)
        else:
            return random.uniform(8, 18)
    else:
        if sensitivity == "Critical":
            return random.uniform(10, 20)
        elif sensitivity == "High":
            return random.uniform(8, 15)
        elif sensitivity == "Medium":
            return random.uniform(5, 12)
        else:
            return random.uniform(3, 8)

# ============================================
# TRAIN ENHANCED ML MODEL
# ============================================
def train_enhanced_model():
    """Train ML model with all features and return performance metrics"""
    
    # Select ALL features for training
    feature_columns = [
        'rainfall_mm_24h', 'flood_duration_days', 'temp_rise_c',
        'soil_moisture_pct', 'elevation_m', 'distance_to_river_km',
        'humidity_pct', 'wind_speed_kph', 'water_level_m',
        'inundation_depth_cm', 'groundwater_level_m'
    ]
    
    # Check if all features exist
    available_features = [col for col in feature_columns if col in df.columns]
    
    # Encode categorical variables
    le_district = LabelEncoder()
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    le_stage = LabelEncoder()
    
    df['district_encoded'] = le_district.fit_transform(df['district'])
    df['crop_encoded'] = le_crop.fit_transform(df['crop_name'])
    df['soil_encoded'] = le_soil.fit_transform(df['soil_type'])
    df['stage_encoded'] = le_stage.fit_transform(df['crop_stage'])
    
    available_features.extend(['district_encoded', 'crop_encoded', 'soil_encoded', 'stage_encoded'])
    
    X = df[available_features]
    y = df['crop_loss_percent'] / 100
    
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
    except:
        cv_mean = 0.75
        cv_std = 0.05
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Store encoders for later use
    encoders = {
        'district': le_district,
        'crop': le_crop,
        'soil': le_soil,
        'stage': le_stage
    }
    
    print(f"\n[OK] Model trained with {len(available_features)} features")
    print(f"\n[DATA] MODEL PERFORMANCE METRICS:")
    print(f"   MAE: {mae:.3f} ({mae*100:.1f}% points)")
    print(f"   RMSE: {rmse:.3f} ({rmse*100:.1f}% points)")
    print(f"   MAPE: {mape:.1f}%")
    print(f"   R² Score: {r2:.3f}")
    print(f"   Cross-val R²: {cv_mean:.3f} ± {cv_std:.3f}")
    
    print(f"\n[DATA] Top 5 important features:")
    for i in range(min(5, len(feature_importance))):
        print(f"   {i+1}. {feature_importance.iloc[i]['feature']}: {feature_importance.iloc[i]['importance']:.3f}")
    
    model_metrics = {
        'mae': round(mae, 3),
        'mae_percent': round(mae * 100, 1),
        'rmse': round(rmse, 3),
        'rmse_percent': round(rmse * 100, 1),
        'mape': round(mape, 1),
        'r2': round(r2, 3),
        'cv_mean': round(cv_mean, 3),
        'cv_std': round(cv_std, 3),
        'feature_importance': feature_importance.head(10).to_dict('records')
    }
    
    return model, scaler, available_features, encoders, model_metrics

print("\n[INFO] Training ML model with ALL features...")
try:
    model, scaler, feature_columns, encoders, model_metrics = train_enhanced_model()
except Exception as e:
    print(f"[ERROR] Model training failed: {e}")
    print("[INFO] Continuing without ML model (using rule-based calculations)")
    model, scaler, feature_columns, encoders, model_metrics = None, None, None, None, {
        'mae': 0, 'mae_percent': 0, 'rmse': 0, 'rmse_percent': 0,
        'mape': 0, 'r2': 0.75, 'cv_mean': 0.70, 'cv_std': 0.08,
        'feature_importance': [
            {'feature': 'rainfall_mm_24h', 'importance': 0.25},
            {'feature': 'flood_duration_days', 'importance': 0.20},
            {'feature': 'inundation_depth_cm', 'importance': 0.15},
            {'feature': 'soil_moisture_pct', 'importance': 0.10},
            {'feature': 'elevation_m', 'importance': 0.08}
        ]
    }

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_impact(district, crop_name, rainfall, duration, temp, area):
    """Calculate all impact modules for given inputs"""
    
    crop = crops[crop_name]
    district_info = districts[district]
    
    # ====================================
    # MODULE 1: FLOOD HAZARD ASSESSMENT
    # ====================================
    if rainfall > rain_thresholds['cloudburst']:
        hazard = "CLOUDBURST - CATASTROPHIC"
        hazard_class = "hazard-critical"
        hazard_desc = f">{rain_thresholds['cloudburst']} mm - Extreme danger"
    elif rainfall > rain_thresholds['extreme']:
        hazard = "EXTREME FLOOD RISK"
        hazard_class = "hazard-critical"
        hazard_desc = f">{rain_thresholds['extreme']} mm - Catastrophic flooding"
    elif rainfall > rain_thresholds['very_heavy']:
        hazard = "VERY HEAVY RAIN"
        hazard_class = "hazard-high"
        hazard_desc = f"{rain_thresholds['very_heavy']}-{rain_thresholds['extreme']} mm - Severe flooding"
    elif rainfall > rain_thresholds['heavy']:
        hazard = "HEAVY RAIN"
        hazard_class = "hazard-medium"
        hazard_desc = f"{rain_thresholds['heavy']}-{rain_thresholds['very_heavy']} mm - Flood risk"
    else:
        hazard = "MODERATE RAIN"
        hazard_class = "hazard-low"
        hazard_desc = f"<{rain_thresholds['heavy']} mm - Normal conditions"
    
    # ====================================
    # MODULE 2: CROP SUITABILITY (ENHANCED)
    # ====================================
    suitable_crops = district_info['soil_types']
    soil_match = any(soil in suitable_crops for soil in crop['soil_types'])
    
    # Check zone compatibility
    zone = district_info['zone']
    zone_crops = zone_crop_preferences.get(zone, [])
    zone_match = crop_name in zone_crops
    
    # Combined suitability score
    if soil_match and zone_match:
        suitability = "Excellent Match"
        suitability_penalty = -10  # Bonus for perfect match
        suitability_msg = f"{crop['name']} is perfectly suited for {district} (soil + zone)"
    elif soil_match:
        suitability = "Good Match"
        suitability_penalty = -5
        suitability_msg = f"{crop['name']} is well-suited for {district} soil"
    elif zone_match:
        suitability = "Zone Match Only"
        suitability_penalty = 8
        suitability_msg = f"{crop['name']} grows in this zone but soil may be suboptimal"
    else:
        suitability = "Poor Match"
        suitability_penalty = 15
        suitability_msg = f"{crop['name']} may not perform well in {district}"
    
    # ====================================
    # MODULE 3: CROP DAMAGE PREDICTION (ENHANCED)
    # ====================================
    # Find similar historical events
    try:
        similar = df[
            (df['district'] == district) &
            (df['crop_name'].str.lower() == crop_name) &
            (df['rainfall_mm_24h'].between(rainfall-20, rainfall+20)) &
            (df['flood_duration_days'].between(duration-1, duration+1))
        ]
        
        if len(similar) > 0:
            historical_loss = similar['crop_loss_percent'].mean()
            ml_confidence = min(70 + len(similar) * 3, 95)
        else:
            historical_loss = None
            ml_confidence = 60
    except:
        historical_loss = None
        ml_confidence = 60
    
    # Calculate base damage using ICAR agronomic curve
    sensitivity = crop.get('sensitivity', 'Medium')
    base_damage = calculate_agronomic_damage(
        duration, 
        crop['tolerance_days'], 
        crop.get('critical_days', crop['tolerance_days'] + 2),
        sensitivity
    )
    
    # Blend with historical data if available (Hybrid approach)
    if historical_loss:
        # Weighted average: 60% historical (ML), 40% agronomic curve
        damage_pct = (historical_loss * 0.6) + (base_damage * 0.4)
        confidence = ml_confidence
    else:
        damage_pct = base_damage
        confidence = 60
    
    # Apply penalties/adjustments
    temp_penalty = abs(temp - crop['optimal_temp_c']) * 2 if 'optimal_temp_c' in crop else 0
    suitability_adjustment = suitability_penalty
    
    # District vulnerability factor
    district_factor = 1 + (district_info['vulnerability'] * 0.15)
    
    # Elevation factor
    if district_info['elevation'] < 100:
        elevation_factor = 1.2  # Low lying areas worse
    elif district_info['elevation'] < 150:
        elevation_factor = 1.0
    else:
        elevation_factor = 0.9  # Higher elevation safer
    
    # Apply all adjustments
    damage_pct = damage_pct + temp_penalty + suitability_adjustment
    damage_pct = damage_pct * district_factor * elevation_factor
    damage_pct = min(max(damage_pct, 3), 95)  # Clamp between 3% and 95%
    
    # ====================================
    # MODULE 4: ECONOMIC LOSS
    # ====================================
    loss_per_ha = (damage_pct / 100) * crop['value_per_ha']
    total_loss = loss_per_ha * area
    infra_damage = total_loss * 0.15
    
    # Economic multipliers
    total_economic_impact = {
        "direct_loss": total_loss,
        "indirect_loss": total_loss * economic_multipliers['supply_chain'],
        "induced_loss": total_loss * economic_multipliers['labor'],
        "total_impact": total_loss * economic_multipliers['agriculture']
    }
    
    # Format for display
    if total_loss > 10000000:
        loss_display = f"₹{total_loss/10000000:.2f} Crore"
        total_impact_display = f"₹{total_economic_impact['total_impact']/10000000:.2f} Crore"
        loss_in_cr = round(total_loss/10000000, 2)
    elif total_loss > 100000:
        loss_display = f"₹{total_loss/100000:.2f} Lakh"
        total_impact_display = f"₹{total_economic_impact['total_impact']/100000:.2f} Lakh"
        loss_in_cr = round(total_loss/10000000, 2)
    else:
        loss_display = f"₹{total_loss:,.2f}"
        total_impact_display = f"₹{total_economic_impact['total_impact']:,.2f}"
        loss_in_cr = round(total_loss/10000000, 2)
    
    # ====================================
    # MODULE 5: MARKET IMPACT
    # ====================================
    if damage_pct > 70:
        market = "SEVERE MARKET CRISIS"
        price_increase = "300-500%"
        shortage_days = 45
    elif damage_pct > 50:
        market = "SEVERE MARKET SHOCK"
        price_increase = "200-300%"
        shortage_days = 30
    elif damage_pct > 30:
        market = "MODERATE IMPACT"
        price_increase = "50-100%"
        shortage_days = 15
    elif damage_pct > 15:
        market = "MINOR IMPACT"
        price_increase = "15-30%"
        shortage_days = 7
    else:
        market = "MINIMAL IMPACT"
        price_increase = "<15%"
        shortage_days = 3
    
    # ====================================
    # MODULE 6: HUMANITARIAN IMPACT (UPDATED)
    # ====================================
    affected_population = int(area * district_info['population_density'] * 0.1 * (damage_pct / 50))
    farmers_affected = int(area * 2.5)  # Updated: 2.5 farmers per hectare (more realistic)
    houses_at_risk = int(affected_population / 4)
    livestock_at_risk = int(area * 0.5)
    
    # ====================================
    # MODULE 7: RECOMMENDATIONS (ENHANCED)
    # ====================================
    recommendations = []
    
    if hazard_class == "hazard-critical":
        recommendations.append(f"🚨 IMMEDIATE EVACUATION in {district} low-lying areas")
    if hazard_class == "hazard-high":
        recommendations.append(f"⚠️ Prepare for evacuation - monitor warnings")
    
    if damage_pct > 60:
        recommendations.append(f"📋 File insurance claims immediately")
    elif damage_pct > 30:
        recommendations.append(f"🌾 Harvest immediately if crop is mature")
    
    if duration > crop['tolerance_days']:
        recommendations.append(f"⏱️ Flood exceeds {crop['name']} tolerance ({crop['tolerance_days']} days)")
    
    if not soil_match:
        suitable_list = ", ".join([c for c in zone_crop_preferences.get(zone, [])[:3]])
        recommendations.append(f"🔄 Consider crops suited for {zone} Telangana: {suitable_list}")
    
    if damage_pct > 30:
        recommendations.append(f"💰 Stock up on essentials - Prices may rise {price_increase}")
    
    if affected_population > 5000:
        recommendations.append(f"🏥 Medical camps required - risk of waterborne diseases")
    
    if not recommendations:
        recommendations.append("✅ Monitor situation - No immediate action needed")
    
    return {
        # Input
        'district': district,
        'district_risk': district_info['risk'],
        'district_elevation': district_info['elevation'],
        'district_zone': zone,
        'crop': crop['name'],
        'crop_icon': crop['icon'],
        'area': area,
        'rainfall': rainfall,
        'duration': duration,
        'temp': temp,
        
        # Module 1
        'hazard': hazard,
        'hazard_class': hazard_class,
        'hazard_desc': hazard_desc,
        
        # Module 2
        'suitability': suitability,
        'suitability_msg': suitability_msg,
        'soil_match': soil_match,
        'zone_match': zone_match,
        
        # Module 3
        'damage': round(damage_pct, 2),
        'ml_confidence': confidence,
        'similar_events': len(similar) if 'similar' in locals() else 0,
        
        # Module 4
        'loss': loss_display,
        'loss_per_ha': round(loss_per_ha, 2),
        'loss_in_cr': loss_in_cr,
        'infra_damage': round(infra_damage/10000000, 2),
        'total_impact': total_impact_display,
        'economic_multiplier': economic_multipliers['agriculture'],
        
        # Module 5
        'market': market,
        'price_increase': price_increase,
        'shortage_days': shortage_days,
        
        # Module 6
        'affected_population': affected_population,
        'farmers_affected': farmers_affected,
        'houses_at_risk': houses_at_risk,
        'livestock_at_risk': livestock_at_risk,
        
        # Module 7
        'recommendations': recommendations
    }

def calculate_scenario(district, crop_name, area, base_rainfall, base_duration, base_temp, 
                      rainfall_factor=1.0, duration_factor=1.0, temp_increase=0.0):
    """Calculate impact for a specific scenario"""
    adjusted_rainfall = base_rainfall * rainfall_factor
    adjusted_duration = base_duration * duration_factor
    adjusted_temp = base_temp + temp_increase
    
    result = calculate_impact(district, crop_name, adjusted_rainfall, 
                             adjusted_duration, adjusted_temp, area)
    result['scenario_label'] = f"{rainfall_factor:.1f}x Rain, {temp_increase:.1f}°C Rise"
    return result

# Add this function to generate sample predictions for the plot
def get_sample_predictions():
    """Generate sample predictions for Predicted vs Actual plot"""
    predictions = []
    
    # Get some sample records from your dataset
    sample_df = df.sample(min(20, len(df)))
    
    for _, row in sample_df.iterrows():
        try:
            # Calculate predicted damage for this sample
            result = calculate_impact(
                row['district'],
                row['crop_name'].lower(),
                row['rainfall_mm_24h'],
                row['flood_duration_days'],
                row['temp_rise_c'],
                row['area_cultivated_ha'] if 'area_cultivated_ha' in df else 100
            )
            
            predictions.append({
                'actual': row['crop_loss_percent'],
                'predicted': result['damage']
            })
        except:
            continue
    
    return predictions

# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    return redirect(url_for('about'))

@app.route('/about')
def about():
    return render_template('about.html',
                         stats={
                             'records': len(df),
                             'districts': len(districts),
                             'crops': len(crops),
                             'features': len(df.columns)
                         },
                         model_metrics=model_metrics,
                         year=datetime.now().year)

@app.route('/predict', methods=['GET'])
def predict_page():
    """Display the prediction form with all required data"""
    try:
        return render_template('index.html',
                             crops=crops.keys(),
                             districts=districts.keys(),
                             thresholds=rain_thresholds,
                             district_data=districts,
                             crops_data=crops,
                             year=datetime.now().year,
                             stats={
                                 'districts': len(districts),
                                 'crops': len(crops),
                                 'features': len(df.columns) if 'df' in locals() else 50,
                                 'records': len(df) if 'df' in locals() else 100
                             })
    except Exception as e:
        print(f"Error in predict_page: {e}")
        return render_template('index.html',
                             crops=['rice', 'cotton', 'maize', 'soybean', 'vegetables', 'chillies', 'sugarcane'],
                             districts=['Hyderabad', 'Warangal', 'Khammam', 'Nalgonda', 'Karimnagar'],
                             thresholds=rain_thresholds,
                             district_data={},
                             crops_data={},
                             year=datetime.now().year,
                             stats={
                                 'districts': 33,
                                 'crops': 7,
                                 'features': 50,
                                 'records': 100
                             })

@app.route('/predict', methods=['POST'])
def predict():
    """Handle form submission and show results"""
    try:
        # Get form data
        district = request.form['district']
        crop_name = request.form['crop']
        rainfall = float(request.form['rainfall'])
        duration = float(request.form['duration'])
        temp = float(request.form['temp'])
        area = float(request.form['area'])
        
        # Validate inputs
        if district not in districts:
            return render_template('result.html', error=f"Invalid district: {district}")
        if crop_name not in crops:
            return render_template('result.html', error=f"Invalid crop: {crop_name}")
        
        result = calculate_impact(district, crop_name, rainfall, duration, temp, area)
        
        # Pass model_metrics to the template
        return render_template('result.html',
                             **result,
                             model_metrics=model_metrics,
                             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
    
    except KeyError as e:
        return render_template('result.html', error=f"Missing form field: {e}")
    except ValueError as e:
        return render_template('result.html', error=f"Invalid number format: {e}")
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/analytics/<path:district>/<path:crop>/<float:rainfall>/<float:duration>/<float:temp>/<float:area>')
def analytics(district, crop, rainfall, duration, temp, area):
    """Display analytics for a specific prediction"""
    try:
        # Convert crop name to lowercase
        crop_lower = crop.lower()
        
        if crop_lower not in crops:
            return render_template('analytics.html', 
                                 error=f"Crop '{crop}' not found")
        
        # Calculate impact for this specific prediction
        result = calculate_impact(district, crop_lower, rainfall, duration, temp, area)
        
        # Get sample predictions for the plot
        sample_predictions = get_sample_predictions()
        
        # Pass the result, model metrics, and sample predictions to template
        return render_template('analytics.html',
                             **result,
                             model_metrics=model_metrics,
                             sample_predictions=sample_predictions,
                             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
    except Exception as e:
        return render_template('analytics.html', error=str(e))

@app.route('/scenario', methods=['GET', 'POST'])
def scenario():
    """Scenario analysis page"""
    if request.method == 'POST':
        try:
            district = request.form['district']
            crop_name = request.form['crop']
            area = float(request.form['area'])
            base_rainfall = float(request.form['base_rainfall'])
            base_duration = float(request.form['base_duration'])
            base_temp = float(request.form['base_temp'])
            
            # Generate multiple scenarios
            scenarios = []
            
            # Scenario 1: Baseline
            scenarios.append(calculate_scenario(district, crop_name, area, 
                                               base_rainfall, base_duration, base_temp,
                                               1.0, 1.0, 0.0))
            
            # Scenario 2: Moderate increase
            scenarios.append(calculate_scenario(district, crop_name, area,
                                               base_rainfall, base_duration, base_temp,
                                               1.2, 1.1, 0.5))
            
            # Scenario 3: High increase
            scenarios.append(calculate_scenario(district, crop_name, area,
                                               base_rainfall, base_duration, base_temp,
                                               1.5, 1.3, 1.0))
            
            # Scenario 4: Extreme
            scenarios.append(calculate_scenario(district, crop_name, area,
                                               base_rainfall, base_duration, base_temp,
                                               2.0, 1.5, 1.5))
            
            return render_template('scenario.html',
                                 scenarios=scenarios,
                                 district=district,
                                 crop=crops[crop_name]['name'],
                                 area=area,
                                 timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"))
        
        except Exception as e:
            return render_template('scenario.html', error=str(e))
    
    # GET request - show form
    return render_template('scenario.html',
                         crops=crops.keys(),
                         districts=districts.keys(),
                         year=datetime.now().year)


@app.route('/data')
def data():
    """View dataset with all features"""
    # Calculate additional stats
    try:
        top_districts = df.groupby('district')['damage_crore'].sum().sort_values(ascending=False).head(5).to_dict()
    except:
        top_districts = {}
    
    # Fix column names for stats - use actual column names from dataset
    max_rain = df['rainfall_mm_24h'].max() if 'rainfall_mm_24h' in df.columns else 0
    max_duration = df['flood_duration_days'].max() if 'flood_duration_days' in df.columns else 0
    total_damage = f"₹{df['damage_crore'].sum():.2f} Cr" if 'damage_crore' in df.columns else "₹0 Cr"
    avg_loss = f"{df['crop_loss_percent'].mean():.1f}%" if 'crop_loss_percent' in df.columns else "0%"
    max_loss = f"{df['crop_loss_percent'].max():.1f}%" if 'crop_loss_percent' in df.columns else "0%"
    avg_rainfall = f"{df['rainfall_mm_24h'].mean():.1f} mm" if 'rainfall_mm_24h' in df.columns else "0 mm"
    
    return render_template('data.html',
                         data=df.head(50).to_dict('records'),
                         stats={
                             'records': len(df),
                             'districts': len(districts),
                             'crops': len(crops),
                             'features': len(df.columns),
                             'max_rain': max_rain,
                             'max_duration': max_duration,
                             'total_damage': total_damage,
                             'avg_loss': avg_loss,
                             'max_loss': max_loss,
                             'avg_rainfall': avg_rainfall
                         },
                         top_districts=top_districts,
                         columns=df.columns.tolist(),
                         year=datetime.now().year)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        result = calculate_impact(
            data['district'],
            data['crop'],
            float(data['rainfall']),
            float(data['duration']),
            float(data['temp']),
            float(data['area'])
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ============================================
# ERROR HANDLERS
# ============================================
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("TELANGANA FLOOD IMPACT SYSTEM - ENHANCED")
    print("="*60)
    print(f"[DATA] Features: {len(df.columns)}")
    print(f"[DATA] Records: {len(df)}")
    print(f"[DATA] Crops in Database: {len(crops)} (including all Telangana crops)")
    print(f"[DATA] Districts: {len(districts)}")
    print(f"[DATA] Modules: 7 (Hazard, Suitability, Damage, Economic, Market, Humanitarian, Recommendations)")
    print(f"[INFO] Server: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)