#!/usr/bin/env python3
"""
Ship Optimizer Application - Main Flask Application with robust fallbacks
"""

from flask import Flask, request, jsonify, render_template
import requests  # For OpenWeather API calls
import math
from datetime import datetime, timedelta, timezone # Added timezone
import pytz
import os
import json
import pickle
import numpy as np
import pandas as pd
import random
import logging # Added logging

# Flask application setup
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Model configuration
MODEL_PREDICTION_INTERVAL_HOURS = 1

# OpenWeather API configuration with fallback mechanism
OPENWEATHER_API_KEY = "7bdddce2087ef6e1bef9016a37dcecbb"  # This key may need to be updated
USE_MOCK_WEATHER = True  # Set to True to use mock weather data if API fails
KHALIFA_PORT_LAT = 24.8029
KHALIFA_PORT_LON = 54.6451
RUWAIS_PORT_LAT = 24.1114
RUWAIS_PORT_LON = 52.7300

# --- Weather API Functions ---
def get_weather_data(lat, lon):
    """
    Get current weather data from OpenWeather API for the specified coordinates.
    Falls back to mock data if API call fails.
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Dictionary with weather data or mock data if API call fails
    """
    # First try to get real weather data from OpenWeather API
    try:
        # Get current weather
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract relevant weather information
            weather_data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "temperature": data["main"]["temp"],  # Celsius
                "wind_speed": data["wind"]["speed"],  # m/s
                "wind_direction": data["wind"]["deg"],  # degrees (meteorological)
                "humidity": data["main"]["humidity"],  # %
                "pressure": data["main"]["pressure"],  # hPa
                "weather_condition": data["weather"][0]["main"],
                "weather_description": data["weather"][0]["description"],
                "clouds": data["clouds"]["all"] if "clouds" in data else 0,  # %
                "visibility": data["visibility"] / 1000 if "visibility" in data else 0,  # km
                "location_name": data["name"]
            }
            
            # Add rain and snow if available
            if "rain" in data and "1h" in data["rain"]:
                weather_data["rain_1h"] = data["rain"]["1h"]  # mm
            if "snow" in data and "1h" in data["snow"]:
                weather_data["snow_1h"] = data["snow"]["1h"]  # mm
            
            # Try to get sea temperature from additional API call
            try:
                # Use OneCall API for sea temperature
                sea_url = f"https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=minutely,hourly,daily,alerts&appid={OPENWEATHER_API_KEY}&units=metric"
                sea_response = requests.get(sea_url, timeout=10)
                
                if sea_response.status_code == 200:
                    sea_data = sea_response.json()
                    if "current" in sea_data and "sea_level" in sea_data["current"]:
                        weather_data["sea_temperature"] = sea_data["current"]["sea_level"]
                    else:
                        # Estimate sea temperature (typically 2-3°C lower than air in warm regions)
                        weather_data["sea_temperature"] = max(15, weather_data["temperature"] - 2.5)
                else:
                    # Estimate sea temperature
                    weather_data["sea_temperature"] = max(15, weather_data["temperature"] - 2.5)
            except Exception as e:
                app.logger.error(f"Error fetching sea temperature: {e}")
                # Estimate sea temperature
                weather_data["sea_temperature"] = max(15, weather_data["temperature"] - 2.5)
                
            return weather_data
        else:
            app.logger.error(f"Error fetching weather data: {response.status_code} - {response.text}")
            # Fall back to mock data
    except Exception as e:
        app.logger.error(f"Exception fetching weather data: {e}")
        # Fall back to mock data
    
    # Generate mock weather data with some randomization for realism
    # Base weather values with some randomization
    base_temp = 30 + random.uniform(-5, 5)  # 25-35°C
    base_wind_speed = 5 + random.uniform(-2, 3)  # 3-8 m/s
    base_wind_dir = random.randint(0, 359)  # 0-359 degrees
    
    mock_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature": base_temp,
        "sea_temperature": max(15, base_temp - 2.5),  # Sea temp is typically 2-3°C lower than air in warm regions
        "wind_speed": base_wind_speed,
        "wind_direction": base_wind_dir,
        "humidity": random.randint(40, 80),
        "pressure": random.randint(1000, 1020),
        "weather_condition": random.choice(["Clear", "Clouds", "Rain"]),
        "weather_description": random.choice(["clear sky", "few clouds", "scattered clouds", "light rain"]),
        "clouds": random.randint(0, 100),
        "visibility": random.randint(7, 10),
        "location_name": f"Near {lat:.2f}, {lon:.2f}"
    }
    
    app.logger.warning(f"Using mock weather data for coordinates {lat}, {lon}")
    return mock_data

def get_route_weather(departure_port, arrival_port):
    """
    Get weather data for the route between departure and arrival ports.
    
    Args:
        departure_port: Name of departure port
        arrival_port: Name of arrival port
        
    Returns:
        Dictionary with weather data for departure, arrival, and midpoint
    """
    # Get coordinates based on port names
    if departure_port == "Khalifa Port":
        departure_lat, departure_lon = KHALIFA_PORT_LAT, KHALIFA_PORT_LON
    elif departure_port == "Ruwais Port":
        departure_lat, departure_lon = RUWAIS_PORT_LAT, RUWAIS_PORT_LON
    else:
        app.logger.error(f"Unknown departure port: {departure_port}")
        return {"error": "Unknown departure port"}
        
    if arrival_port == "Khalifa Port":
        arrival_lat, arrival_lon = KHALIFA_PORT_LAT, KHALIFA_PORT_LON
    elif arrival_port == "Ruwais Port":
        arrival_lat, arrival_lon = RUWAIS_PORT_LAT, RUWAIS_PORT_LON
    else:
        app.logger.error(f"Unknown arrival port: {arrival_port}")
        return {"error": "Unknown arrival port"}
    
    # Calculate midpoint coordinates (simple average)
    midpoint_lat = (departure_lat + arrival_lat) / 2
    midpoint_lon = (departure_lon + arrival_lon) / 2
    
    # Get weather data for each point
    departure_weather = get_weather_data(departure_lat, departure_lon)
    arrival_weather = get_weather_data(arrival_lat, arrival_lon)
    midpoint_weather = get_weather_data(midpoint_lat, midpoint_lon)
    
    # Combine into route weather data
    route_weather = {
        "departure": departure_weather,
        "midpoint": midpoint_weather,
        "arrival": arrival_weather,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    # Calculate route average wind conditions (simple average)
    # Ensure we have valid data for all points, otherwise use user-provided or default values
    if departure_weather and midpoint_weather and arrival_weather and \
       isinstance(departure_weather, dict) and isinstance(midpoint_weather, dict) and isinstance(arrival_weather, dict) and \
       all(k in departure_weather for k in ["wind_speed", "wind_direction"]) and \
       all(k in midpoint_weather for k in ["wind_speed", "wind_direction"]) and \
       all(k in arrival_weather for k in ["wind_speed", "wind_direction"]):
        
        avg_wind_speed = (departure_weather["wind_speed"] + midpoint_weather["wind_speed"] + arrival_weather["wind_speed"]) / 3
        
        # For wind direction, we need to handle the circular nature (0-360 degrees)
        # Convert to cartesian coordinates, average, then convert back to polar
        wind_x = sum([
            departure_weather["wind_speed"] * math.cos(math.radians(departure_weather["wind_direction"])),
            midpoint_weather["wind_speed"] * math.cos(math.radians(midpoint_weather["wind_direction"])),
            arrival_weather["wind_speed"] * math.cos(math.radians(arrival_weather["wind_direction"]))
        ]) / 3
        
        wind_y = sum([
            departure_weather["wind_speed"] * math.sin(math.radians(departure_weather["wind_direction"])),
            midpoint_weather["wind_speed"] * math.sin(math.radians(midpoint_weather["wind_direction"])),
            arrival_weather["wind_speed"] * math.sin(math.radians(arrival_weather["wind_direction"]))
        ]) / 3
        
        avg_wind_direction = (math.degrees(math.atan2(wind_y, wind_x)) + 360) % 360
        
        route_weather["average"] = {
            "wind_speed": avg_wind_speed,
            "wind_direction": avg_wind_direction
        }
    else:
        app.logger.warning("Could not calculate average route weather, using defaults.")
        # Use default values if we couldn't get weather data
        route_weather["average"] = {
            "wind_speed": 5.0,  # Default wind speed in m/s
            "wind_direction": 90.0  # Default wind direction in degrees
        }
    
    return route_weather

# --- Helper Functions ---
def load_model(model_version="v12"):
    """
    Load the trained model and feature list.
    Falls back to a simple prediction function if model loading fails.
    """
    try:
        # Use app.root_path for robust path construction
        model_dir = os.path.join(app.root_path, 'model_files')
        model_path = os.path.join(model_dir, f"model_{model_version}.pkl")
        features_path = os.path.join(model_dir, f"model_features_{model_version}.json")
        
        app.logger.info(f"Attempting to load model from: {model_path}") 
        app.logger.info(f"Attempting to load features from: {features_path}")
        app.logger.info(f"Files in model directory ({model_dir}): {os.listdir(model_dir) if os.path.exists(model_dir) else 'Directory not found'}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(features_path, 'r') as f:
            features = json.load(f)
        
        app.logger.info("Model and features loaded successfully.")
        return model, features
    except Exception as e:
        app.logger.error(f"Error loading model: {e}", exc_info=True) # Log full traceback
        app.logger.warning("Using fallback prediction model")
        
        # Create a simple fallback model that mimics the real model's interface
        class FallbackModel:
            def predict(self, X):
                # Simple linear model based on speed and duration
                # For a 12-hour voyage at 10 knots, predict around 7 MT
                results = []
                for _, row in X.iterrows():
                    speed = row.get('speed', 10)
                    duration = row.get('duration_hours', 12)
                    
                    # Base fuel rate (MT per hour)
                    base_rate = 0.6  # ~7.2 MT for 12 hours
                    
                    # Speed effect (higher speeds use more fuel)
                    speed_factor = 1.0
                    if speed > 12:
                        speed_factor += (speed - 12) * 0.1  # 10% increase per knot above 12
                    elif speed < 10:
                        speed_factor -= (10 - speed) * 0.05  # 5% decrease per knot below 10
                    
                    # Wind effect if available
                    wind_factor = 1.0
                    if 'head_wind' in row:
                        head_wind = row['head_wind']
                        if head_wind > 0:  # Headwind increases fuel consumption
                            wind_factor += min(0.2, head_wind * 0.02)  # Up to 20% increase
                        else:  # Tailwind decreases fuel consumption
                            wind_factor -= min(0.15, abs(head_wind) * 0.015)  # Up to 15% decrease
                    
                    # Calculate fuel consumption
                    fuel = base_rate * duration * speed_factor * wind_factor
                    
                    # Ensure within reasonable range for a 12-hour voyage
                    fuel_12hr_equivalent = fuel * (12 / duration) if duration > 0 else fuel
                    if fuel_12hr_equivalent < 5:
                        fuel = (5 / 12) * duration
                    elif fuel_12hr_equivalent > 9:
                        fuel = (9 / 12) * duration
                    
                    results.append(fuel)
                
                return np.array(results)
        
        # Create a list of features the fallback model can use
        fallback_features = [
            "speed", "distance_nm", "duration_hours", 
            "wind_speed", "wind_direction", "course",
            "relative_wind_speed", "relative_wind_direction",
            "head_wind", "cross_wind"
        ]
        
        return FallbackModel(), fallback_features

def get_relevant_waypoints(departure_port, arrival_port):
    """Get waypoints for the route between departure and arrival ports"""
    # Simplified waypoints for the Khalifa Port to Ruwais route
    khalifa_to_ruwais = [
        {"name": "Khalifa Port", "lat": 24.8029, "lon": 54.6451, "course_to_next": 315},
        {"name": "Waypoint 1", "lat": 24.8500, "lon": 54.5000, "course_to_next": 300},
        {"name": "Waypoint 2", "lat": 24.9000, "lon": 54.3000, "course_to_next": 290},
        {"name": "Waypoint 3", "lat": 24.9500, "lon": 54.0000, "course_to_next": 280},
        {"name": "Waypoint 4", "lat": 24.9800, "lon": 53.7000, "course_to_next": 270},
        {"name": "Waypoint 5", "lat": 24.9800, "lon": 53.4000, "course_to_next": 260},
        {"name": "Waypoint 6", "lat": 24.9500, "lon": 53.1000, "course_to_next": 250},
        {"name": "Waypoint 7", "lat": 24.9000, "lon": 52.9000, "course_to_next": 240},
        {"name": "Ruwais Port", "lat": 24.1114, "lon": 52.7300, "course_to_next": 0}
    ]
    
    ruwais_to_khalifa = list(reversed(khalifa_to_ruwais))
    # Update course_to_next for reversed route
    for i in range(len(ruwais_to_khalifa) - 1):
        # Calculate opposite course (add 180 degrees and normalize to 0-360)
        ruwais_to_khalifa[i]["course_to_next"] = (khalifa_to_ruwais[-(i+2)]["course_to_next"] + 180) % 360
    
    if departure_port == "Khalifa Port" and arrival_port == "Ruwais Port":
        return khalifa_to_ruwais
    elif departure_port == "Ruwais Port" and arrival_port == "Khalifa Port":
        return ruwais_to_khalifa
    else:
        return []

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in nautical miles using Haversine formula"""
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 3440  # Radius of Earth in nautical miles
    
    return c * r

def calculate_relative_wind(wind_speed, wind_direction, course):
    """Calculate relative wind speed and direction based on true wind and vessel course"""
    # Convert to radians
    wind_dir_rad = math.radians(wind_direction)
    course_rad = math.radians(course)
    
    # Calculate relative wind direction (0 = head wind, 180 = tail wind)
    rel_wind_dir_rad = wind_dir_rad - course_rad
    rel_wind_dir = math.degrees(rel_wind_dir_rad) % 360
    
    # Normalize to -180 to 180 range
    if rel_wind_dir > 180:
        rel_wind_dir -= 360
    
    # Calculate relative wind speed components
    wind_x = wind_speed * math.sin(wind_dir_rad)
    wind_y = wind_speed * math.cos(wind_dir_rad)
    
    ship_x = 0  # Assuming ship speed doesn't affect wind speed calculation
    ship_y = 0
    
    rel_wind_x = wind_x - ship_x
    rel_wind_y = wind_y - ship_y
    
    # Calculate relative wind speed
    rel_wind_speed = math.sqrt(rel_wind_x**2 + rel_wind_y**2)
    
    # Calculate wind components relative to vessel heading
    head_wind = -rel_wind_speed * math.cos(rel_wind_dir_rad)  # Positive = head wind
    cross_wind = rel_wind_speed * math.sin(rel_wind_dir_rad)  # Positive = wind from starboard
    
    return {
        "relative_wind_speed": rel_wind_speed,
        "relative_wind_direction": rel_wind_dir,
        "head_wind": head_wind,
        "cross_wind": cross_wind
    }

def predict_fuel(model, features, speed_kn, distance_nm, duration_hours, wind_speed, wind_direction, course):
    """Predict fuel consumption for a voyage segment using the loaded model"""
    # Prepare input data for the model
    input_data = pd.DataFrame([{ 
        "speed": speed_kn,
        "distance_nm": distance_nm,
        "duration_hours": duration_hours,
        "wind_speed": wind_speed,
        "wind_direction": wind_direction,
        "course": course
    }])
    
    # Calculate relative wind features
    rel_wind = calculate_relative_wind(wind_speed, wind_direction, course)
    input_data["relative_wind_speed"] = rel_wind["relative_wind_speed"]
    input_data["relative_wind_direction"] = rel_wind["relative_wind_direction"]
    input_data["head_wind"] = rel_wind["head_wind"]
    input_data["cross_wind"] = rel_wind["cross_wind"]
    
    # Ensure all required features are present, fill missing with 0 or defaults
    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0 # Or a more appropriate default
            
    # Select only the features the model expects
    input_data = input_data[features]
    
    # Predict fuel consumption
    try:
        fuel_prediction = model.predict(input_data)[0]
        # Ensure prediction is non-negative
        return max(0, fuel_prediction)
    except Exception as e:
        app.logger.error(f"Error during fuel prediction: {e}", exc_info=True)
        # Fallback prediction if model fails
        # Simple estimate: 0.6 MT/hour at 10 knots, adjust for speed
        base_rate = 0.6 * (speed_kn / 10)**2 # Fuel consumption scales roughly with speed squared
        return max(0.1, base_rate * duration_hours) # Ensure a minimum fuel prediction

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """API endpoint to get weather data for the route"""
    departure_port = request.args.get('departure_port')
    arrival_port = request.args.get('arrival_port')
    
    if not departure_port or not arrival_port:
        return jsonify({"error": "Missing departure or arrival port"}), 400
        
    route_weather = get_route_weather(departure_port, arrival_port)
    return jsonify(route_weather)

@app.route('/api/optimize', methods=['POST'])
def optimize_route():
    """API endpoint to optimize the voyage based on input parameters"""
    try:
        data = request.get_json()
        app.logger.info(f"Received optimization request data: {data}") # Log raw input
        
        departure_port = data.get('departure_port')
        arrival_port = data.get('arrival_port')
        required_arrival_time_str = data.get('required_arrival_time')
        current_wind_speed = float(data.get('wind_speed', 5.0)) # Default 5 m/s
        current_wind_direction = float(data.get('wind_direction', 90.0)) # Default 90 degrees
        
        # Validate inputs
        if not all([departure_port, arrival_port, required_arrival_time_str]):
            app.logger.error("Missing required input parameters")
            return jsonify({"error": "Missing required input parameters"}), 400
            
        # Get waypoints for the route
        waypoints = get_relevant_waypoints(departure_port, arrival_port)
        if not waypoints:
            app.logger.error(f"Invalid route specified: {departure_port} to {arrival_port}")
            return jsonify({"error": "Invalid route specified"}), 400
            
        # Define Dubai timezone
        dubai_tz = pytz.timezone('Asia/Dubai')

        try:
            app.logger.info(f"Parsing arrival time string: \t{required_arrival_time_str}")
            # Attempt to parse the arrival time string, trying multiple formats
            parsed_time = None
            formats_to_try = [
                "%Y-%m-%dT%H:%M", # Format from datetime-local input
                "%Y-%m-%dT%H:%M:%S", # ISO format without Z
                "%Y-%m-%d %H:%M:%S", # Space separator
                "%Y-%m-%dT%H:%M:%S%z", # ISO format with UTC offset (e.g., +0400)
                "%Y-%m-%dT%H:%M:%S.%f%z", # ISO format with microseconds and offset
                "%m/%d/%Y, %I:%M %p" # Format from frontend example: 05/31/2025, 12:00 AM
            ]
            
            for fmt in formats_to_try:
                try:
                    parsed_time = datetime.strptime(required_arrival_time_str, fmt)
                    app.logger.info(f"Successfully parsed \t{required_arrival_time_str}\t using format \t{fmt}")
                    break # Stop trying formats once one works
                except ValueError:
                    continue # Try next format

            if parsed_time is None:
                 # If all formats failed, raise an error
                 raise ValueError("Could not parse arrival time string with any known format.")

            # Ensure the parsed time is timezone-aware and in Dubai timezone
            if parsed_time.tzinfo is None or parsed_time.tzinfo.utcoffset(parsed_time) is None:
                # Input was naive, assume it represents Dubai time
                required_arrival_time = dubai_tz.localize(parsed_time)
                app.logger.info(f"Localized naive time to Dubai TZ: {required_arrival_time}")
            else:
                # Input was already timezone-aware, convert it to Dubai time
                required_arrival_time = parsed_time.astimezone(dubai_tz)
                app.logger.info(f"Converted aware time to Dubai TZ: {required_arrival_time}")

        except (ValueError, TypeError, AttributeError) as e:
            app.logger.error(f"Error processing arrival time \t{required_arrival_time_str}\t: {e}", exc_info=True)
            return jsonify({"error": f"Invalid arrival time format: \t{required_arrival_time_str}\t. Please use a standard format like YYYY-MM-DDTHH:MM:SS or MM/DD/YYYY, HH:MM AM/PM."}), 400

        # Get current time, guaranteed to be timezone-aware in Dubai timezone
        now = datetime.now(dubai_tz)

        # Log before subtraction
        app.logger.info(f"NOW (Dubai): {now}, TZ: {now.tzinfo}, Type: {type(now)}")
        app.logger.info(f"ARRIVAL (Dubai): {required_arrival_time}, TZ: {required_arrival_time.tzinfo}, Type: {type(required_arrival_time)}")

        # Calculate time difference (both should now be aware and in the same timezone)
        time_until_arrival = (required_arrival_time - now).total_seconds() / 3600  # hours
        app.logger.info(f"Time until arrival (hours): {time_until_arrival}")

        if time_until_arrival <= 0:
            app.logger.error(f"Required arrival time {required_arrival_time} is not in the future relative to {now}")
            return jsonify({"error": "Required arrival time must be in the future"}), 400
            
        # Get weather data for the route
        route_weather = get_route_weather(departure_port, arrival_port)
        
        # Use route average wind if available, otherwise use user input
        if route_weather and 'average' in route_weather and isinstance(route_weather['average'], dict):
            wind_speed = route_weather['average'].get('wind_speed', current_wind_speed)
            wind_direction = route_weather['average'].get('wind_direction', current_wind_direction)
            app.logger.info(f"Using average route weather: Speed={wind_speed}, Dir={wind_direction}")
        else:
            wind_speed = current_wind_speed
            wind_direction = current_wind_direction
            app.logger.info(f"Using user-provided weather: Speed={wind_speed}, Dir={wind_direction}")
        
        # Load model
        model, features = load_model()
        
        # Calculate total distance
        total_distance_nm = 0
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            segment_distance = calculate_distance(wp1['lat'], wp1['lon'], wp2['lat'], wp2['lon'])
            total_distance_nm += segment_distance
            waypoints[i]['distance_to_next_nm'] = segment_distance
        
        # Last waypoint has no next waypoint
        waypoints[-1]['distance_to_next_nm'] = 0
        app.logger.info(f"Total route distance: {total_distance_nm:.2f} nm")
        
        # Calculate optimal speed based on arrival time
        optimal_speed_kn = total_distance_nm / time_until_arrival
        app.logger.info(f"Calculated optimal speed: {optimal_speed_kn:.2f} knots")
        
        # Basic validation for speed (e.g., vessel max speed)
        MAX_SPEED_KN = 15 # Example max speed
        MIN_SPEED_KN = 5  # Example min speed
        if not (MIN_SPEED_KN <= optimal_speed_kn <= MAX_SPEED_KN):
             app.logger.error(f"Calculated speed {optimal_speed_kn:.1f} knots is outside operational limits ({MIN_SPEED_KN}-{MAX_SPEED_KN} knots)")
             return jsonify({
                "error": f"Required arrival time results in an unrealistic speed ({optimal_speed_kn:.1f} knots). Please adjust arrival time. Min: {MIN_SPEED_KN} kn, Max: {MAX_SPEED_KN} kn."
            }), 400
            
        # Simulate voyage segment by segment
        voyage_plan = []
        total_fuel_predicted = 0
        current_time = now
        
        app.logger.info("Starting voyage simulation...")
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            segment_distance = wp1['distance_to_next_nm']
            segment_duration_hours = segment_distance / optimal_speed_kn
            segment_course = wp1['course_to_next']
            
            # Predict fuel for this segment
            segment_fuel = predict_fuel(
                model, features, optimal_speed_kn, segment_distance, 
                segment_duration_hours, wind_speed, wind_direction, segment_course
            )
            total_fuel_predicted += segment_fuel
            
            # Calculate segment end time
            segment_end_time = current_time + timedelta(hours=segment_duration_hours)
            
            voyage_plan.append({
                "segment": i + 1,
                "from": wp1['name'],
                "to": wp2['name'],
                "distance_nm": round(segment_distance, 2),
                "course": segment_course,
                "speed_kn": round(optimal_speed_kn, 2),
                "duration_hours": round(segment_duration_hours, 2),
                "predicted_fuel_mt": round(segment_fuel, 3),
                "estimated_departure_time": current_time.isoformat(),
                "estimated_arrival_time": segment_end_time.isoformat()
            })
            
            # Update current time for next segment
            current_time = segment_end_time
            
        app.logger.info(f"Voyage simulation complete. Total predicted fuel: {total_fuel_predicted:.3f} MT")
        # Prepare results
        results = {
            "success": True, # Add success flag for frontend
            "departure_port": departure_port,
            "arrival_port": arrival_port,
            "required_arrival_time": required_arrival_time.isoformat(),
            "total_distance_nm": round(total_distance_nm, 2),
            "calculated_optimal_speed_kn": round(optimal_speed_kn, 2),
            "estimated_voyage_duration_hours": round(total_distance_nm / optimal_speed_kn, 2),
            "total_predicted_fuel_mt": round(total_fuel_predicted, 3),
            "wind_speed_used_mps": round(wind_speed, 2),
            "wind_direction_used_deg": round(wind_direction, 1),
            "voyage_plan": voyage_plan
        }
        
        return jsonify(results)
        
    except Exception as e:
        app.logger.error(f"Error in /api/optimize: {e}", exc_info=True)
        # import traceback
        # traceback.print_exc() # Print detailed traceback to logs (already done by logger)
        return jsonify({"error": "An internal error occurred during optimization."}), 500

if __name__ == '__main__':
    # Use Gunicorn in production, this is for local development
    # Set debug=False for production-like testing
    app.run(debug=False, host='0.0.0.0', port=5000)

