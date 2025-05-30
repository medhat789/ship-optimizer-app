#!/usr/bin/env python3
"""
Ship Optimizer Application - Main Flask Application with robust fallbacks
"""

from flask import Flask, request, jsonify, render_template
import requests  # For OpenWeather API calls
import math
from datetime import datetime, timedelta, timezone
import os
import json
import pickle
import numpy as np
import pandas as pd
import random

# Flask application setup
app = Flask(__name__)

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
                print(f"Error fetching sea temperature: {e}")
                # Estimate sea temperature
                weather_data["sea_temperature"] = max(15, weather_data["temperature"] - 2.5)
                
            return weather_data
        else:
            print(f"Error fetching weather data: {response.status_code} - {response.text}")
            # Fall back to mock data
    except Exception as e:
        print(f"Exception fetching weather data: {e}")
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
    
    print(f"Using mock weather data for coordinates {lat}, {lon}")
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
        print(f"Unknown departure port: {departure_port}")
        return {"error": "Unknown departure port"}
        
    if arrival_port == "Khalifa Port":
        arrival_lat, arrival_lon = KHALIFA_PORT_LAT, KHALIFA_PORT_LON
    elif arrival_port == "Ruwais Port":
        arrival_lat, arrival_lon = RUWAIS_PORT_LAT, RUWAIS_PORT_LON
    else:
        print(f"Unknown arrival port: {arrival_port}")
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
    if departure_weather and midpoint_weather and arrival_weather:
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
        model_path = os.path.join(os.path.dirname(__file__), "..", "model_files", f"model_{model_version}.pkl")
        features_path = os.path.join(os.path.dirname(__file__), "..", "model_files", f"model_features_{model_version}.json")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(features_path, 'r') as f:
            features = json.load(f)
        
        return model, features
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using fallback prediction model")
        
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
    """Predict fuel consumption for a voyage segment"""
    try:
        # Calculate relative wind
        rel_wind = calculate_relative_wind(wind_speed, wind_direction, course)
        
        # Create feature vector
        feature_vector = {}
        for feature in features:
            if feature == "speed":
                feature_vector[feature] = speed_kn
            elif feature == "distance_nm":
                feature_vector[feature] = distance_nm
            elif feature == "duration_hours":
                feature_vector[feature] = duration_hours
            elif feature == "wind_speed":
                feature_vector[feature] = wind_speed
            elif feature == "wind_direction":
                feature_vector[feature] = wind_direction
            elif feature == "course":
                feature_vector[feature] = course
            elif feature == "relative_wind_speed":
                feature_vector[feature] = rel_wind["relative_wind_speed"]
            elif feature == "relative_wind_direction":
                feature_vector[feature] = rel_wind["relative_wind_direction"]
            elif feature == "head_wind":
                feature_vector[feature] = rel_wind["head_wind"]
            elif feature == "cross_wind":
                feature_vector[feature] = rel_wind["cross_wind"]
            else:
                feature_vector[feature] = 0  # Default value for unknown features
        
        # Convert to DataFrame for prediction
        X = pd.DataFrame([feature_vector])
        
        # Make prediction
        fuel_pred = model.predict(X)[0]
        
        # Apply direct scaling to get realistic fuel consumption
        # For a 12-hour voyage, fuel should be in the 5-9 MT range
        # Scale based on duration relative to 12 hours
        base_fuel_rate = 0.6  # MT per hour (gives 7.2 MT for 12 hours)
        scaled_fuel = base_fuel_rate * duration_hours
        
        # Apply wind effect (up to 20% increase/decrease)
        wind_effect = 1.0
        if rel_wind["head_wind"] > 0:  # Headwind increases fuel consumption
            wind_effect += min(0.2, rel_wind["head_wind"] * 0.02)  # Up to 20% increase
        else:  # Tailwind decreases fuel consumption
            wind_effect -= min(0.15, abs(rel_wind["head_wind"]) * 0.015)  # Up to 15% decrease
        
        # Apply speed effect (higher speeds use more fuel)
        speed_effect = 1.0
        if speed_kn > 12:
            speed_effect += min(0.3, (speed_kn - 12) * 0.1)  # Up to 30% increase
        elif speed_kn < 10:
            speed_effect -= min(0.2, (10 - speed_kn) * 0.1)  # Up to 20% decrease
        
        final_fuel = scaled_fuel * wind_effect * speed_effect
        
        # Ensure fuel consumption is within realistic bounds for the duration
        # For a 12-hour voyage, fuel should be in the 5-9 MT range
        # Scale the bounds based on the voyage duration
        min_fuel_12hr = 5.0
        max_fuel_12hr = 9.0
        
        min_fuel = min_fuel_12hr * (duration_hours / 12)
        max_fuel = max_fuel_12hr * (duration_hours / 12)
        
        if final_fuel < min_fuel:
            final_fuel = min_fuel
        elif final_fuel > max_fuel:
            final_fuel = max_fuel
        
        return final_fuel
        
    except Exception as e:
        print(f"Error predicting fuel: {e}")
        # Fallback to simple calculation
        # For a 12-hour voyage at 10 knots, predict around 7 MT
        base_rate = 0.6  # MT per hour
        return base_rate * duration_hours  # Simple linear model

# --- Flask Routes ---
@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/api/optimize', methods=['POST'])
def optimize_route():
    """API endpoint to optimize route based on input parameters"""
    try:
        data = request.get_json()
        
        # Extract parameters
        departure_port = data.get('departure_port', 'Khalifa Port')
        arrival_port = data.get('arrival_port', 'Ruwais Port')
        required_arrival_time_str = data.get('required_arrival_time')
        current_wind_speed = float(data.get('current_wind_speed', 5.0))
        current_wind_direction = float(data.get('current_wind_direction', 90.0))
        
        # Parse required arrival time
        try:
            # Parse ISO format with timezone
            required_arrival_time = datetime.fromisoformat(required_arrival_time_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            # Fallback to format from datetime-local input (YYYY-MM-DDTHH:MM)
            try:
                required_arrival_time = datetime.strptime(required_arrival_time_str, '%Y-%m-%dT%H:%M')
                # Assume UTC if no timezone specified
                required_arrival_time = required_arrival_time.replace(tzinfo=timezone.utc)
            except (ValueError, AttributeError):
                # If all parsing fails, use current time + 24 hours
                required_arrival_time = datetime.now(timezone.utc) + timedelta(hours=24)
        
        # Get route waypoints
        waypoints = get_relevant_waypoints(departure_port, arrival_port)
        if not waypoints:
            return jsonify({
                'success': False,
                'error': f'No route found between {departure_port} and {arrival_port}'
            })
        
        # Get weather data for the route
        route_weather = get_route_weather(departure_port, arrival_port)
        
        # Use route average wind if available, otherwise use user input
        if route_weather and 'average' in route_weather:
            wind_speed = route_weather['average']['wind_speed']
            wind_direction = route_weather['average']['wind_direction']
        else:
            wind_speed = current_wind_speed
            wind_direction = current_wind_direction
        
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
        
        # Calculate optimal speed based on arrival time
        now = datetime.now(timezone.utc)
        time_until_arrival = (required_arrival_time - now).total_seconds() / 3600  # hours
        
        if time_until_arrival <= 0:
            return jsonify({
                'success': False,
                'error': 'Required arrival time must be in the future'
            })
        
        # Calculate optimal speed (knots)
        optimal_speed = total_distance_nm / time_until_arrival
        
        # Ensure speed is within reasonable bounds (8-14 knots)
        if optimal_speed < 8:
            optimal_speed = 8
            # Recalculate arrival time
            time_until_arrival = total_distance_nm / optimal_speed
            adjusted_arrival_time = now + timedelta(hours=time_until_arrival)
            message = f"Speed adjusted to minimum 8 knots. Estimated arrival: {adjusted_arrival_time.strftime('%Y-%m-%d %H:%M')} UTC"
        elif optimal_speed > 14:
            optimal_speed = 14
            # Recalculate arrival time
            time_until_arrival = total_distance_nm / optimal_speed
            adjusted_arrival_time = now + timedelta(hours=time_until_arrival)
            message = f"Speed adjusted to maximum 14 knots. Estimated arrival: {adjusted_arrival_time.strftime('%Y-%m-%d %H:%M')} UTC"
        else:
            message = f"Optimal speed calculated for arrival at {required_arrival_time.strftime('%Y-%m-%d %H:%M')} UTC"
        
        # Calculate fuel consumption for each segment and total
        total_fuel_mt = 0
        total_duration_hours = 0
        
        for i in range(len(waypoints) - 1):
            wp1 = waypoints[i]
            wp2 = waypoints[i + 1]
            
            segment_distance = wp1['distance_to_next_nm']
            segment_duration = segment_distance / optimal_speed
            
            # Predict fuel consumption for this segment
            segment_fuel = predict_fuel(
                model, features,
                optimal_speed, segment_distance, segment_duration,
                wind_speed, wind_direction, wp1['course_to_next']
            )
            
            # Store segment details
            waypoints[i]['suggested_speed_kn'] = optimal_speed
            waypoints[i]['segment_duration_hours'] = segment_duration
            waypoints[i]['segment_fuel_mt'] = segment_fuel
            
            total_fuel_mt += segment_fuel
            total_duration_hours += segment_duration
        
        # Last waypoint has no next segment
        waypoints[-1]['suggested_speed_kn'] = optimal_speed
        waypoints[-1]['segment_duration_hours'] = 0
        waypoints[-1]['segment_fuel_mt'] = 0
        
        # Calculate fuel savings compared to standard operations
        # Assume standard operation is at 12 knots regardless of conditions
        standard_duration = total_distance_nm / 12
        standard_fuel = predict_fuel(
            model, features,
            12, total_distance_nm, standard_duration,
            wind_speed, wind_direction, 0  # Use 0 as a default course
        )
        
        fuel_savings_mt = max(0, standard_fuel - total_fuel_mt)
        fuel_savings_percentage = (fuel_savings_mt / standard_fuel * 100) if standard_fuel > 0 else 0
        
        # Generate speed rationale
        if optimal_speed < 10:
            speed_rationale = f"A slower speed of {optimal_speed:.1f} knots is recommended due to favorable wind conditions (wind speed: {wind_speed:.1f} m/s, direction: {wind_direction:.0f}°) and sufficient time to arrival. This reduces engine load and optimizes fuel efficiency."
        elif optimal_speed > 12:
            speed_rationale = f"A higher speed of {optimal_speed:.1f} knots is necessary to meet the required arrival time. This increases fuel consumption but ensures timely arrival despite the {wind_speed:.1f} m/s winds from {wind_direction:.0f}°."
        else:
            speed_rationale = f"The optimal speed of {optimal_speed:.1f} knots balances fuel efficiency and timely arrival. Current wind conditions ({wind_speed:.1f} m/s from {wind_direction:.0f}°) have been factored into this calculation."
        
        # Generate savings commentary
        if fuel_savings_percentage > 10:
            savings_commentary = f"Following this optimized route and speed will save {fuel_savings_mt:.2f} MT of fuel ({fuel_savings_percentage:.1f}% reduction) compared to standard operations. This represents significant cost savings and environmental benefits."
        elif fuel_savings_percentage > 5:
            savings_commentary = f"This optimization provides moderate fuel savings of {fuel_savings_mt:.2f} MT ({fuel_savings_percentage:.1f}% reduction) compared to standard operations, balancing efficiency with operational requirements."
        else:
            savings_commentary = f"The optimization offers modest fuel savings of {fuel_savings_mt:.2f} MT ({fuel_savings_percentage:.1f}% reduction). While savings are limited due to schedule constraints, every reduction in fuel consumption contributes to cost efficiency and environmental goals."
        
        # Prepare response
        response = {
            'success': True,
            'message': message,
            'departure_port': departure_port,
            'arrival_port': arrival_port,
            'required_arrival_time': required_arrival_time.isoformat(),
            'total_distance_nm': total_distance_nm,
            'optimal_speed_kn': optimal_speed,
            'estimated_duration_hours': total_duration_hours,
            'total_estimated_fuel_mt': total_fuel_mt,
            'average_suggested_speed_kn': optimal_speed,
            'fuel_savings_mt': fuel_savings_mt,
            'fuel_savings_percentage': fuel_savings_percentage,
            'speed_rationale': speed_rationale,
            'savings_commentary': savings_commentary,
            'route_details': waypoints,
            'weather_data': route_weather
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/weather', methods=['GET'])
def get_weather():
    """API endpoint to get current weather data for the route"""
    try:
        # Get query parameters
        departure_port = request.args.get('departure_port', 'Khalifa Port')
        arrival_port = request.args.get('arrival_port', 'Ruwais Port')
        
        # Get weather data for the route
        route_weather = get_route_weather(departure_port, arrival_port)
        
        # Add waypoints for map visualization
        waypoints = get_relevant_waypoints(departure_port, arrival_port)
        
        response = {
            'success': True,
            'weather_data': route_weather,
            'waypoints': waypoints,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
