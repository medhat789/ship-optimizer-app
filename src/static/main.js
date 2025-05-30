// Main JavaScript for Ship Optimizer App
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tabs
    const tabs = document.querySelectorAll('.nav-link');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all tabs and hide all tab contents
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(content => content.classList.add('hidden'));
            
            // Add active class to clicked tab and show corresponding content
            this.classList.add('active');
            const targetId = this.getAttribute('data-target');
            document.getElementById(targetId).classList.remove('hidden');
            
            // If weather tab is clicked, refresh weather data
            if (targetId === 'weather-tab') {
                fetchWeatherData();
            }
        });
    });
    
    // Set default tab
    if (tabs.length > 0) {
        tabs[0].click();
    }
    
    // Initialize form submission
    const optimizerForm = document.getElementById('optimizer-form');
    if (optimizerForm) {
        optimizerForm.addEventListener('submit', function(e) {
            e.preventDefault();
            getOptimization();
        });
    }
    
    // Set default arrival time to 24 hours from now
    const arrivalTimeInput = document.getElementById('required-arrival-time');
    if (arrivalTimeInput) {
        const now = new Date();
        now.setHours(now.getHours() + 24);
        now.setMinutes(0);
        now.setSeconds(0);
        now.setMilliseconds(0);
        const year = now.getFullYear();
        const month = (now.getMonth() + 1).toString().padStart(2, '0');
        const day = now.getDate().toString().padStart(2, '0');
        const hours = now.getHours().toString().padStart(2, '0');
        const minutes = now.getMinutes().toString().padStart(2, '0');
        arrivalTimeInput.value = `${year}-${month}-${day}T${hours}:${minutes}`;
    }
    
    // Initialize weather data
    fetchWeatherData();
    
    // Set up weather refresh interval (every 15 minutes)
    setInterval(fetchWeatherData, 15 * 60 * 1000);
    
    // Initialize maps
    initializeRouteMaps();
});

// Global variables for maps
let routeMap = null;
let weatherMap = null;
let routeLayer = null;
let weatherLayer = null;
let waypointMarkers = [];

function initializeRouteMaps() {
    // Initialize route map in results tab
    const routeMapContainer = document.getElementById('route-map-container');
    if (routeMapContainer) {
        // Create a div with specific height for the route map
        routeMapContainer.style.height = '400px';
        routeMapContainer.style.width = '100%';
        routeMapContainer.style.marginTop = '1rem';
        routeMapContainer.style.marginBottom = '1rem';
        routeMapContainer.style.borderRadius = 'var(--border-radius)';
        routeMapContainer.style.border = '1px solid var(--gray-light)';
        
        routeMap = L.map('route-map-container').setView([24.5, 53.7], 9);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(routeMap);
    }
    
    // Initialize weather map in weather tab
    const weatherMapContainer = document.getElementById('weather-map-container');
    if (weatherMapContainer) {
        // Create a div with specific height for the weather map
        weatherMapContainer.style.height = '400px';
        weatherMapContainer.style.width = '100%';
        weatherMapContainer.style.marginTop = '1rem';
        weatherMapContainer.style.marginBottom = '1rem';
        weatherMapContainer.style.borderRadius = 'var(--border-radius)';
        weatherMapContainer.style.border = '1px solid var(--gray-light)';
        
        weatherMap = L.map('weather-map-container').setView([24.5, 53.7], 9);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(weatherMap);
        
        // Add default route to weather map
        const defaultWaypoints = getDefaultWaypoints("Khalifa Port", "Ruwais Port");
        drawRouteOnMap(weatherMap, defaultWaypoints, true);
    }
}

function getDefaultWaypoints(departurePort, arrivalPort) {
    // Simplified waypoints for the Khalifa Port to Ruwais route
    const khalifaToRuwais = [
        {"name": "Khalifa Port", "lat": 24.8029, "lon": 54.6451, "course_to_next": 315},
        {"name": "Waypoint 1", "lat": 24.8500, "lon": 54.5000, "course_to_next": 300},
        {"name": "Waypoint 2", "lat": 24.9000, "lon": 54.3000, "course_to_next": 290},
        {"name": "Waypoint 3", "lat": 24.9500, "lon": 54.0000, "course_to_next": 280},
        {"name": "Waypoint 4", "lat": 24.9800, "lon": 53.7000, "course_to_next": 270},
        {"name": "Waypoint 5", "lat": 24.9800, "lon": 53.4000, "course_to_next": 260},
        {"name": "Waypoint 6", "lat": 24.9500, "lon": 53.1000, "course_to_next": 250},
        {"name": "Waypoint 7", "lat": 24.9000, "lon": 52.9000, "course_to_next": 240},
        {"name": "Ruwais Port", "lat": 24.1114, "lon": 52.7300, "course_to_next": 0}
    ];
    
    const ruwaisToKhalifa = [
        {"name": "Ruwais Port", "lat": 24.1114, "lon": 52.7300, "course_to_next": 60},
        {"name": "Waypoint 7", "lat": 24.9000, "lon": 52.9000, "course_to_next": 70},
        {"name": "Waypoint 6", "lat": 24.9500, "lon": 53.1000, "course_to_next": 80},
        {"name": "Waypoint 5", "lat": 24.9800, "lon": 53.4000, "course_to_next": 90},
        {"name": "Waypoint 4", "lat": 24.9800, "lon": 53.7000, "course_to_next": 100},
        {"name": "Waypoint 3", "lat": 24.9500, "lon": 54.0000, "course_to_next": 110},
        {"name": "Waypoint 2", "lat": 24.9000, "lon": 54.3000, "course_to_next": 120},
        {"name": "Waypoint 1", "lat": 24.8500, "lon": 54.5000, "course_to_next": 135},
        {"name": "Khalifa Port", "lat": 24.8029, "lon": 54.6451, "course_to_next": 0}
    ];
    
    if (departurePort === "Khalifa Port" && arrivalPort === "Ruwais Port") {
        return khalifaToRuwais;
    } else if (departurePort === "Ruwais Port" && arrivalPort === "Khalifa Port") {
        return ruwaisToKhalifa;
    } else {
        return [];
    }
}

function drawRouteOnMap(map, waypoints, isWeatherMap = false) {
    // Clear existing route and markers
    if (map) {
        map.eachLayer(function(layer) {
            if (layer instanceof L.Polyline || layer instanceof L.Marker) {
                map.removeLayer(layer);
            }
        });
        
        if (waypoints && waypoints.length > 0) {
            // Create route line
            const routeCoordinates = waypoints.map(wp => [wp.lat, wp.lon]);
            const routeLine = L.polyline(routeCoordinates, {
                color: '#0066cc',
                weight: 4,
                opacity: 0.8
            }).addTo(map);
            
            // Add markers for each waypoint
            waypoints.forEach((waypoint, index) => {
                const isEndpoint = index === 0 || index === waypoints.length - 1;
                const markerIcon = L.divIcon({
                    className: isEndpoint ? 'endpoint-marker' : 'waypoint-marker',
                    html: `<i class="fas ${isEndpoint ? 'fa-anchor' : 'fa-map-marker-alt'}" style="color: ${isEndpoint ? '#ff9900' : '#0066cc'}; font-size: ${isEndpoint ? '24px' : '18px'};"></i>`,
                    iconSize: [24, 24],
                    iconAnchor: [12, 24]
                });
                
                const marker = L.marker([waypoint.lat, waypoint.lon], {
                    icon: markerIcon,
                    title: waypoint.name
                }).addTo(map);
                
                // Add popup with waypoint info
                const popupContent = `
                    <strong>${waypoint.name}</strong><br>
                    Lat: ${waypoint.lat.toFixed(4)}<br>
                    Lon: ${waypoint.lon.toFixed(4)}
                    ${waypoint.course_to_next ? '<br>Course: ' + waypoint.course_to_next + '°' : ''}
                    ${waypoint.suggested_speed_kn ? '<br>Speed: ' + waypoint.suggested_speed_kn + ' knots' : ''}
                `;
                marker.bindPopup(popupContent);
                
                // If this is a weather map, add wind direction indicators
                if (isWeatherMap && index % 2 === 0) { // Add wind indicators at every other waypoint
                    addWindIndicator(map, waypoint.lat, waypoint.lon, 90); // Default direction, will be updated
                }
            });
            
            // Fit map to route bounds
            map.fitBounds(routeLine.getBounds(), {
                padding: [50, 50]
            });
        }
    }
}

function addWindIndicator(map, lat, lon, windDirection) {
    // Create a wind direction arrow
    const arrowIcon = L.divIcon({
        className: 'wind-indicator',
        html: `<div style="transform: rotate(${windDirection}deg); color: #ff9900; font-size: 16px;">➤</div>`,
        iconSize: [20, 20],
        iconAnchor: [10, 10]
    });
    
    L.marker([lat, lon], {
        icon: arrowIcon,
        interactive: false
    }).addTo(map);
}

function updateWindIndicators(map, windDirection) {
    // Update all wind indicators on the map
    const windIndicators = document.querySelectorAll('.wind-indicator div');
    windIndicators.forEach(indicator => {
        indicator.style.transform = `rotate(${windDirection}deg)`;
    });
}

function fetchWeatherData() {
    // Get current port selections
    const departurePort = document.getElementById('departure-port')?.value || 'Khalifa Port';
    const arrivalPort = document.getElementById('arrival-port')?.value || 'Ruwais Port';
    
    // Fetch weather data from API
    fetch(`/api/weather?departure_port=${encodeURIComponent(departurePort)}&arrival_port=${encodeURIComponent(arrivalPort)}`)
        .then(response => response.json())
        .then(data => {
            if (data.success && data.weather_data) {
                updateWeatherUI(data.weather_data);
                
                // Update weather map with waypoints if available
                if (data.waypoints && weatherMap) {
                    drawRouteOnMap(weatherMap, data.waypoints, true);
                    
                    // Update wind indicators based on weather data
                    if (data.weather_data.average && data.weather_data.average.wind_direction) {
                        updateWindIndicators(weatherMap, data.weather_data.average.wind_direction);
                    }
                }
            } else {
                console.error('Error fetching weather data:', data.error || 'Unknown error');
            }
        })
        .catch(error => {
            console.error('Error fetching weather data:', error);
        });
}

function updateWeatherUI(weatherData) {
    // Update weather metrics in the UI
    if (weatherData) {
        // Update wind data
        if (weatherData.average) {
            updateMetric('wind-speed', weatherData.average.wind_speed, 'm/s');
            updateMetric('wind-direction', weatherData.average.wind_direction, '°');
            
            // Also update the form inputs for consistency
            const windSpeedInput = document.getElementById('current-wind-speed');
            const windDirInput = document.getElementById('current-wind-direction');
            
            if (windSpeedInput) windSpeedInput.value = weatherData.average.wind_speed.toFixed(1);
            if (windDirInput) windDirInput.value = Math.round(weatherData.average.wind_direction);
        }
        
        // Update other weather metrics if available
        if (weatherData.departure) {
            // Use departure port data for most metrics
            if (weatherData.departure.visibility) {
                updateMetric('visibility', weatherData.departure.visibility, 'km');
            }
            
            if (weatherData.departure.humidity) {
                updateMetric('humidity', weatherData.departure.humidity, '%');
            }
            
            if (weatherData.departure.pressure) {
                updateMetric('pressure', weatherData.departure.pressure, 'hPa');
            }
            
            // Update sea temperature
            if (weatherData.departure.sea_temperature) {
                updateMetric('sea-temperature', weatherData.departure.sea_temperature, '°C');
                
                // Update all elements with sea-temp class
                const seaTempElements = document.querySelectorAll('.sea-temp');
                seaTempElements.forEach(el => {
                    el.textContent = weatherData.departure.sea_temperature.toFixed(1);
                });
            }
            
            // Update ambient temperature
            if (weatherData.departure.temperature) {
                updateMetric('ambient-temperature', weatherData.departure.temperature, '°C');
                
                // Update all elements with ambient-temp class
                const ambientTempElements = document.querySelectorAll('.ambient-temp');
                ambientTempElements.forEach(el => {
                    el.textContent = weatherData.departure.temperature.toFixed(1);
                });
            }
            
            // Calculate wave height based on wind speed (simplified Beaufort scale)
            const windSpeed = weatherData.departure.wind_speed || weatherData.average.wind_speed;
            let waveHeight = 0.1;  // Default calm
            
            if (windSpeed > 3 && windSpeed <= 6) {
                waveHeight = 0.2;  // Light breeze
            } else if (windSpeed > 6 && windSpeed <= 10) {
                waveHeight = 0.6;  // Moderate breeze
            } else if (windSpeed > 10 && windSpeed <= 16) {
                waveHeight = 1.0;  // Fresh breeze
            } else if (windSpeed > 16 && windSpeed <= 21) {
                waveHeight = 2.0;  // Strong breeze
            } else if (windSpeed > 21) {
                waveHeight = 3.0;  // Near gale or higher
            }
            
            updateMetric('wave-height', waveHeight, 'm');
        }
    }
}

function updateMetric(id, value, unit) {
    const element = document.getElementById(id);
    if (element) {
        const valueElement = element.querySelector('.metric-value');
        if (valueElement) {
            valueElement.textContent = typeof value === 'number' ? value.toFixed(1) : value;
        }
    }
}

async function getOptimization() {
    const form = document.getElementById("optimizer-form");
    const resultsDiv = document.getElementById("optimization-results");
    const loadingElement = document.getElementById("loading-indicator");
    const errorElement = document.getElementById("error-message");
    const optimizationInsights = document.getElementById("optimization-insights");

    // Clear previous results and show loading
    if (resultsDiv) resultsDiv.innerHTML = "";
    if (errorElement) errorElement.classList.add('hidden');
    if (loadingElement) loadingElement.classList.remove('hidden');
    if (optimizationInsights) optimizationInsights.classList.add('hidden');

    const formData = new FormData(form);
    const data = {
        departure_port: formData.get("departure_port"),
        arrival_port: formData.get("arrival_port"),
        required_arrival_time: formData.get("required_arrival_time"),
        current_wind_speed: parseFloat(formData.get("current_wind_speed")),
        current_wind_direction: parseFloat(formData.get("current_wind_direction")),
    };

    // Basic validation
    if (data.departure_port === data.arrival_port) {
        showError("Departure and arrival ports cannot be the same.");
        return;
    }
    if (!data.required_arrival_time) {
        showError("Required arrival time must be set.");
        return;
    }

    try {
        const response = await fetch("/api/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        if (loadingElement) loadingElement.classList.add('hidden');
        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || `HTTP error! status: ${response.status}`);
        }

        displayResults(result);
        
        // Show the results tab after successful optimization
        const resultsTab = document.querySelector('[data-target="results-tab"]');
        if (resultsTab) {
            resultsTab.click();
        }
        
        // Update weather data if available
        if (result.weather_data) {
            updateWeatherUI(result.weather_data);
        }
        
    } catch (error) {
        if (loadingElement) loadingElement.classList.add('hidden');
        showError(`Error: ${error.message}`);
        if (resultsDiv) resultsDiv.innerHTML = "<p>Could not retrieve optimization results.</p>";
    }
}

function displayResults(result) {
    const resultsDiv = document.getElementById("optimization-results");
    if (!resultsDiv) return;
    
    // Update metrics
    updateMetric('total-fuel', result.total_estimated_fuel_mt, 'MT');
    updateMetric('voyage-duration', result.estimated_duration_hours, 'hours');
    updateMetric('avg-speed', result.average_suggested_speed_kn, 'knots');
    
    // Update fuel savings metrics
    if (document.getElementById('fuel-savings')) {
        updateMetric('fuel-savings', result.fuel_savings_mt || 0, 'MT');
        const savingsPercentage = document.getElementById('savings-percentage');
        if (savingsPercentage) {
            const percentage = result.fuel_savings_percentage || 0;
            savingsPercentage.textContent = `${percentage.toFixed(1)}% savings`;
            savingsPercentage.classList.add('positive');
        }
    }
    
    // Display route details table
    let tableHtml = '';
    if (result.route_details && result.route_details.length > 0) {
        tableHtml = `
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Segment</th>
                            <th>Waypoint</th>
                            <th>Course To Next (°)</th>
                            <th>Distance (nm)</th>
                            <th>Suggested Speed (kn)</th>
                        </tr>
                    </thead>
                    <tbody>
        `;
        
        result.route_details.forEach((segment, index) => {
            tableHtml += `
                <tr>
                    <td>${index + 1}</td>
                    <td>${segment.name || "N/A"}</td>
                    <td>${formatValue(segment.course_to_next, 1)}</td>
                    <td>${formatValue(segment.distance_to_next_nm, 1)}</td>
                    <td>${formatValue(segment.suggested_speed_kn, 1)}</td>
                </tr>
            `;
        });
        
        tableHtml += `
                    </tbody>
                </table>
            </div>
        `;
    } else {
        tableHtml = `<p>No route details available.</p>`;
    }
    
    document.getElementById('route-details').innerHTML = tableHtml;
    
    // Display message if available
    if (result.message) {
        const messageElement = document.getElementById('optimization-message');
        if (messageElement) {
            messageElement.textContent = result.message;
            messageElement.classList.remove('hidden');
        }
    }
    
    // Update rationale and commentary
    const speedRationale = document.getElementById('speed-rationale');
    if (speedRationale && result.speed_rationale) {
        speedRationale.textContent = result.speed_rationale;
    } else if (speedRationale) {
        speedRationale.textContent = "Speed optimized based on current conditions and arrival requirements.";
    }
    
    const savingsCommentary = document.getElementById('savings-commentary');
    if (savingsCommentary && result.savings_commentary) {
        savingsCommentary.textContent = result.savings_commentary;
        savingsCommentary.classList.add('alert-success');
    } else if (savingsCommentary) {
        savingsCommentary.textContent = "Following the optimized route and speed recommendations will result in fuel savings compared to standard operations.";
    }
    
    // Show optimization insights section
    const insightsSection = document.getElementById('optimization-insights');
    if (insightsSection) {
        insightsSection.classList.remove('hidden');
    }
    
    // Show results section
    document.getElementById('results-section').classList.remove('hidden');
    
    // Update route visualization map
    if (routeMap && result.route_details && result.route_details.length > 0) {
        // Convert route details to waypoints format
        const waypoints = result.route_details.map(segment => ({
            name: segment.name || "Waypoint",
            lat: segment.lat || 0,
            lon: segment.lon || 0,
            course_to_next: segment.course_to_next || 0,
            suggested_speed_kn: segment.suggested_speed_kn || 0
        }));
        
        // Draw route on map
        drawRouteOnMap(routeMap, waypoints);
        
        // Also update the weather map if available
        if (weatherMap) {
            drawRouteOnMap(weatherMap, waypoints, true);
            
            // Update wind indicators if weather data is available
            if (result.weather_data && result.weather_data.average) {
                updateWindIndicators(weatherMap, result.weather_data.average.wind_direction);
            }
        }
    }
}

function formatValue(value, decimals = 0) {
    if (value === undefined || value === null) return "N/A";
    return typeof value === 'number' ? value.toFixed(decimals) : value;
}

function showError(message) {
    const errorElement = document.getElementById("error-message");
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.classList.remove('hidden');
        
        // Hide loading indicator
        const loadingElement = document.getElementById("loading-indicator");
        if (loadingElement) loadingElement.classList.add('hidden');
    }
}
