# fishnet_tn.py - Tamil Nadu Coastal Fisheries AI System
import streamlit as st
import requests
from ultralytics import YOLO
import numpy as np
from PIL import Image
import geocoder
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point, Polygon
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import geopy.distance
import plotly.express as px

# ===========================
# 🚀 Configuration & Constants
# ===========================
st.set_page_config(
    page_title="Tamil Nadu Coastal Fisheries AI",
    layout="wide",
    page_icon="🐟"
)

# API Keys (store in .streamlit/secrets.toml)
try:
    API_KEYS = {
        "OPENWEATHER": st.secrets["OPENWEATHER_API_KEY"],
        "IUCN": st.secrets["IUCN_API_KEY"],
        "ALPHA_VANTAGE": st.secrets["ALPHA_VANTAGE_KEY"],
        "FISHBASE": st.secrets["FISHBASE_API_KEY"],
        "IMD": st.secrets["IMD_API_KEY"]
    }
except KeyError as e:
    st.error(f"⚠️ Missing API key: {e}. Check secrets configuration.")
    st.stop()

# Tamil Nadu Coastal Boundaries
TN_COASTAL_BOUNDS = {
    "min_lat": 8.0, "max_lat": 13.5,
    "min_lon": 77.0, "max_lon": 80.5
}

TN_COASTAL_POLYGON = Polygon([
    (8.0883, 77.5432), (9.2833, 79.3167),
    (10.7667, 79.85), (13.0833, 80.2833)
])

# Local Marine Species Database
LOCAL_SPECIES = {
    "fish": {
        "status": "Common",
        "regulations": "Standard fishing regulations apply",
        "image": "https://upload.wikimedia.org/wikipedia/commons/2/23/Indian_mackerel.jpg"
    },
    "boat": {
        "status": "Registered",
        "regulations": "Valid license required",
        "image": "https://upload.wikimedia.org/wikipedia/commons/6/63/Fishing_boat_Portugal.jpg"
    }
}

# ===========================
# 🔄 Core Functionality
# ===========================
class EnhancedFisheriesSystem:
    def __init__(self):
        self.model = self.load_model()
        self.current_location = self.get_coastal_location()
        self.weather_data = self.get_coastal_weather(*self.current_location)
        self.fishing_status = self.check_fishing_ban()
        self.ocean_data = get_real_time_ocean_data(*self.current_location)
        
    @st.cache_resource
    def load_model(_self):
        """Load default YOLOv8 model"""
        try:
            model = YOLO('yolov8n.pt')
            model.conf = 0.6
            return model
        except Exception as e:
            st.error(f"Model Loading Error: {str(e)}")
            return None
    
    def get_coastal_location(self):
        """Get current GPS coordinates within TN coastal area"""
        try:
            g = geocoder.ip('me')
            if g.latlng:
                lat = max(TN_COASTAL_BOUNDS["min_lat"], min(g.latlng[0], TN_COASTAL_BOUNDS["max_lat"]))
                lon = max(TN_COASTAL_BOUNDS["min_lon"], min(g.latlng[1], TN_COASTAL_BOUNDS["max_lon"]))
                return [lat, lon]
            return [10.7, 79.8]  # Default: Nagapattinam
        except:
            return [10.7, 79.8]
    
    @st.cache_data(ttl=300)
    def get_coastal_weather(_self, lat, lon):
        """Get weather data with caching"""
        try:
            response = requests.get(
                "https://api.openweathermap.org/data/2.5/weather",
                params={
                    "lat": lat,
                    "lon": lon,
                    "appid": API_KEYS["OPENWEATHER"],
                    "units": "metric"
                },
                timeout=10
            )
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "wind_speed": data["wind"]["speed"],
                "condition": data["weather"][0]["description"]
            }
        except:
            return {
                "temp": 24.38,
                "wind_speed": 5.0,
                "condition": "Clear Sky"
            }
    
    def check_fishing_ban(self):
        """Check 60-day fishing ban status"""
        today = datetime.now().date()
        ban_start = datetime(today.year, 6, 15).date()
        ban_end = datetime(today.year, 8, 15).date()
        return "active" if ban_start <= today <= ban_end else "inactive"
    
    def calculate_emissions(self, fuel_type, consumption_rate, hours):
        """Calculate CO2 emissions with TN-specific factors"""
        factors = {"diesel": 2.68, "petrol": 2.31}
        return round(consumption_rate * hours * factors[fuel_type.lower()], 2)
# ===========================
# 🌊 Real-time Ocean Data
# ===========================
@st.cache_data(ttl=600)
def get_real_time_ocean_data(lat, lon):
    """Fetch real-time ocean conditions from IMD"""
    try:
        response = requests.get(
            "https://api.imd.gov.in/ocean",
            params={
                "lat": lat,
                "lon": lon,
                "key": API_KEYS["IMD"]
            },
            timeout=5
        )
        data = response.json()
        return {
            "wave_height": data.get("waveHeight", 0.0),
            "water_temp": data.get("waterTemp", 24.38)
        }
    except:
        return {
            "wave_height": 0.0,
            "water_temp": 24.38
        }

# ===========================
# 🗺️ Coastal Mapping
# ===========================
def create_coastal_map(location):
    """Create interactive coastal map"""
    m = folium.Map(location=location, zoom_start=9)
    
    folium.GeoJson(
        TN_COASTAL_POLYGON,
        name="TN Coastal Waters",
        style_function=lambda x: {"color": "blue", "weight": 3}
    ).add_to(m)
    
    folium.Marker(
        location,
        tooltip="Your Position",
        icon=folium.Icon(color="green", icon="ship")
    ).add_to(m)
    
    return m

# ===========================
# 🎣 TN Fishing Advisor
# ===========================
class TamilNaduFishingAdvisor:
    def __init__(self, weather, ocean_data, location):
        self.weather = weather
        self.ocean_data = ocean_data
        self.location = location
        self.base_movement = 0.02  # Degrees (~2km)
        
    def predict_movement(self):
        """Predict fish movement within TN coastal bounds"""
        try:
            # Consider both weather and ocean conditions
            wind_factor = self.weather["wind_speed"] / 5
            current_factor = self.ocean_data.get("current_speed", 0.5)
            
            new_lat = self.location[0] + (self.base_movement * wind_factor)
            new_lon = self.location[1] + (self.base_movement * current_factor)
            
            # Ensure within TN coastal bounds
            new_lat = max(TN_COASTAL_BOUNDS["min_lat"], min(new_lat, TN_COASTAL_BOUNDS["max_lat"]))
            new_lon = max(TN_COASTAL_BOUNDS["min_lon"], min(new_lon, TN_COASTAL_BOUNDS["max_lon"]))
            
            return (round(new_lat, 4), round(new_lon, 4))
        except:
            return self.location

# ===========================
# 🚨 Safety & Alerts System
# ===========================
def display_safety_alerts(weather, ocean_data):
    """Show weather and ocean condition warnings"""
    alerts = []
    
    if weather["wind_speed"] > 15:
        alerts.append("⚠️ High Wind Warning: Avoid offshore activities")
    if ocean_data["wave_height"] > 2.5:
        alerts.append("🌊 Rough Sea Conditions: Exercise caution")
    if weather["temp"] > 35:
        alerts.append("🔥 Heat Warning: Limit daytime operations")
        
    for alert in alerts:
        st.error(alert)

# ===========================
# 📈 Historical Analysis
# ===========================
def plot_catch_trends():
    """Visualize historical fishing data"""
    try:
        df = pd.read_csv("historical_catches.csv")
    except:
        # Sample data for demonstration
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=12, freq='M'),
            'catch': np.random.randint(50, 200, size=12)
        })
    
    fig = px.line(df, x='date', y='catch', 
                 title="Monthly Catch Trends",
                 labels={'catch': 'Total Catch (kg)', 'date': 'Date'})
    st.plotly_chart(fig)

# ===========================
# 🌐 Multilingual Support
# ===========================
def get_translation(lang):
    """Return UI translations"""
    translations = {
        "English": {
            "title": "Tamil Nadu Coastal Fisheries Management 🐟",
            "weather": "Coastal Environment",
            "species": "Local Species Info"
        },
        "தமிழ்": {
            "title": "தமிழ்நாடு கடலோர மீன்பிடி மேலாண்மை 🐟",
            "weather": "கடலோர சூழல்",
            "species": "உள்ளூர் மீன் இனங்கள்"
        }
    }
    return translations.get(lang, translations["English"])

# ===========================
# 🚀 Streamlit Interface
# ===========================
def main():
    # Initialize system components
    fs = EnhancedFisheriesSystem()
    advisor = TamilNaduFishingAdvisor(fs.weather_data, fs.ocean_data, fs.current_location)
    
    # Language selection
    lang = st.sidebar.radio("भाषा/Language", ["English", "தமிழ்"], index=0)
    trans = get_translation(lang)
    
    st.title(trans["title"])
    
    # Safety alerts
    display_safety_alerts(fs.weather_data, fs.ocean_data)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Coastal navigation map
        st.subheader("Coastal Navigation Map")
        folium_static(create_coastal_map(fs.current_location), width=800)
        
        # Fish detection system
        st.subheader("Fish Detection System")
        uploaded_file = st.file_uploader("Upload Fishing Image", type=["jpg", "png", "jpeg"])
        if uploaded_file and fs.model:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Coastal Fishing Image", use_container_width=True)
                
                results = fs.model(np.array(img))
                
                if len(results[0]) > 0:
                    st.subheader("Detection Results")
                    detections = pd.DataFrame(
                        results[0].boxes.data.cpu().numpy(),
                        columns=['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class']
                    )
                    detections['species'] = detections['class'].apply(lambda x: results[0].names[int(x)])
                    
                    for _, row in detections.iterrows():
                        with st.expander(f"{row['species'].capitalize()} Details"):
                            st.markdown(f"""
                            **Confidence**: {row['confidence']:.1%}  
                            **Size**: {(row['xmax']-row['xmin']):.0f}px × {(row['ymax']-row['ymin']):.0f}px
                            """)
                            display_species_info(row['species'])
                else:
                    st.warning("No marine species detected")
                    
            except Exception as e:
                st.error(f"Image Processing Error: {str(e)}")
                
        # Historical analysis
        with st.expander("Historical Catch Analysis"):
            plot_catch_trends()
    
    with col2:
        # Environmental data
        st.subheader(trans["weather"])
        cols = st.columns(2)
        cols[0].metric("Water Temp", f"{fs.ocean_data['water_temp']:.1f}°C")
        cols[1].metric("Wave Height", f"{fs.ocean_data['wave_height']:.1f}m")
        st.caption(f"Wind Speed: {fs.weather_data['wind_speed']} m/s")
        
        # Fishing status
        st.subheader("Fishing Status")
        if fs.fishing_status == "active":
            st.error("🚫 Fishing Ban Enforced (Jun 15-Aug 15)")
        else:
            st.success("✅ Fishing Permitted")
        
        # Trip planner
        st.subheader("Trip Planner")
        fuel_type = st.selectbox("Fuel Type", ["Diesel"], index=0)
        consumption = st.number_input("Fuel Rate (liters/hour)", 
                                    min_value=1.0, max_value=50.0, value=5.0)
        hours = st.number_input("Duration (hours)", 
                              min_value=1, max_value=24, value=8)
        
        emissions = fs.calculate_emissions(fuel_type.lower(), consumption, hours)
        st.metric("CO₂ Emissions", f"{emissions} kg")
        
        # AI recommendations
        st.subheader("AI Recommendations")
        show_ai_recommendations(fs, advisor)
        
        # Species information
        st.sidebar.subheader(trans["species"])
        species = st.sidebar.selectbox("Select Species", 
                                      list(LOCAL_SPECIES.keys()))
        display_species_info(species)

# ===========================
# 🐟 Species Information
# ===========================
def display_species_info(species_name):
    """Show ecological information from database"""
    info = LOCAL_SPECIES.get(species_name, {})
    if info:
        with st.expander(f"About {species_name}"):
            st.markdown(f"""
            **Conservation Status**: {info.get('status', 'Unknown')}  
            **Fishing Regulations**: {info.get('regulations', 'None')}
            """)
            try:
                st.image(f"images/{info['image']}")
            except:
                pass

# ===========================
# 🤖 AI Recommendation Engine
# ===========================
def show_ai_recommendations(system, advisor):
    """Generate smart fishing recommendations"""
    recs = []
    
    if system.weather_data["temp"] > 28:
        recs.append("🎣 Target surface-feeding species")
    
    if system.fishing_status == "active":
        recs.append("⛔ Ban Alert: Focus on equipment maintenance")
    else:
        recs.append(f"📍 Suggested Location: {advisor.predict_movement()}")
    
    if system.ocean_data["water_temp"] < 22:
        recs.append("❄️ Cold Water: Try deeper waters")
    
    for rec in recs:
        st.markdown(f"- {rec}")

# ===========================
# 📱 Mobile Optimization
# ===========================
st.markdown("""
<style>
@media screen and (max-width: 768px) {
    .stNumberInput, .stSelectbox, .stRadio { 
        width: 100% !important; 
    }
    .stButton>button {
        width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()