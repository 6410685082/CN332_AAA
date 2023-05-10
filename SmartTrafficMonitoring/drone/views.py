from django.shortcuts import render
import requests
import folium
import requests
from .models import Weather

def map(request):
    return render(request,'drone/map.html')

def CheckDrone(request,task_id, lat = 14.068542, lon = 100.605965):
    latitude = lat
    longitude = lon
    m = folium.Map(location=[latitude, longitude], zoom_start=13)
    folium.Marker(
        location=[latitude, longitude],
        popup='My Location',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    m.save('drone/templates/drone/map.html')

    # Replace YOUR_API_KEY with your actual API key
    api_key = '9b67ff866b4d733e828c350064236922'

    # Replace CITY_NAME with the city you want to check the weather for
    city = 'Bangkok'

    # Send a GET request to the OpenWeatherMap API
    response = requests.get(f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}')

    # Parse the JSON response
    data = response.json()

    # Fetch weather data from OpenWeatherMap API
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    print(data)

    # Extract weather information
    temp = data["main"]["temp"]
    conditions = data["weather"][0]["description"]
    icon = data["weather"][0]["icon"]
    wind_speed = data["wind"]["speed"]
    clouds = data["clouds"]["all"]
    humidity = data['main']['humidity']
    description = data['weather'][0]['description']

    weather = Weather.objects.get(task = task_id) 
    weather.location = "latitude = " + str(latitude) + " , longitude = " + str(longitude)
    weather.wind_speed = wind_speed
    weather.latitude = latitude 
    weather.longitude = longitude
    weather.temp = temp
    weather.weather_report = description
    weather.humidity = humidity 
    weather.clouds = clouds
    weather.save()
