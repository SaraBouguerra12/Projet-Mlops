import requests

payload = {
    'instances': [
        {
            'speed': 120,
            'weight': 1500,
            'engine_size': 1.6,
            'fuel': 7.2
        }
    ]
}

response = requests.post(
    'http://localhost:8000/predict',
    json=payload
)

print(response.status_code)
print(response.text)