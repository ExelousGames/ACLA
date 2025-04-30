import requests

base_url = "http://localhost:7001"

def get_pokemon_info(name):
    url = f"{base_url}/userinfo"
    json = {'infoDto': { 'name' : name }}

    try:
        response = requests.post(url,json)
        if response.status_code == 200:
            pokemon_data = response.json()
            return pokemon_data
    except response:
        print(f"Failed to retrieve data {response}")

