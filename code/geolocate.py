import requests
import math
from itertools import combinations
import pandas as pd

API_KEY = "e82d48d0750049b087fa7175089fcb53"
GEOCODE_URL = "https://api.opencagedata.com/geocode/v1/json"

bike_stations = [
    "PREŠERNOV TRG-PETKOVŠKOVO NABREŽJE",
    "POGAČARJEV TRG-TRŽNICA",
    "KONGRESNI TRG-ŠUBIČEVA ULICA",
    "CANKARJEVA UL.-NAMA",
    "BREG",
    "GRUDNOVO NABREŽJE-KARLOVŠKA C.",
    "MIKLOŠIČEV PARK",
    "BAVARSKI DVOR",
    "TRG OF-KOLODVORSKA UL.",
    "MASARYKOVA DDC",
    "VILHARJEVA CESTA",
    "PARK NAVJE-ŽELEZNA CESTA",
    "TRG MDB",
    "PARKIRIŠČE NUK 2-FF",
    "AMBROŽEV TRG",
    "GH ŠENTPETER-NJEGOŠEVA C.",
    "ILIRSKA ULICA",
    "TRŽAŠKA C.-ILIRIJA",
    "TIVOLI",
    "STARA CERKEV",
    "KINO ŠIŠKA",
    "ŠPICA",
    "BARJANSKA C.-CENTER STAREJŠIH TRNOVO",
    "ZALOŠKA C.-GRABLOVIČEVA C.",
    "TRŽNICA MOSTE",
    "ROŽNA DOLINA-ŠKRABČEVA UL.",
    "DUNAJSKA C.-PS PETROL",
    "PLEČNIKOV STADION",
    "DUNAJSKA C.-PS MERCATOR",
    "LIDL - VOJKOVA CESTA",
    "ŠPORTNI CENTER STOŽICE",
    "KOPRSKA ULICA",
    "MERCATOR CENTER ŠIŠKA",
    "CITYPARK",
    "BTC CITY/DVORANA A",
    "BTC CITY ATLANTIS",
    "TRNOVO",
    "P+R BARJE",
    "P + R DOLGI MOST",
    "BONIFACIJA",
    "ANTONOV TRG",
    "BRATOVŠEVA PLOŠČAD",
    "BS4-STOŽICE",
    "SAVSKO NASELJE 2-LINHARTOVA CESTA",
    "SAVSKO NASELJE 1-ŠMARTINSKA CESTA",
    "SITULA",
    "ŠTEPANJSKO NASELJE 1-JAKČEVA ULICA",
    "HOFER-KAJUHOVA",
    "BRODARJEV TRG",
    "PREGLOV TRG",
    "LIDL-LITIJSKA CESTA",
    "ŽIVALSKI VRT",
    "CESTA NA ROŽNIK",
    "ŠMARTINSKI PARK",
    "POLJANSKA-POTOČNIKOVA",
    "SREDNJA FRIZERSKA ŠOLA",
    "POVŠETOVA-GRABLOVIČEVA",
    "TRŽNICA KOSEZE",
    "LIDL BEŽIGRAD",
    "MERCATOR MARKET - CELOVŠKA C. 163",
    "RAKOVNIK",
    "ALEJA - CELOVŠKA CESTA",
    "IKEA",
    "KOPALIŠČE KOLEZIJA",
    "VIŠKO POLJE",
    "KOSEŠKI BAJER",
    "DRAVLJE",
    "ČRNUČE",
    "STUDENEC",
    "POLJE",
    "ZALOG",
    "LIDL - RUDNIK",
    "PRUŠNIKOVA",
    "POVŠETOVA - KAJUHOVA",
    "SOSESKA NOVO BRDO",
    "TEHNOLOŠKI PARK",
    "VOJKOVA - GASILSKA BRIGADA",
    #"GERBIČEVA - ŠPORTNI PARK SVOBODA",
    "GERBIČEVA ULICA",
    "DOLENJSKA C. - STRELIŠČE",
    "ROŠKA - STRELIŠKA",
    "LEK - VEROVŠKOVA",
    "VOKA - SLOVENČEVA",
    "SUPERNOVA LJUBLJANA - RUDNIK",
]


def geocode(location: str) -> tuple:
    params = {
        "q": f"{location}, Ljubljana, Slovenia",
        "key": API_KEY,
        "language": "en",
        "limit": 1,
    }
    response = requests.get(GEOCODE_URL, params=params)
    data = response.json()
    print("Geocoding", location)
    try:
        lat = data["results"][0]["geometry"]["lat"]
        lng = data["results"][0]["geometry"]["lng"]
    except IndexError:
        lng = 0
        lat = 0
        print("ohono!!!", location)

    return lat, lng


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.sin(dLon / 2) * math.sin(
        dLon / 2
    ) * math.cos(lat1) * math.cos(lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


coordinates = {station: geocode(station) for station in bike_stations}

# Initialize an empty DataFrame with bike stations as row and column names
distance_df = pd.DataFrame(columns=bike_stations, index=bike_stations)

for station1, station2 in combinations(coordinates.keys(), 2):
    coord1 = coordinates[station1]
    coord2 = coordinates[station2]
    distance = haversine(coord1[0], coord1[1], coord2[0], coord2[1])

    # Fill the DataFrame with calculated distances
    distance_df.loc[station1, station2] = distance
    distance_df.loc[station2, station1] = distance

# Fill the diagonal with 0's
for station in bike_stations:
    distance_df.loc[station, station] = 0

# Replace "GERBIČEVA ULICA" column name and row name with "GERBIČEVA - ŠPORTNI PARK SVOBODA",

distance_df = distance_df.rename(
    columns={"GERBIČEVA ULICA": "GERBIČEVA - ŠPORTNI PARK SVOBODA"}
)
distance_df = distance_df.rename(
    index={"GERBIČEVA ULICA": "GERBIČEVA - ŠPORTNI PARK SVOBODA"}
)

# Save the DataFrame to a CSV file
distance_df.to_csv("../data/distance_matrix.csv")
