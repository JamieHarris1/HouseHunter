from pathlib import Path
import urllib.request

# Get the folder this Python file is in
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

# CSV
CSV_PATH = DATA_DIR / "pp-complete.csv"

# Download URL
CSV_URL = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-complete.csv"

# Make sure Data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


# Download CSV if it doesn't already exist
if not CSV_PATH.exists():
    print("Downloading Price Paid Data...")
    print("This is around 5 GB, so it may take a while.")

    urllib.request.urlretrieve(CSV_URL, CSV_PATH)

    print("Download complete.")
else:
    print("CSV already exists. Skipping download.")

