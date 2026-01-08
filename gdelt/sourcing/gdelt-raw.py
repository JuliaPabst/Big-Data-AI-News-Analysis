import boto3
import requests
import gzip
import io
from datetime import datetime
from botocore.exceptions import NoCredentialsError
from dataclasses import dataclass

BUCKET_NAME = "bigdata-mapping-ai-project"
REGION = "eu-central-1"
BASE_PATH = "./data-sourcing/data/"

s3 = boto3.client("s3", region_name=REGION)

@dataclass
class URLStruct:
   url: str
   filename: str

# ==========================================================

from datetime import datetime, timedelta

def get_source_urls(year):
    urls = []

    current = datetime(year, 1, 1, 16, 1, 0)
    end = datetime(year + 1, 1, 1, 16, 1, 0)

    while current < end:
        timestamp = current.strftime("%Y%m%d%H%M%S")
        urls.append(URLStruct(
           url = f"http://data.gdeltproject.org/gdeltv3/gal/{timestamp}.gal.json.gz",
           filename = f"gdelt-articles/{timestamp}.json"
        ))
        current += timedelta(days=1)

    return urls

# ==========================================================

def fetch_and_upload_data(urls):
  print(f"Fetching data...")

  for url in urls:
    response = requests.get(url.url)

    if response.status_code != 200:
      print(f"Error {response.status_code}: Url {url} found nothing")
      continue

    gz_buffer = io.BytesIO(response.content)
    with gzip.GzipFile(fileobj=gz_buffer) as gz:
      csv_bytes = gz.read()

      upload_to_S3(csv_bytes, url.filename)
  
# ==========================================================

def upload_to_S3(data, filename):      
  fileobj = io.BytesIO(data)

  try:
    s3.upload_fileobj(fileobj, BUCKET_NAME, filename)
    print(f"Success {filename}")
  except FileNotFoundError:
    print(f"Could not find local file {filename}.")
    return
  except NoCredentialsError:
    print(f"No AWS credentials found. Run 'aws configure'. {filename}")
    return
  except Exception as e:
    print(f"Error: {e}; {filename}")
    return

# ==========================================================

fetch_and_upload_data(get_source_urls(2024))