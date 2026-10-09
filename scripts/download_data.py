"""Download the selected Kaggle dataset without credentials (public endpoint)."""
from pathlib import Path
from urllib.request import urlopen, Request
import io, json, zipfile
ROOT=Path(__file__).resolve().parents[1]
BASE='https://www.kaggle.com/api/v1/datasets/'
REF='prasadnirmal/srilankan-second-vehiclecar-price-dataset'

def fetch(url):
    with urlopen(Request(url,headers={'User-Agent':'vehicle-price-coursework/1.0'}),timeout=90) as response:
        return response.read()

if __name__=='__main__':
    out=ROOT/'data/raw';out.mkdir(parents=True,exist_ok=True)
    if (out/'car_price_dataset.csv').exists():
        raise SystemExit('Original CSV already exists. Preserve it; download a new version into a separate project if needed.')
    metadata=json.loads(fetch(BASE+'view/'+REF))
    archive=zipfile.ZipFile(io.BytesIO(fetch(BASE+'download/'+REF)))
    (out/'car_price_dataset.csv').write_bytes(archive.read('car_price_dataset.csv'))
    (out/'kaggle_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Downloaded',REF,'version',metadata.get('currentVersionNumber'))
