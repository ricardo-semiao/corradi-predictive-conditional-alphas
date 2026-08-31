import sqlite3
import requests
import time
import os

class BadHeadersError(Exception):
    pass

# Parameters:
PATH_SAVE = "data/trades.db"
URL = (
    "https://www.dropbox.com/scl/fo/p99wtwn7c5wqamwb348fj/AAHB_eGp3u3KC-j0Kadfqnw/"
    "factor_neutralized_returns_DELIVERY/data/raw/trades.db"
    "?rlkey=hv7qnw3j2utr0pokpw25bs7kr&dl=1"
)

DATA_ETAG = "1780314867411941d" # Dropbox id for file version
DATA_SIZE = 61940215808 # 57GB
# Both hard coded from earlier request

CHUNK_SIZE = 8 * 1024 ** 2 # 8MB
MAX_RETRIES = 20


# Initializing session:
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7,es-ES;q=0.6,es;q=0.5",
    "Upgrade-Insecure-Requests": "1",
    "DNT": "1",
    "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}
# Could add `Connection: keep-alive` and `Keep-Alive: timeout=60, max=1000`,
# but probably won't do much
s = requests.Session()
s.headers.update(headers)


# Main loop:
data_got = 0
iteration = 0
while data_got < DATA_SIZE and iteration < MAX_RETRIES:
    iteration += 1
    time.sleep(1.25 ** (iteration + 5))
    print(
        f"{iteration}. Progress: {data_got / (1024**3):.2f} GB / "
        f"{DATA_SIZE / (1024**3):.2f} GB ({data_got / DATA_SIZE:.1%}) ..."
    )

    if os.path.exists(PATH_SAVE):
        data_got = os.path.getsize(PATH_SAVE)
    s.headers.update({"Range": f"bytes={data_got}-"})

    try:
        with s.get(URL, stream = True) as response:
            response.raise_for_status()

            etag = response.headers.get("Etag", "").replace('"', '')
            if etag and etag != DATA_ETAG:
                print(
                    f"Warning: Etag mismatch. Expected {DATA_ETAG}, got {etag}."
                    "Continuing download..."
                )

            content_type = response.headers.get("Content-Type", "").lower()
            if content_type == 'application/binary':
                raise BadHeadersError("Response is not a binary file.")

            content_length = int(response.headers.get("Content-Length", 0))
            if content_length and content_length != DATA_SIZE - data_got:
                raise BadHeadersError(
                    f"Content-Length mismatch. Expected {DATA_SIZE - data_got},"
                    f" got {content_length}."
                )

            with open(PATH_SAVE, "ab") as f:
                for chunk in response.iter_content(chunk_size = CHUNK_SIZE):
                    if chunk:  # Filter out keep-alive empty chunks
                        f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"\nNetwork drop encountered: {e}. Retrying connection.")
        continue
    except BadHeadersError as e:
        print(f"\n{e}. Retrying connection.")
        continue
    finally:
        s.close()

try:
    print("Download complete! Checking database integrity...")
    conn = sqlite3.connect(PATH_SAVE)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
except sqlite3.Error as e:
    print(f"SQLite database error: {e}")
