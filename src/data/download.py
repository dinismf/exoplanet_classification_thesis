import os
import pandas as pd
import aiohttp
import asyncio
from bs4 import BeautifulSoup
import logging
from tqdm import tqdm

# Base URL for Kepler light curves
BASE_URL = 'http://archive.stsci.edu/pub/kepler/lightcurves/'

# Setup logging to output to a file
logging.basicConfig(
    filename='downloads.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set a default timeout for HTTP requests (e.g., 10 seconds)
TIMEOUT = aiohttp.ClientTimeout(total=10)

# Limit the number of concurrent tasks with a semaphore
MAX_CONCURRENT_REQUESTS = 100
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Batch size for processing kepids
BATCH_SIZE = 500


# ---------------- Utility Functions ----------------

def load_kepids_from_csv(file_path):
    """
    Load and return the kepids from a CSV file.
    Kepids are zero-padded to 9 digits.
    """
    df = pd.read_csv(file_path)
    kepids = df['kepid'].astype(str).str.zfill(9)
    return kepids


def build_kepid_url_and_folder(kepid, base_download_dir):
    """
    Construct the URL and download folder for the given kepid.
    The folder structure mirrors the URL structure.
    """
    kepid_folder = kepid[:4]  # First 4 digits of the kepid
    full_url = f"{BASE_URL}{kepid_folder}/{kepid}/"
    download_folder = os.path.join(base_download_dir, kepid_folder, kepid)
    os.makedirs(download_folder, exist_ok=True)  # Create folder if not exists
    return full_url, download_folder


def parse_fits_urls(html_content, base_url):
    """
    Parse and return all .fits file URLs from the HTML content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    fits_urls = [
        os.path.join(base_url, link.get('href'))
        for link in soup.find_all('a')
        if link.get('href') and link.get('href').endswith('_llc.fits')
    ]
    return fits_urls


# ---------------- Asynchronous Functions ----------------

async def fetch_html(session, url):
    """
    Asynchronously fetch the HTML content from the provided URL,
    while disabling SSL verification and applying a timeout.
    """
    try:
        async with semaphore:
            async with session.get(url, ssl=False, timeout=TIMEOUT) as response:
                response.raise_for_status()
                return await response.text()
    except asyncio.TimeoutError:
        logging.error(f"Timeout while fetching {url}")
        return None
    except aiohttp.ClientError as e:
        logging.error(f"Failed to fetch {url}: {e}")
        return None


async def download_file(session, url, download_folder):
    """
    Asynchronously download a single .fits file to the specified folder,
    while disabling SSL verification and applying a timeout.
    """
    filename = os.path.join(download_folder, os.path.basename(url))
    try:
        async with semaphore:
            async with session.get(url, ssl=False, timeout=TIMEOUT) as response:
                response.raise_for_status()
                with open(filename, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
        logging.info(f"Downloaded: {filename}")
        return True
    except asyncio.TimeoutError:
        logging.error(f"Timeout while downloading {url}")
        return False
    except aiohttp.ClientError as e:
        logging.error(f"Failed to download {url}: {e}")
        return False


async def process_kepid(session, kepid, base_download_dir):
    """
    Process a single kepid:
    - Fetch the .fits URLs.
    - Download the corresponding .fits files.
    """
    full_url, download_folder = build_kepid_url_and_folder(kepid, base_download_dir)

    # Fetch HTML and parse for .fits files
    html_content = await fetch_html(session, full_url)
    if not html_content:
        return 0  # No HTML content, no downloads

    fits_urls = parse_fits_urls(html_content, full_url)
    if not fits_urls:
        return 0  # No .fits files to download

    # Asynchronously download all .fits files
    download_tasks = [
        download_file(session, fits_url, download_folder)
        for fits_url in fits_urls
    ]

    results = await asyncio.gather(*download_tasks)
    return sum(results)  # Count of successful downloads


async def process_kepid_batch(session, kepid_batch, base_download_dir):
    """
    Process a batch of kepids concurrently.
    """
    tasks = [
        process_kepid(session, kepid, base_download_dir)
        for kepid in kepid_batch
    ]
    await asyncio.gather(*tasks)


# ---------------- Batching Logic ----------------

async def download_all_kepids(csv_file, base_download_dir):
    """
    Main function to manage and process all kepids:
    - Fetch URLs for each kepid.
    - Download the corresponding .fits files in batches.
    """
    kepids = load_kepids_from_csv(csv_file)

    async with aiohttp.ClientSession() as session:
        # Process kepids in batches
        for i in tqdm(range(0, len(kepids), BATCH_SIZE), desc="Processing Batches"):
            kepid_batch = kepids[i:i + BATCH_SIZE]
            await process_kepid_batch(session, kepid_batch, base_download_dir)


if __name__ == "__main__":
    csv_file = "../../data/raw/q1_q17_dr24_tce.csv"
    base_download_dir = "../../data/raw/fits"

    # Run the event loop to process all kepids
    asyncio.run(download_all_kepids(csv_file, base_download_dir))
