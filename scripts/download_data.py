from pathlib import Path
from urllib.request import urlretrieve
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Link currently exposed by the CFPB August 2026 archive.
DOWNLOAD_URL = (
    "https://files.consumerfinance.gov/f/documents/"
    "CCDB_Export_21_August_2026.zip"
)

ZIP_PATH = RAW_DIR / "cfpb_august_2026.zip"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading CFPB complaint narrative archive...")
    print(DOWNLOAD_URL)

    try:
        urlretrieve(DOWNLOAD_URL, ZIP_PATH)
    except Exception as exc:
        print("\nAutomatic download failed.")
        print(f"Reason: {exc}")
        print(
            "\nManual fallback:\n"
            "1. Open the CFPB Consumer Complaint Database Narratives Archive.\n"
            "2. Download the August 2026 CCDB Export.\n"
            "3. Put the ZIP or extracted CSV under data/raw/.\n"
        )
        return

    print(f"Downloaded: {ZIP_PATH}")

    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as archive:
            archive.extractall(RAW_DIR)

        print(f"Extracted files to: {RAW_DIR}")

    except zipfile.BadZipFile:
        print(
            "The downloaded file was not recognized as a ZIP archive. "
            "Please download the archive manually from the CFPB website."
        )


if __name__ == "__main__":
    main()
