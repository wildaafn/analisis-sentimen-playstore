"""
scraping.py
===========
Script scraping ulasan aplikasi dari Google Play Store.
Target: ≥10.000 ulasan dari beberapa aplikasi populer Indonesia.

Cara menjalankan:
    pip install google-play-scraper pandas
    python scraping.py
"""

import time
import pandas as pd
from google_play_scraper import reviews, Sort
from google_play_scraper.exceptions import NotFoundError

# ── Konfigurasi Aplikasi ──────────────────────────────────────────────────────
APP_CONFIG = [
    {"app_id": "com.gojek.app",        "app_name": "Gojek"},
    {"app_id": "id.co.tokopedia",      "app_name": "Tokopedia"},
    {"app_id": "com.shopee.id",        "app_name": "Shopee"},
    {"app_id": "com.traveloka.android","app_name": "Traveloka"},
    {"app_id": "com.bukalapak.android","app_name": "Bukalapak"},
]

TARGET_PER_APP = 2500
LANG           = "id"
COUNTRY        = "id"
OUTPUT_CSV     = "dataset_ulasan.csv"
OUTPUT_JSON    = "dataset_ulasan.json"


def scrape_app(app_id: str, app_name: str, count: int) -> list:
    """Scrape ulasan dari satu aplikasi Play Store."""
    all_reviews = []
    continuation_token = None
    batch_size = 200

    print(f"\n[INFO] Mulai scraping '{app_name}' ...")

    while len(all_reviews) < count:
        try:
            remaining   = count - len(all_reviews)
            fetch_count = min(batch_size, remaining)

            result, continuation_token = reviews(
                app_id,
                lang=LANG,
                country=COUNTRY,
                sort=Sort.NEWEST,
                count=fetch_count,
                continuation_token=continuation_token,
            )

            if not result:
                print(f"  [WARN] Tidak ada data lagi untuk '{app_name}'.")
                break

            for r in result:
                all_reviews.append({
                    "app_name" : app_name,
                    "app_id"   : app_id,
                    "review_id": r.get("reviewId", ""),
                    "username" : r.get("userName", ""),
                    "review"   : r.get("content", ""),
                    "rating"   : r.get("score", 0),
                    "thumbs_up": r.get("thumbsUpCount", 0),
                    "at"       : str(r.get("at", "")),
                })

            print(f"  [INFO] '{app_name}': {len(all_reviews)}/{count} terkumpul")

            if continuation_token is None:
                break

            time.sleep(1.5)

        except NotFoundError:
            print(f"  [ERROR] App '{app_id}' tidak ditemukan.")
            break
        except Exception as e:
            print(f"  [ERROR] {e}. Coba ulang dalam 5 detik ...")
            time.sleep(5)

    print(f"  [DONE] '{app_name}': {len(all_reviews)} ulasan.")
    return all_reviews


def main():
    all_data = []

    for cfg in APP_CONFIG:
        data = scrape_app(cfg["app_id"], cfg["app_name"], TARGET_PER_APP)
        all_data.extend(data)
        time.sleep(3)

    df = pd.DataFrame(all_data)

    # Deduplikasi
    before = len(df)
    df.drop_duplicates(subset=["review_id"], inplace=True)
    df.drop_duplicates(subset=["review"],    inplace=True)
    df.dropna(subset=["review"],             inplace=True)
    df = df[df["review"].str.strip() != ""]
    df.reset_index(drop=True, inplace=True)
    after = len(df)

    print(f"\n[INFO] Total setelah deduplikasi: {after} (dari {before})")

    df.to_csv(OUTPUT_CSV,  index=False, encoding="utf-8")
    df.to_json(OUTPUT_JSON, orient="records", force_ascii=False, indent=2)

    print(f"[INFO] Disimpan ke '{OUTPUT_CSV}' dan '{OUTPUT_JSON}'.")
    print(f"[INFO] Jumlah data akhir: {len(df)} baris.")
    print(f"\nDistribusi rating:\n{df['rating'].value_counts().sort_index()}")
    print(f"\nDistribusi aplikasi:\n{df['app_name'].value_counts()}")


if __name__ == "__main__":
    main()
