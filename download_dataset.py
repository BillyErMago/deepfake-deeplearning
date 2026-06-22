import os
import argparse
import pandas as pd
import requests
from tqdm import tqdm

def download_image(img_id, img_url, output_dir):
    img_path = os.path.join(output_dir, f"{img_id}.jpg")
    
    # Skip if already downloaded
    if os.path.exists(img_path):
        return True
        
    try:
        response = requests.get(img_url, timeout=10)
        response.raise_for_status()
        with open(img_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"\n[WARNING] Impossibile scaricare l'immagine {img_id} da {img_url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Scarica le immagini del dataset Deepfake")
    parser.add_argument("--csv_file", type=str, default="./FINAL_DATASET.csv", help="Percorso del file CSV")
    parser.add_argument("--output_dir", type=str, default="./data/images", help="Cartella di destinazione")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"[ERRORE] Il file CSV {args.csv_file} non esiste. Crealo o passalo tramite --csv_file.")
        return

    print(f"[INFO] Creazione cartella di output: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[INFO] Lettura file {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    
    if 'image_id' not in df.columns or 'image_url' not in df.columns:
        print("[ERRORE] Il file CSV deve contenere le colonne 'image_id' e 'image_url'.")
        return

    print(f"[INFO] Inizio download di {len(df)} immagini...")
    
    success_count = 0
    fail_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scaricamento"):
        img_id = str(row['image_id'])
        img_url = row['image_url']
        
        success = download_image(img_id, img_url, args.output_dir)
        if success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n[INFO] Download completato!")
    print(f"       - Immagini scaricate: {success_count}")
    print(f"       - Fallimenti (saltate): {fail_count}")

if __name__ == "__main__":
    main()
