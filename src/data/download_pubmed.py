from pathlib import Path
from datasets import load_dataset

# Configuration
HF_DATASET_NAME = 'armanc/pubmed-rct20k'
DATA_ROOT = Path('data')
DF_PATHS = {
    'train': DATA_ROOT / 'train.parquet',
    'test': DATA_ROOT / 'test.parquet',
    'validation': DATA_ROOT / 'validation.parquet'
}


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    if all(path.exists() for path in DF_PATHS.values()):
        print("Los archivos Parquet ya existen. Omitiendo descarga.")
        return

    print("Descargando el dataset desde Hugging Face...")
    dataset = load_dataset(HF_DATASET_NAME, token=True)

    for split, path in DF_PATHS.items():
        print(f"Guardando el split '{split}' en {path}...")
        dataset[split].to_parquet(path)


if __name__ == '__main__':
    main()
