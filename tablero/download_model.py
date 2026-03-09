"""
MAIA · PubMed Microproyecto 1
──────────────────────────────────────────────────────────────────────────────
Script de preparación del entorno: descarga la carpeta completa del modelo
SciBERT desde Google Drive y valida la integridad de cada archivo con MD5.

Ejecutar UNA VEZ después de clonar el repositorio y antes del build Docker.

Archivos descargados (carpeta scibert_pubmed en Drive):
    model.safetensors      419 MB  — pesos del modelo fine-tuned
    tokenizer.json         696 KB  — vocabulario SciBERT
    tokenizer_config.json    1 KB  — configuración del tokenizador
    config.json              1 KB  — arquitectura del modelo
    label_meta.json          1 KB  — mapeo label2id / id2label
    training_args.bin        5 KB  — argumentos de entrenamiento

Uso:
    python download_model.py

Cómo obtener el FOLDER_ID de Google Drive:
    1. Subir TODOS los archivos del modelo a UNA carpeta en Google Drive
    2. Clic derecho sobre la CARPETA → Compartir → "Cualquier persona con el enlace"
    3. Copiar el enlace. Tendrá esta forma:
       https://drive.google.com/drive/folders/1aBcDeFgHiJkLmNoPqRsTuVwXyZ
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                               Este es el FOLDER_ID
    4. Reemplazar GDRIVE_FOLDER_ID_DEFAULT con ese valor

Cómo obtener los hash MD5 de tus archivos originales:
    Linux/macOS:  md5sum backend/app/model/scibert_pubmed/*
    Windows:      Get-FileHash backend\app\model\scibert_pubmed\* -Algorithm MD5
    Luego reemplazar los valores "PENDIENTE" en MD5_HASHES con los resultados.
"""

import hashlib
import os
import sys
from pathlib import Path

# ── Configuración ─────────────────────────────────────────────────────────
# Reemplazar con el FOLDER_ID real después de crear la carpeta en Google Drive
GDRIVE_FOLDER_ID_DEFAULT = "REEMPLAZAR_CON_FOLDER_ID_DE_GOOGLE_DRIVE"

# Destino local — directorio donde se guardan todos los archivos
DEST_DIR = (
    Path(__file__).resolve().parent
    / "backend" / "app" / "model" / "scibert_pubmed"
)

# ── Hash MD5 de cada archivo ───────────────────────────────────────────────
# Reemplazar "PENDIENTE" con el hash MD5 real de cada archivo original.
# Obtener con:
#   Linux/macOS : md5sum backend/app/model/scibert_pubmed/*
#   Windows PS  : Get-FileHash backend\app\model\scibert_pubmed\* -Algorithm MD5
#
# Si un hash es "PENDIENTE", ese archivo se valida solo por tamaño (fallback).
MD5_HASHES = {
    "model.safetensors":    "PENDIENTE",
    "tokenizer.json":       "PENDIENTE",
    "tokenizer_config.json":"PENDIENTE",
    "config.json":          "PENDIENTE",
    "label_meta.json":      "PENDIENTE",
    "training_args.bin":    "PENDIENTE",
}

# Tamaños mínimos en bytes (fallback cuando el hash es "PENDIENTE")
MIN_SIZES = {
    "model.safetensors":    400_000_000,
    "tokenizer.json":           500_000,
    "tokenizer_config.json":        100,
    "config.json":                  100,
    "label_meta.json":              100,
    "training_args.bin":          1_000,
}

# ─────────────────────────────────────────────────────────────────────────
def _md5(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """Calcula el MD5 de un archivo en bloques para no saturar RAM."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _validate_file(filename: str) -> tuple[bool, str]:
    """
    Valida un archivo descargado.
    Si el hash MD5 está configurado → compara MD5 (verificación exacta).
    Si el hash es 'PENDIENTE'       → compara solo tamaño mínimo (fallback).
    Devuelve (ok: bool, mensaje: str).
    """
    dest = DEST_DIR / filename
    expected_md5 = MD5_HASHES.get(filename, "PENDIENTE")

    if not dest.exists():
        return False, "archivo no encontrado"

    size = dest.stat().st_size
    size_mb = size / 1_000_000

    if expected_md5 != "PENDIENTE":
        # Verificación MD5 exacta
        print(f"    calculando MD5...", end=" ", flush=True)
        actual_md5 = _md5(dest)
        if actual_md5 == expected_md5:
            return True, f"{size_mb:.1f} MB  md5 ✓"
        else:
            return False, (
                f"MD5 no coincide\n"
                f"      esperado : {expected_md5}\n"
                f"      obtenido : {actual_md5}"
            )
    else:
        # Fallback: validación por tamaño mínimo
        min_size = MIN_SIZES.get(filename, 1)
        if size >= min_size:
            return True, f"{size_mb:.1f} MB  (hash PENDIENTE — validado por tamaño)"
        else:
            return False, f"tamaño insuficiente ({size:,} bytes, mínimo: {min_size:,})"


def _all_valid() -> bool:
    """True si todos los archivos pasan validación."""
    for filename in MD5_HASHES:
        ok, _ = _validate_file(filename)
        if not ok:
            return False
    return True


def main():
    folder_id = os.environ.get("GDRIVE_FOLDER_ID", GDRIVE_FOLDER_ID_DEFAULT)

    # 1. Validar que el FOLDER_ID fue configurado
    if folder_id == GDRIVE_FOLDER_ID_DEFAULT:
        print("ERROR: El FOLDER_ID de Google Drive no está configurado.")
        print()
        print("  Editar download_model.py y reemplazar GDRIVE_FOLDER_ID_DEFAULT,")
        print("  o definir la variable de entorno:")
        print("    export GDRIVE_FOLDER_ID=1aBcDeFgHiJkLmNoPqRsTuVwXyZ  # Linux/macOS")
        print('    $env:GDRIVE_FOLDER_ID="1aBcDeFgHiJkLmNoPqRsTuVwXyZ"  # PowerShell')
        print()
        print("  Cómo obtener el FOLDER_ID:")
        print("  1. Subir todos los archivos del modelo a una carpeta en Google Drive")
        print("  2. Clic derecho sobre la CARPETA → Compartir → 'Cualquier persona con el enlace'")
        print("  3. El enlace tiene la forma:")
        print("     https://drive.google.com/drive/folders/<FOLDER_ID>")
        sys.exit(1)

    # 2. Verificar si todos los archivos ya existen y son válidos
    if _all_valid():
        print("✓ Todos los archivos del modelo ya existen y son válidos — se omite la descarga.")
        print()
        for filename in MD5_HASHES:
            ok, msg = _validate_file(filename)
            print(f"  ✓ {filename:<30} {msg}")
        print()
        print("Para forzar re-descarga, elimine los archivos y vuelva a ejecutar.")
        sys.exit(0)

    # 3. Verificar / instalar gdown >= 4.6
    try:
        import gdown
        from packaging.version import Version
        if Version(gdown.__version__) < Version("4.6.0"):
            raise ImportError("versión antigua")
    except Exception:
        print("Instalando/actualizando gdown...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "gdown", "-q"],
            capture_output=True,
        )
        if result.returncode != 0:
            print("ERROR: No se pudo instalar gdown.")
            print("  Instalar manualmente con: pip install --upgrade gdown")
            sys.exit(1)
        import gdown
        print("✓ gdown listo.\n")

    # 4. Crear directorio destino
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Descargar carpeta completa desde Google Drive
    print("Descargando modelo SciBERT completo desde Google Drive...")
    print(f"  FOLDER_ID : {folder_id}")
    print(f"  Destino   : {DEST_DIR}")
    print()

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        gdown.download_folder(
            url=url,
            output=str(DEST_DIR),
            quiet=False,
            use_cookies=False,
            remaining_ok=True,
        )
    except Exception as e:
        print(f"\nERROR durante la descarga: {e}")
        sys.exit(1)

    # 6. Validar integridad de cada archivo
    print("\nValidando integridad de archivos descargados...")
    errors = []
    for filename in MD5_HASHES:
        print(f"  {filename:<30}", end=" ", flush=True)
        ok, msg = _validate_file(filename)
        if ok:
            print(f"✓  {msg}")
        else:
            print(f"✗  {msg}")
            errors.append(filename)

    if errors:
        print(f"\nERROR: {len(errors)} archivo(s) no pasaron la validación:")
        for f in errors:
            print(f"  - {f}")
        print()
        print("  Causas comunes:")
        print("  - Descarga incompleta (reintentar: python download_model.py)")
        print("  - Hash MD5 incorrecto en el script (verificar con md5sum / Get-FileHash)")
        print("  - La CARPETA de Drive no es pública")
        print("  - gdown desactualizado: pip install --upgrade gdown")
        # Eliminar archivos corruptos para forzar re-descarga en siguiente ejecución
        for f in errors:
            p = DEST_DIR / f
            if p.exists():
                p.unlink()
                print(f"  (eliminado {f} para re-descarga)")
        sys.exit(1)

    print(f"\n✓ Modelo completo e íntegro en: {DEST_DIR}")
    print()
    print("Ahora puede construir el contenedor:")
    print("  docker compose build")
    print("  docker compose up -d")


if __name__ == "__main__":
    main()
