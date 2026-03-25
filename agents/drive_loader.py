"""
drive_loader.py — Descarga Parquets desde Google Drive a memoria.

Uso:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from agents.drive_loader import load_parquet_from_drive

    creds = service_account.Credentials.from_service_account_file(KEY_FILE, scopes=SCOPES)
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    df = load_parquet_from_drive(service, "sellout_consolidado.parquet", FOLDER_ID)
"""

import io
import pandas as pd
from googleapiclient.http import MediaIoBaseDownload


def _buscar_file_id(service, file_name: str, folder_id: str):
    """Retorna el fileId del archivo en la carpeta, o None si no existe."""
    q = (
        f"name='{file_name}' "
        f"and '{folder_id}' in parents "
        f"and trashed=false"
    )
    resp = (
        service.files()
        .list(q=q, fields="files(id, name)", spaces="drive")
        .execute()
    )
    archivos = resp.get("files", [])
    return archivos[0]["id"] if archivos else None


def load_parquet_from_drive(service, file_name: str, folder_id: str) -> pd.DataFrame:
    """
    Descarga un Parquet de Google Drive a memoria y lo retorna como DataFrame.

    Parámetros
    ----------
    service     : recurso Drive v3 autenticado (googleapiclient.discovery.build)
    file_name   : nombre exacto del archivo en Drive (ej: "sellout_consolidado.parquet")
    folder_id   : ID de la carpeta de Drive donde vive el archivo

    Retorna
    -------
    pd.DataFrame con el contenido del Parquet.

    Lanza
    -----
    FileNotFoundError si el archivo no existe en la carpeta.
    RuntimeError si la descarga falla.
    """
    file_id = _buscar_file_id(service, file_name, folder_id)
    if file_id is None:
        raise FileNotFoundError(
            f"'{file_name}' no encontrado en Drive folder {folder_id}"
        )

    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)
    return pd.read_parquet(buffer)
