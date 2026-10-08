from dotenv import load_dotenv
load_dotenv()
from app.drive_client import _get_service

service = _get_service()
parent_id = "1nDXa8xhRwQigelS2UiRsN8kZ_oR0Htm5"

results = service.files().list(
    q=f"'{parent_id}' in parents and trashed = false and mimeType = 'application/vnd.google-apps.folder'",
    fields="files(id, name, modifiedTime)",
    supportsAllDrives=True,
    includeItemsFromAllDrives=True,
).execute()

folders = results.get("files", [])
print(f"Found {len(folders)} subfolder(s):")
for f in folders:
    print(f"  {f['name']}  |  {f['id']}  |  modified: {f['modifiedTime']}")