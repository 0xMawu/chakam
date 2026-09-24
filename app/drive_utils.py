"""
Parses a Google Drive folder ID out of the URL formats admins are likely to
paste. Deliberately strict: if we can't confidently extract an ID, we raise
rather than guess, since folder isolation (Section 7) depends on only ever
ingesting folder IDs that were explicitly approved and correctly captured.
"""
import re

# Matches the two common "share link" shapes:
#   https://drive.google.com/drive/folders/<ID>?usp=sharing
#   https://drive.google.com/drive/u/0/folders/<ID>
_FOLDER_PATH_RE = re.compile(r"/folders/([a-zA-Z0-9_-]+)")

# Matches a bare folder ID pasted directly (no URL at all).
_BARE_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{10,}$")


class InvalidDriveFolderLink(ValueError):
    pass


def extract_folder_id(raw_input: str) -> str:
    """
    Extract a Google Drive folder ID from a pasted link or bare ID string.
    Raises InvalidDriveFolderLink if nothing usable is found.
    """
    text = raw_input.strip()
    if not text:
        raise InvalidDriveFolderLink("Please paste a Google Drive folder link.")

    match = _FOLDER_PATH_RE.search(text)
    if match:
        return match.group(1)

    if _BARE_ID_RE.match(text):
        return text

    raise InvalidDriveFolderLink(
        "Couldn't find a folder ID in that link. Make sure you're pasting a "
        "Google Drive *folder* link (drive.google.com/drive/folders/...)."
    )
