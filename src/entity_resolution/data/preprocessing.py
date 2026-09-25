def normalize_text(text: str) -> str:
    """Normalize business names and addresses."""
    if text is None or str(text).lower() == 'nan' or str(text).lower() == 'null':
        return ""
    return str(text).lower().strip()
