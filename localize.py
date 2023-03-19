import os


is_english = os.environ.get("LANG", "en_US")[: len("en")] == "en"


def localize(str_eng: str, str_rus: str) -> str:
    return str_eng if is_english else str_rus
