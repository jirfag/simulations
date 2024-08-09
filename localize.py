import os


is_russian = os.environ.get("LANG", "en_US")[: len("en")] == "ru"


def localize(str_eng: str, str_rus: str) -> str:
    return str_rus if is_russian else str_eng
