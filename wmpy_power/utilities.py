from functools import wraps
import io
import logging
from pathlib import Path
import requests
import sys
from timeit import default_timer as timer
from tqdm import tqdm
from typing import Callable
import zipfile


data_manifest = {
    'tutorial': 'https://zenodo.org/records/10530347/files/wmpy_power_tutorial_wauw.zip?download=1',
}


def download_data(data: str = 'tutorial', to: str = './'):
    url = data_manifest[data]
    destination = Path(to).mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    with io.BytesIO() as stream:
        with tqdm.wrapattr(
            stream,
            'write',
            file=sys.stdout,
            miniters=1,
            desc=url,
            total=int(r.headers.get('content-length', 0))
        ) as file:
            for chunk in r.iter_content(chunk_size=4096):
                file.write(chunk)
        with zipfile.ZipFile(stream) as zipped:
            # extract each file in the zip to the destination
            for f in zipped.namelist():
                zipped.extract(f, destination)


def pretty_timer(seconds: float) -> str:
    """
    Formats an elapsed number of seconds in a human friendly way
    :param seconds: number of seconds to prettily format
    :return: string representing the time in reasonably scaled units
    """
    if seconds < 1:
        return f"{round(seconds * 1.0e3, 0)} milliseconds"
    elif seconds < 60:
        return f"{round(seconds, 3)} seconds"
    elif seconds < 3600:
        return f"{int(round(seconds) // 60)} minutes and {int(round(seconds) % 60)} seconds"
    elif seconds < 86400:
        return f"{int(round(seconds) // 3600)} hours, {int((round(seconds) % 3600) // 60)} minutes, and {int(round(seconds) % 60)} seconds"
    else:
        return f"{int(round(seconds) // 86400)} days, {int((round(seconds) % 86400) // 3600)} hours, and {int((round(seconds) % 3600) // 60)} minutes"


def timing(logger: logging.Logger, prefix: str = None) -> Callable:
    """
    Decorator for timing a method and outputting to log
    :param logger: logger to handle the log message
    :param prefix: a string to prepend to the time; default is the method's __name__
    :return: the same method that now reports its timing
    """

    def outer(f: Callable) -> Callable:
        @wraps(f)
        def wrap(*args, **kw):
            t = timer()
            result = f(*args, **kw)
            seconds = timer() - t
            logger.info(
                f"{prefix if prefix is not None else f.__name__}: {pretty_timer(seconds)}"
            )
            return result

        return wrap

    return outer
