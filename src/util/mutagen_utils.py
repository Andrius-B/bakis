from mutagen.id3 import ID3
from mutagen.easyid3 import EasyID3
from mutagen.flac import FLAC
import logging

log = logging.getLogger(__name__)

def read_mp3_metadata(filepath: str):
    audioFile = EasyID3(filepath)
    def read_property(key):
        try:
            return audioFile[key][0]
        except Exception as err:
            log.error(f"Failed loading metadata property `{key}` from id3 of {filepath}")
            # log.exception(err) 
            return None
    keys = {
        'title': 'title',
        'artist': 'artist',
        'album': 'album',
        'genre': 'genre',
        'date': 'date',
        'yt_link': 'musicip_fingerprint',
        'spotify_uri': 'acoustid_fingerprint',
    }
    metadata = {}
    for name in keys:
        value = read_property(keys[name])
        if value is not None:
            metadata[name] = value
    return metadata

def read_flac_metadata(filepath):
    meta = FLAC(filepath)
    if "artist" in meta and "album" in meta and "title" in meta:
        return {
            "artist": meta["artist"],
            "album": meta["album"],
            "title": meta["title"]
        }
    else:
        return None