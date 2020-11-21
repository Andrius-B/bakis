from mutagen.id3 import ID3
from mutagen.flac import FLAC

def read_mp3_metadata(filepath):
    id3 = ID3(filepath)
    return {
        "artist": id3["TPE1"],
        "album": id3["TALB"],
        "title": id3["TIT2"]
    }

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