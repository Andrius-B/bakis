from librosa.core import get_duration

def get_file_duration(filepath):
    return get_duration(filename=filepath)