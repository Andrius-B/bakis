from logging import getLogger
from sqlite3 import connect
from typing import Generator, List
from pickle import dumps, loads
from math import floor, ceil
from torch import cat
import numpy as np
import random

log = getLogger(__name__)

class SQLiteStorageObject:
    def __init__(
        self,
        artist: str, album: str, track_name: str,
        sample_rate: int, number_frames: int, filepath: str,
        spectrogram, metadata_id: int = None, data_id: int = None
        ):
        self.artist = artist
        self.album = album
        self.track_name = track_name
        self.sample_rate = sample_rate
        self.number_frames = number_frames
        self.filepath = filepath
        self.spectrogram = spectrogram
        self.metadata_id = metadata_id
        self.data_id = data_id

class SQLiteStorage:
    def __init__(
        self,
        db_file: str,
        window_generation_strategy = None
    ):
        self._db_file = db_file
        self.conn = connect(self._db_file)
        if(window_generation_strategy == None):
            self.window_generation_strategy = RandomSubsampleWindowGenerationStrategySQLite(2**16, 2**16 / 4)
        else:
            self.window_generation_strategy = window_generation_strategy
        self.chunk_size = 128
        self._commit(["""
        CREATE TABLE IF NOT EXISTS metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            artist STRING NOT NULL,
            album STRING NOT NULL,
            track_name STRING NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS filedata (
            id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            metadata_id INTEGER NOT NULL,
            filepath STRING NOT NULL,
            sample_rate INTEGER NOT NULL,
            number_frames INTEGER NOT NULL,
            FOREIGN KEY(metadata_id) REFERENCES metadata(id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS filechunks (
            id INTEGER NOT NULL,
            filedata_id INTEGER NOT NULL,
            spectrogram_chunk_blob BLOB NOT NULL,
            PRIMARY KEY (id, filedata_id),
            FOREIGN KEY(filedata_id) REFERENCES filedata(id)
        );
        """])

    def _commit(self, statements: List[str], args: List = None):
        cursor = self.conn.cursor()
        for i, statement in enumerate(statements):
            if args != None:
                cursor.execute(statement, args[i])
            else:
                cursor.execute(statement)
        self.conn.commit()
    
    def insert_new_track(self, track: SQLiteStorageObject):
        cursor = self.conn.cursor()
        values1 = (str(track.artist), str(track.album), str(track.track_name))
        log.info(f"Inserting values into metadata: {str(values1)}")
        cursor.execute("""
        INSERT INTO metadata (artist, album, track_name) VALUES (?, ?, ?)
        """, values1)
        m_id = cursor.lastrowid
        values2 = (m_id, track.filepath, track.sample_rate, track.number_frames)
        log.info(f"Inserting values into filedata: {str(values2)}")
        cursor.execute("""
        INSERT INTO filedata (metadata_id, filepath, sample_rate, number_frames)
            VALUES (?, ?, ?, ?)
        """, values2)
        filedata_id = cursor.lastrowid
        print(f"Spectrogram shape: {track.spectrogram.shape}")
        s = track.spectrogram.shape
        i = 0
        chunk_index = 1
        while(i < s[2] - self.chunk_size):
            chunk_end = min(i + self.chunk_size, s[2])
            chunk = track.spectrogram[:, :, i:chunk_end]
            print(f"Chunk starting at {i} shape {chunk.shape}")
            chunk_blob = dumps(chunk)
            cursor.execute("""
            INSERT INTO filechunks (id, filedata_id, spectrogram_chunk_blob)
                VALUES (?, ?, ?)
            """, (chunk_index, filedata_id, chunk_blob))
            i += self.chunk_size
            chunk_index += 1
        self.conn.commit()

    def generate_windows(self) -> List[int]:
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT metadata.id,filedata.id,filedata.number_frames,filedata.sample_rate FROM metadata INNER JOIN filedata ON metadata.id = filedata.id
        """)
        for item in cursor:
            track = SQLiteAudioFileDescriptor(item[0], item[1], item[2], item[3])
            for window in self.window_generation_strategy.generate_windows(track, self):
                yield window

    def load_blob(self, filedata_id: int, start_slice: int, end_slice: int):
        """
        The start slice and end slice is the time dimension in the spectrogram --
        by convention, the network uses square spectrcograms of 129x129, but here slice specifies the second
        dimension
        """
        cursor = self.conn.cursor()
        first_chunk_idx = floor(start_slice/self.chunk_size)+1
        last_chunk_idx = floor(end_slice/self.chunk_size)+1
        chunk_ids = range(first_chunk_idx, last_chunk_idx+1)
        print(f"Will load blob from chunks: {list(chunk_ids)}")
        result = None
        for chunk_id in chunk_ids:
            cursor.execute("""
            SELECT spectrogram_chunk_blob FROM filechunks WHERE id = ? AND filedata_id = ?
            """, (chunk_id, filedata_id))
            try:
                chunk_blob = cursor.fetchone()[0]
            except Exception as e:
                print(f"Read blob failed with params: {str((chunk_id, filedata_id))}")
                raise e
            chunk = loads(chunk_blob)
            if result is None:
                result = chunk
            else:
                print(f"Before append: {result.shape}")
                result = np.append(result, chunk, axis=2)
                print(f"After append: {result.shape}")
        relative_chunk_offset_start = first_chunk_idx*self.chunk_size
        print(f"Concatinated result tensor shape: {result.shape} (asked for {end_slice - start_slice})")
        result_chunk_start = start_slice - relative_chunk_offset_start
        result_chunk_end = end_slice - relative_chunk_offset_start
        result = result[:, :, result_chunk_start:result_chunk_end]
        print(f"Sliced result tensor shape: {result.shape} (asked for {start_slice}-{end_slice} or {end_slice-start_slice} total)")
        return result


class SQLiteAudioFileDescriptor:
    """
    Class used for generating windos for sqlite datasets
    """
    def __init__(self, metadata_id: int, data_id: int, number_frames:int, sample_rate: int):
        self.metadata_id = metadata_id
        self.data_id = data_id
        self.number_frames = number_frames
        self.sample_rate = sample_rate
    def __repr__(self):
        return (f"SQLiteAudioFileDescriptor(metadata_id={self.metadata_id},data_id={self.data_id},"
            + f"number_frames={self.number_frames},sample_rate={self.sample_rate})")

class SQLiteAudioData:
    def __init__(self, metadata: SQLiteAudioFileDescriptor, spectrogram):
        self.metadata = metadata
        self.spectrogram = spectrogram

class AbstractSQLiteWindow:
    def read_data(self) -> SQLiteAudioData:
        return None

class SpecificAudioFileWindowSQLite(AbstractSQLiteWindow):
    def __init__(self, storage_ref: SQLiteStorage, track: SQLiteAudioFileDescriptor, window_start: float, window_end: float):
        self.storage = storage_ref
        self.track = track
        self.window_start = window_start
        self.window_end = window_end

    def __repr__(self):
        # print(f"file_index: {self.file_index} files: {self.storage._file_array}")
        return f"{self.track.data_id}|({self.window_start},{self.window_end})"
    def read_data(self):
        spectro_start = int((self.window_start * self.track.sample_rate) / 512)
        spectro_stop = int(round((self.window_end * self.track.sample_rate) / 512, -1))        
        blob = self.storage.load_blob(self.track.data_id, spectro_start, spectro_stop)
        return SQLiteAudioData(self.track, blob)


    @classmethod
    def from_string(cls, s, storage):
        data_id, window = s.split("|")
        w_start, w_end = map(float, window[1:-1].split(","))
        return SpecificAudioFileWindowSQLite(storage, int(data_id), w_start, w_end)

class RandomAudioFileWindowSQLite(AbstractSQLiteWindow):

    def __init__(self, storage_ref: SQLiteStorage, track: SQLiteAudioFileDescriptor, window_index: int, window_len: float):
        self.storage = storage_ref
        self.track = track
        self.window_len = window_len
        self.window_index = window_index

    def __repr__(self):
        # print(f"file_index: {self.file_index} files: {self.storage._file_array}")
        return f"{self.track.data_id}|({self.window_index},{self.window_len})"

    def read_data(self):
        # blob = self.storage.load_blob(self.track.data_id)
        
        max_offset = int((self.track.number_frames / 512) - ((self.window_len * self.track.sample_rate) / 512))
        offset = random.randint(0, max_offset)
        spectro_length = int(round((self.window_len * self.track.sample_rate) / 512, -1))
        spectro_stop = offset + spectro_length
        blob = self.storage.load_blob(self.track.data_id, offset, spectro_stop)
        # # print(f"Narrowing (d_id: {self.track.data_id}) tensor: {blob.shape} to [{offset}:{offset + spectro_length}] ({spectro_length}) on the second dim")
        # try:
        #     # blob = blob[:, :, offset:spectro_stop, :]
        #     gc.collect()
        # except Exception as e:
        #     print("Narrowing failed!")
        #     print(f"Item it failed on:{self.track}")
        #     raise e
        return SQLiteAudioData(self.track, blob)

    # def get_filepath(self) -> str:
    #     return self.storage._file_array[self.file_index]

    @classmethod
    def from_string(cls, s, storage):
        data_id, window = s.split("|")
        window_index = int(window[1:-1].split(",")[0])
        window_len = float(window[1:-1].split(",")[1])
        return RandomAudioFileWindowSQLite(storage, int(data_id), window_index, window_len)


class UniformReadWindowGenerationStrategySQLite:
    """
    Window generation strategy that reads windows from a file with a uniform step
    This generates A LOT of windows.
    """

    def __init__(
        self,
        window_len: int = 2**16,
        window_hop: int = (2**16 / 4),
        overread: int = 1.10
    ):
        """Parameters here are sample counts"""
        self.window_len: int = window_len
        self.window_hop: int = window_hop
        self.overread = overread

    def generate_windows(self, track: SQLiteAudioFileDescriptor, storage: SQLiteStorage):
        window_size = self.window_len / track.sample_rate  # how long is a sample of 2**16 samples when resampled to 41kHz?
        window_size += 1 / (track.sample_rate*10)  # to make the float error more consistent
        i = 0  # index measued in seconds
        duration = track.number_frames / track.sample_rate
        step = self.window_hop / track.sample_rate  # leave overlap.
        while(i < duration - (window_size * self.overread)):
            # w = SpecificAudioFileWindow(storage, f_id, i, i + window_size)
            # print(f"Generated window: {str(w)} max offset: {duration - window_size} current offset: {i}")
            yield SpecificAudioFileWindowSQLite(storage, track, i, i + window_size)
            i += step


class RandomSubsampleWindowGenerationStrategySQLite:
    """
    Window generation strategy that generates a preset amount of windows per file depending on duration,
    it only lists the index of the window. This is intended to be used for a random read of the required length
    to generate less windows.
    """

    def __init__(
        self,
        window_len: int = 2**16,
        average_hop: int = 2**16,
        overread: float = 1.0
    ):
        """Parameters here are sample counts"""
        self.window_len: int = window_len
        self.average_hop: int = average_hop
        self.overread = overread

    def generate_windows(self, track: SQLiteAudioFileDescriptor, storage: SQLiteStorage):
        window_size = self.window_len / track.sample_rate
        # window_size += 1 / (44100*10)
        i = 0  # index measued in seconds
        step = self.average_hop / track.sample_rate
        generated_windows = 0
        duration = track.number_frames / track.sample_rate
        while(i < duration - (window_size * self.overread)):
            yield RandomAudioFileWindowSQLite(storage, track, generated_windows, window_size)
            i += step
            generated_windows += 1
