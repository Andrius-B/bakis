from dataclasses import dataclass, asdict
from typing import List
import json

@dataclass
class GraphNode:
    track_idx: int
    track_name: str
    track_cluster_size: float

@dataclass
class GraphLink:
    source_track_index: int
    target_track_index: int
    distance: float

@dataclass
class GraphData:
    nodes: List[GraphNode]
    links: List[GraphLink]
    total_samples: int

    def json(self) -> str:
        return json.dumps(asdict(self), indent=4, sort_keys=True)