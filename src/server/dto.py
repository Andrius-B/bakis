from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
import json

@dataclass
class GraphNode:
    track_idx: int
    track_name: str
    track_cluster_size: float
    metadata: Optional[Dict[str, str]]

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
        return json.dumps(asdict(self))