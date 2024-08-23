from dataclasses import dataclass


@dataclass
class Request:
    request_id: str
    arrival_time: float


