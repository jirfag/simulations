from dataclasses import dataclass
from typing import List
from typing import NamedTuple

from shared import TimeDuration, TimePoint


class ClientResult(NamedTuple):
    timepoint: TimePoint
    duration: TimeDuration
    is_ok: bool


ClientResultList = List[ClientResult]


class ServerResult(NamedTuple):
    timepoint: TimePoint  # end of the request time
    handle_duration: TimeDuration
    pending_req_count: int
    is_ok: bool


ServerResultList = List[ServerResult]
