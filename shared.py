from dataclasses import dataclass
from typing import Any, Optional

TimePoint = int
TimeDuration = int


# TimeInterval represents time interval with both sided included
@dataclass
class TimeInterval:
    begin: TimePoint
    end: TimePoint

    # is_in returns is timepoint inside interval with both sided included
    def is_in(self, timepoint: TimePoint) -> bool:
        return timepoint >= self.begin and timepoint <= self.end

    def length(self) -> TimeDuration:
        return self.end - self.begin

    def intersect(self, other: Any) -> Optional[Any]:
        res = TimeInterval(max(self.begin, other.begin), min(self.end, other.end))
        if res.begin > res.end:
            return None
        return res


TIME_DURATION_SECOND: TimeDuration = 1_000_000_000  # precision of nanoseconds
TIME_DURATION_MS: TimeDuration = 1_000_000

MEAN_RTT_MS = 5