from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from localize import localize

from shared import TimeInterval


@dataclass
class VisualizeParams:
    needed_panels: List[str]
    id_to_color_num: Dict[str, int]
    out_image_path: str
    needed_lines: Optional[List[str]] = None
    row_n: int = 1
    col_n: int = 0
    shareX: bool = False
    dpi: int = 600
    xlim: Tuple[Optional[float], Optional[float]] = (None, None)
    highlight_interval: Optional[TimeInterval] = None


PANEL_CLIENT_UPTIME = localize("client uptime", "аптайм клиентов")
PANEL_SERVER_RPS_AMPLIFICATION = localize("server rps amplification", "амплификация нагрузки на сервер")
PANEL_CLIENT_LATENCY_P50 = localize("p50 client latency", "p50 клиенткие тайминги")
PANEL_CLIENT_LATENCY_P99 = localize("p99 client latency", "p99 клиенткие тайминги")
PANEL_CLIENT_LATENCY_AVG = localize("avg client latency", "средние клиенткие тайминги")
PANEL_SERVER_FAILURE_RATE = localize("server failure rate", "доля ошибок сервера")
PANEL_SERVER_AVG_REQ_QUEUE = localize("avg server queue size", "средний размер очереди запросов сервера")
PANEL_SERVER_MAX_REQ_QUEUE = localize("max server queue size", "макс размер очереди запросов сервера")
PANEL_SERVER_UPTIME = localize("server uptime", "аптайм сервера")
PANEL_SERVER_AVG_CPU_USAGE = localize("avg server cpu usage", "средний cpu usage сервера")
PANEL_SERVER_AVG_LATENCY = localize("avg server latency", "средние серверные тайминги")
