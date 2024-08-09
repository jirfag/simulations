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


PANEL_CLIENT_UPTIME = localize("Client Uptime", "аптайм клиентов")
PANEL_SERVER_RPS_AMPLIFICATION = localize(
    "Server RPS Amplification", "амплификация нагрузки на сервер"
)
PANEL_CLIENT_LATENCY_P50 = localize("Client Latency p50", "p50 клиенткие тайминги")
PANEL_CLIENT_LATENCY_P99 = localize("Client Latency p99", "p99 клиенткие тайминги")
PANEL_CLIENT_LATENCY_AVG = localize("Client Latency Avg", "средние клиенткие тайминги")
PANEL_SERVER_FAILURE_RATE = localize("Server Failure Rate", "доля ошибок сервера")
PANEL_SERVER_AVG_REQ_QUEUE = localize(
    "Avg Server Queue Size", "средний размер очереди запросов сервера"
)
PANEL_SERVER_MAX_REQ_QUEUE = localize(
    "Max Server Queue Size", "макс размер очереди запросов сервера"
)
PANEL_SERVER_UPTIME = localize("Server Uptime", "аптайм сервера")
PANEL_SERVER_AVG_CPU_USAGE = localize(
    "Avg Server CPU Usage", "средний cpu usage сервера"
)
PANEL_SERVER_AVG_LATENCY = localize("Avg Server Latency", "средние серверные тайминги")
