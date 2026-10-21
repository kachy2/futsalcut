"""Regions of interest: normalized polygons (display orientation) and their pixel masks.

Rasterization rule (parity contract with the web port): pixel (x, y) is inside when its centre
(x + 0.5, y + 0.5), in normalized units ((x + 0.5) / W, (y + 0.5) / H), is inside the polygon by the
even-odd crossing test below, evaluated in float64.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

Point = tuple[float, float]


def point_in_polygon(poly: list[Point], x: float, y: float) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xc = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < xc:
                inside = not inside
    return inside


def polygon_mask(poly: list[Point], width: int, height: int) -> np.ndarray:
    """Boolean (H, W) mask, vectorised version of point_in_polygon at pixel centres."""
    xs = (np.arange(width, dtype=np.float64) + 0.5) / width
    ys = (np.arange(height, dtype=np.float64) + 0.5) / height
    X, Y = np.meshgrid(xs, ys)
    inside = np.zeros((height, width), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if y1 == y2:
            continue
        crosses = (y1 > Y) != (y2 > Y)
        xc = (x2 - x1) * (Y - y1) / (y2 - y1) + x1
        inside ^= crosses & (X < xc)
    return inside


def clip_below(poly: list[Point], y_min: float) -> list[Point]:
    """Sutherland-Hodgman clip keeping the part with y >= y_min (image y grows downwards)."""
    out: list[Point] = []
    n = len(poly)
    for i in range(n):
        cur, nxt = poly[i], poly[(i + 1) % n]
        cin, nin = cur[1] >= y_min, nxt[1] >= y_min
        if cin:
            out.append(cur)
        if cin != nin:
            t = (y_min - cur[1]) / (nxt[1] - cur[1])
            out.append((cur[0] + t * (nxt[0] - cur[0]), y_min))
    return out


@dataclass
class Roi:
    near: list[Point]
    band_y: float | None = None
    ignore: list[list[Point]] = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "Roi":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(near=[tuple(p) for p in d["near"]], band_y=d.get("band_y"),
                   ignore=[[tuple(p) for p in poly] for poly in d.get("ignore", [])])

    def zone(self, strict: bool = True) -> list[Point]:
        """Counting zone: the near polygon, clipped below the midfield band in strict mode."""
        if strict and self.band_y is not None:
            return clip_below(self.near, self.band_y)
        return self.near

    def mask(self, width: int, height: int, strict: bool = True) -> np.ndarray:
        m = polygon_mask(self.zone(strict), width, height)
        for poly in self.ignore:
            m &= ~polygon_mask(poly, width, height)
        return m
