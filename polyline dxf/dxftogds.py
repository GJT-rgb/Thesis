#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DXF → GDSII converter using ezdxf + gdstk.

- Supports: LINE, ARC, SPLINE, ELLIPSE
- Skips: 3DSOLID (projection can be added if needed)
- Uses ezdxf's .flattening(distance) for robust curve approximation (works on old/new ezdxf)
- Auto-scales from DXF $INSUNITS to GDS microns (library unit = 1e-6 m)
- Closed curves → Polygons; Open curves → Paths (configurable width)
"""

import argparse
import math
import os
import sys
from typing import Iterable, List, Sequence, Tuple

import ezdxf
import gdstk


# -----------------------------
# Configuration (edit if you want to run by double-click)
# -----------------------------
DEFAULT_INPUT = r"E:/EMBL/Master EMBL/Python/polyline dxf/Cardio_projectioncentered.dxf"
DEFAULT_OUTPUT = r"E:/EMBL/Master EMBL/Python/polyline dxf/Cardio_projection3D.gds"


# -----------------------------
# Units: map AutoCAD $INSUNITS code to meters per drawing unit
# -----------------------------
INSUNITS_TO_METERS = {
    0: None,               # Unspecified
    1: 0.0254,             # Inches
    2: 0.3048,             # Feet
    3: 1609.344,           # Miles
    4: 1e-3,               # Millimeters
    5: 1e-2,               # Centimeters
    6: 1.0,                # Meters
    7: 1e3,                # Kilometers
    8: 2.54e-8,            # Microinches
    9: 2.54e-5,            # Mils (thou)
    10: 0.9144,            # Yards
    11: 1e-10,             # Angstroms
    12: 1e-9,              # Nanometers
    13: 1e-6,              # Microns (micrometers)
    14: 1e-1,              # Decimeters
    15: 10.0,              # Decameters
    16: 100.0,             # Hectometers
    17: 1e9,               # Gigameters
    21: 1200.0 / 3937.0,   # US Survey Foot
}


def get_scale_to_gds_microns(doc, assume_mm_if_unspecified: bool = True) -> Tuple[float, str]:
    """
    Compute scale factor from DXF drawing units to microns (GDS library unit = 1e-6 m).
    Returns (scale_factor, unit_name).
    """
    code = int(doc.header.get("$INSUNITS", 0) or 0)
    meters_per_unit = INSUNITS_TO_METERS.get(code)

    if meters_per_unit is None:
        # Unspecified: assume millimeters
        if assume_mm_if_unspecified:
            meters_per_unit = 1e-3
            unit_name = "unspecified → assumed millimeters"
        else:
            meters_per_unit = 1.0
            unit_name = "unspecified → assumed meters"
    else:
        unit_name = {
            1: "inches", 2: "feet", 3: "miles", 4: "millimeters", 5: "centimeters",
            6: "meters", 7: "kilometers", 8: "microinches", 9: "mils", 10: "yards",
            11: "angstroms", 12: "nanometers", 13: "microns", 14: "decimeters",
            15: "decameters", 16: "hectometers", 17: "gigameters", 21: "US survey feet"
        }.get(code, f"code {code}")

    # GDS library unit = 1e-6 m (micron)
    # scale factor = meters_per_dxf_unit / (1e-6 m per GDS unit) = value in microns
    scale = meters_per_unit / 1e-6
    return float(scale), unit_name


# -----------------------------
# Geometry helpers
# -----------------------------
Point = Tuple[float, float]


def scale_points(pts: Iterable[Sequence[float]], s: float) -> List[Point]:
    return [(float(p[0]) * s, float(p[1]) * s) for p in pts]


def is_closed(pts: Sequence[Point], tol: float = 1e-6) -> bool:
    if len(pts) < 2:
        return False
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    return (x0 - x1) ** 2 + (y0 - y1) ** 2 <= tol * tol


def add_polygon_or_path(cell: gdstk.Cell, pts: List[Point], layer: int, path_width: float, close_tol: float = 1e-6):
    """Add a polygon if closed; otherwise add a FlexPath with the given width."""
    if len(pts) < 2:
        return
    if is_closed(pts, tol=close_tol) and len(pts) >= 3:
        # Ensure exact closure for polygons
        if pts[0] != pts[-1]:
            pts = pts + [pts[0]]
        cell.add(gdstk.Polygon(pts, layer=layer))
    else:
        cell.add(gdstk.FlexPath(pts, width=max(path_width, 1e-6), layer=layer))


def arc_points(center: Point, radius: float, a0: float, a1: float, steps: int = 128) -> List[Point]:
    """Sample an arc into polyline points."""
    # normalize ordering
    if a1 < a0:
        a0, a1 = a1, a0
    dt = (a1 - a0) / max(1, steps)
    return [(center[0] + radius * math.cos(a0 + i * dt),
             center[1] + radius * math.sin(a0 + i * dt)) for i in range(steps + 1)]


def flatten_curve(entity, distance: float) -> List[Point]:
    """Flatten any ezdxf curve-like entity into (x, y) points (unscaled)."""
    return [(p[0], p[1]) for p in entity.flattening(distance)]


# -----------------------------
# Main conversion
# -----------------------------
def dxf_to_gds(
    dxf_path: str,
    gds_path: str,
    *,
    layer: int = 1,
    path_width: float = 1.0,
    flat_tol: float = 0.05,
    assume_mm_if_unspecified: bool = True,
) -> None:
    """
    Convert DXF to GDSII.

    Args:
        dxf_path: input DXF file.
        gds_path: output GDS file.
        layer: GDS layer number (int).
        path_width: width (in GDS units, microns) for open curves (paths).
        flat_tol: flattening tolerance in DXF units (smaller = smoother).
        assume_mm_if_unspecified: if $INSUNITS missing, assume mm (True) or meters (False).
    """
    if not os.path.isfile(dxf_path):
        raise FileNotFoundError(f"DXF file not found: {dxf_path}")

    print(f"[INFO] Reading DXF: {dxf_path}")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    scale_to_gds, unit_name = get_scale_to_gds_microns(doc, assume_mm_if_unspecified)
    print(f"[INFO] DXF units: {unit_name} → scale to GDS microns: ×{scale_to_gds:g}")

    # Prepare GDS library with micron units
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("DXF_IMPORT")

    counts = {"LINE": 0, "ARC": 0, "SPLINE": 0, "ELLIPSE": 0, "3DSOLID": 0, "OTHER": 0}
    added = 0

    for e in msp:
        etype = e.dxftype()

        try:
            if etype == "LINE":
                start = (e.dxf.start.x * scale_to_gds, e.dxf.start.y * scale_to_gds)
                end = (e.dxf.end.x * scale_to_gds, e.dxf.end.y * scale_to_gds)
                cell.add(gdstk.FlexPath([start, end], width=max(path_width, 1e-6), layer=layer))
                counts["LINE"] += 1
                added += 1

            elif etype == "ARC":
                center = (e.dxf.center.x * scale_to_gds, e.dxf.center.y * scale_to_gds)
                radius = float(e.dxf.radius) * scale_to_gds
                a0 = math.radians(float(e.dxf.start_angle))
                a1 = math.radians(float(e.dxf.end_angle))
                # Sample arc into polyline, then add as path/polygon depending on closure
                steps = max(16, int(180 * max(1.0, radius) ** 0.25))
                pts = arc_points(center, radius, a0, a1, steps=steps)
                add_polygon_or_path(cell, pts, layer=layer, path_width=path_width)
                counts["ARC"] += 1
                added += 1

            elif etype == "SPLINE":
                pts_unscaled = flatten_curve(e, distance=flat_tol)
                pts = scale_points(pts_unscaled, scale_to_gds)
                add_polygon_or_path(cell, pts, layer=layer, path_width=path_width)
                counts["SPLINE"] += 1
                added += 1

            elif etype == "ELLIPSE":
                pts_unscaled = flatten_curve(e, distance=flat_tol)
                pts = scale_points(pts_unscaled, scale_to_gds)
                add_polygon_or_path(cell, pts, layer=layer, path_width=path_width)
                counts["ELLIPSE"] += 1
                added += 1

            elif etype == "3DSOLID":
                counts["3DSOLID"] += 1
                # Skipped: needs projection/silhouette extraction for 2D export

            else:
                counts["OTHER"] += 1

        except Exception as ex:
            print(f"[WARN] Failed to convert entity {etype}: {ex!r}")

    if added == 0:
        print("[WARN] No 2D geometry added to GDS. "
              "Your DXF may contain only unsupported entities (e.g., 3DSOLID) or be empty.")
    else:
        print(f"[INFO] Added {added} geometry items to GDS.")

    # Write GDS
    outdir = os.path.dirname(os.path.abspath(gds_path)) or "."
    os.makedirs(outdir, exist_ok=True)
    lib.write_gds(gds_path)

    print(f"[DONE] GDS file written to: {gds_path}")
    print("[STATS] Imported counts:", counts)


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert DXF (2D) to GDSII using ezdxf + gdstk.")
    p.add_argument("input", nargs="?", default=DEFAULT_INPUT, help="Input DXF file path")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output GDS file path")
    p.add_argument("--layer", type=int, default=1, help="GDS layer number (default: 1)")
    p.add_argument("--path-width", type=float, default=1.0,
                   help="Width (in GDS units, microns) for open curves (default: 1.0)")
    p.add_argument("--flat-tol", type=float, default=0.05,
                   help="Flattening tolerance in DXF units (default: 0.05)")
    p.add_argument("--assume-mm", dest="assume_mm", action="store_true",
                   help="If DXF units unspecified, assume millimeters (default)")
    p.add_argument("--assume-m", dest="assume_mm", action="store_false",
                   help="If DXF units unspecified, assume meters")
    p.set_defaults(assume_mm=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        dxf_to_gds(
            dxf_path=args.input,
            gds_path=args.output,
            layer=args.layer,
            path_width=args.path_width,
            flat_tol=args.flat_tol,
            assume_mm_if_unspecified=args.assume_mm,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except ezdxf.DXFStructureError as e:
        print(f"[ERROR] Corrupt or unsupported DXF: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        sys.exit(3)