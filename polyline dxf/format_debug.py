from typing import Dict, List, Tuple, Iterable, Optional
import os
import sys
import time
import traceback

import gdstk
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
try:
    # Shapely 1.8+ / 2.x
    from shapely.validation import make_valid
except Exception:
    # Fallback if make_valid is unavailable
    def make_valid(geom):
        # buffer(0) is a common topology fix
        return geom.buffer(0)

import ezdxf
from ezdxf import units as dxf_units


def _print_versions():
    try:
        import shapely, ezdxf
        print("Python:", sys.version, flush=True)
        print("gdstk:", getattr(gdstk, "__version__", "unknown"), flush=True)
        print("Shapely:", getattr(shapely, "__version__", "unknown"), flush=True)
        print("ezdxf:", getattr(ezdxf, "__version__", "unknown"), flush=True)
    except Exception as e:
        print("Could not print versions:", e, flush=True)


def _as_polygon(points):
    if not points or len(points) < 3:
        return None
    # drop last point if already closed
    if points[0][0] == points[-1][0] and points[0][1] == points[-1][1]:
        pts = points[:-1]
    else:
        pts = points
    if len(pts) < 3:
        return None
    return Polygon(pts)


def _collect_polys_from_lib(lib, layers: Optional[Iterable[Tuple[int, int]]]):
    """
    Robust collector that includes geometry from referenced cells.
    Prefers gdstk's get_polygons(by_spec=True, include_paths=True) on top-level cells,
    falling back to manual iteration if necessary.
    """
    out: Dict[Tuple[int, int], List[Polygon]] = {}

    # Prefer top-level cells; they include the full design by following references
    tops = []
    try:
        tops = list(lib.top_level())
    except Exception:
        # Older gdstk versions might differ; fallback to all cells
        tops = list(getattr(lib, "cells", []))

    if not tops:
        tops = list(getattr(lib, "cells", []))

    used_get_polygons = False
    for cell in tops:
        # Try robust API: include polygons from references
        try:
            # include_paths tries to convert paths to polygons
            # depth=None (or omitting) should traverse references fully
            poly_dict = cell.get_polygons(by_spec=True, include_paths=True)  # type: ignore
            used_get_polygons = True
            for (lay, dt), poly_arrays in poly_dict.items():
                if layers and (lay, dt) not in layers:
                    continue
                for arr in poly_arrays:
                    poly = _as_polygon([tuple(p) for p in arr])
                    if poly and not poly.is_empty:
                        out.setdefault((lay, dt), []).append(poly)
        except Exception:
            # Fall back to manual collection (direct contents only)
            for ply in getattr(cell, "polygons", []):
                lay, dt = getattr(ply, "layer", None), getattr(ply, "datatype", None)
                if lay is None or dt is None:
                    continue
                if layers and (lay, dt) not in layers:
                    continue
                poly = _as_polygon([tuple(p) for p in ply.points])
                if poly and not poly.is_empty:
                    out.setdefault((lay, dt), []).append(poly)

            for path in getattr(cell, "paths", []):
                lay, dt = getattr(path, "layer", None), getattr(path, "datatype", None)
                if lay is None or dt is None:
                    continue
                if layers and (lay, dt) not in layers:
                    continue
                try:
                    for pts in path.to_polygons():
                        poly = _as_polygon([tuple(p) for p in pts])
                        if poly and not poly.is_empty:
                            out.setdefault((lay, dt), []).append(poly)
                except Exception:
                    # If to_polygons is not available or fails, skip that path
                    pass

    print(f"Used get_polygons: {used_get_polygons}", flush=True)
    return out


def gds_to_merged_dxf(
    input_gds: str,
    output_dxf: str,
    *,
    layers: Optional[Iterable[Tuple[int, int]]] = None,
    dxf_units_mm: bool = True,
    scale: float = 1.0,
    simplify: float = 0.0,
    merge_touching: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Read a GDS, merge touching/overlapping polygons per (layer, datatype),
    preserve interior holes, and save as DXF with closed polylines.
    """
    def _robust_union(polys: List[Polygon]):
        if not polys:
            return MultiPolygon()
        fixed = []
        for p in polys:
            if not p.is_valid:
                p = make_valid(p)
            p = p.buffer(0)  # clean topology
            if not p.is_empty:
                fixed.append(p)
        if not fixed:
            return MultiPolygon()
        if merge_touching:
            merged = unary_union(fixed)
        else:
            # Separate touching shapes by shrinking slightly, then expand back
            eps = 1e-9 * max(1.0, scale)
            shrunk = [p.buffer(-eps) for p in fixed if not p.buffer(-eps).is_empty]
            merged = unary_union(shrunk).buffer(eps)
        return merged.buffer(0)

    def _ring_coords(seq, _scale: float, _simp: float):
        coords = list(seq.coords)
        if len(coords) >= 2 and coords[0] == coords[-1]:
            coords = coords[:-1]
        if _simp > 0:
            poly_tmp = Polygon(coords)
            poly_tmp = poly_tmp.simplify(_simp, preserve_topology=True)
            coords = list(poly_tmp.exterior.coords)
            if len(coords) >= 2 and coords[0] == coords[-1]:
                coords = coords[:-1]
        return [(x * _scale, y * _scale) for (x, y) in coords]

    # ---------- I/O checks ----------
    print("[0/6] Environment and paths", flush=True)
    _print_versions()
    if not os.path.exists(input_gds):
        raise FileNotFoundError(f"Input GDS not found: {input_gds}")
    out_dir = os.path.dirname(output_dxf)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # ---------- read GDS ----------
    t0 = time.perf_counter()
    print(f"[1/6] Reading GDS: {input_gds}", flush=True)
    lib = gdstk.read_gds(input_gds)
    print(f"    - #cells in lib: {len(getattr(lib, 'cells', []))}", flush=True)

    # ---------- collect polys ----------
    print("[2/6] Collecting polygons (including references)...", flush=True)
    layer_geoms = _collect_polys_from_lib(lib, layers)
    total_polys = sum(len(v) for v in layer_geoms.values())
    print(f"    - layer pairs: {len(layer_geoms)} | total polys: {total_polys}", flush=True)
    if not layer_geoms:
        raise ValueError(
            "No polygons/paths found. "
            "Likely causes: (a) wrong layers filter, (b) geometry only in referenced cells "
            "and collector didn't reach them, (c) empty/unsupported content."
        )

    if dry_run:
        print("[DRY RUN] Stopping before merge/write.", flush=True)
        print(f"Elapsed: {time.perf_counter() - t0:.2f}s", flush=True)
        return

    # ---------- prepare DXF ----------
    print("[3/6] Preparing DXF document...", flush=True)
    doc = ezdxf.new("R2018")
    if dxf_units_mm:
        doc.units = dxf_units.MM
    msp = doc.modelspace()

    # ---------- merge and write ----------
    print("[4/6] Merging and writing per (layer,datatype)...", flush=True)
    for i, ((lay, dt), polys) in enumerate(layer_geoms.items(), start=1):
        print(f"    - [{i}/{len(layer_geoms)}] L{lay}/D{dt}: {len(polys)} polygon(s)", flush=True)
        t_merge = time.perf_counter()
        merged = _robust_union(polys)
        print(f"      merge time: {time.perf_counter() - t_merge:.2f}s", flush=True)
        if merged.is_empty:
            print("      merged empty, skipping", flush=True)
            continue

        if isinstance(merged, Polygon):
            poly_iter = [merged]
        else:
            poly_iter = [g for g in getattr(merged, "geoms", []) if isinstance(g, Polygon)]

        dxf_layer_name = f"L{lay}_D{dt}"
        if dxf_layer_name not in doc.layers:
            doc.layers.add(dxf_layer_name)

        for spoly in poly_iter:
            if spoly.area <= 0:
                continue
            # Outer boundary
            outer = _ring_coords(spoly.exterior, scale, simplify)
            if len(outer) >= 3:
                msp.add_lwpolyline(
                    outer,
                    close=True,
                    dxfattribs={"layer": dxf_layer_name, "color": 7},
                )
            # Holes as inner closed polylines
            for hole in spoly.interiors:
                hole_pts = _ring_coords(hole, scale, simplify)
                if len(hole_pts) >= 3:
                    msp.add_lwpolyline(
                        hole_pts,
                        close=True,
                        dxfattribs={"layer": dxf_layer_name, "color": 1},
                    )

    print("[5/6] Saving DXF...", flush=True)
    doc.saveas(output_dxf)
    print(f"[6/6] Saved merged DXF with polylines → {output_dxf}", flush=True)
    print(f"Total time: {time.perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    # >>> Update your paths here (use raw strings or forward slashes)
    input_path = r"E:\EMBL\Master EMBL\Python\polyline dxf\Cardio_projection3D.gds"
    output_path = r"E:\EMBL\Master EMBL\Python\polyline dxf\Cardio_projection3D_polylined.dxf"

    print("Starting conversion...", flush=True)
    try:
        # 1) Optional: quick inspection only (uncomment once to verify)
        # gds_to_merged_dxf(input_path, output_path, dry_run=True)

        # 2) Actual conversion
        gds_to_merged_dxf(
            input_gds=input_path,
            output_dxf=output_path,
            scale=1.0,          # set to 0.001 if GDS units are µm and you want mm in DXF
            simplify=0.0,       # try 0.001 * scale if you need simplification
            merge_touching=True,
            dry_run=False,
        )
        print("Done.", flush=True)
    except Exception as e:
        print("ERROR:", e, flush=True)
        traceback.print_exc()
    finally:
        # Keep the console open if you run without debugging
        try:
            if os.environ.get("VSCODE_DEBUG_MODE") is None:
                input("Press Enter to close...")
        except Exception:
            pass
