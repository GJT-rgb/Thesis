from typing import Dict, List, Tuple, Iterable, Optional
import gdstk
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import ezdxf
from ezdxf import units as dxf_units


def gds_to_merged_dxf(
    input_gds = r"E:/EMBL/Master EMBL/Python/polyline dxf/Cardio_projection3D.gds",
    output_dxf = r"E:/EMBL/Master EMBL/Python/polyline dxf/Cardio_projection3D_polylined.dxf",
    
    layers: Optional[Iterable[Tuple[int, int]]] = None,
    dxf_units_mm: bool = True,
    scale: float = 1,
    simplify: float = 0.0,
    merge_touching: bool = True,
) -> None:
    """
    Read a GDS, merge touching/overlapping polygons per (layer, datatype),
    preserve interior holes, and save as DXF with closed polylines.

    Args:
        input_gds: Path to input .gds
        output_dxf: Path to output .dxf
        layers: Optional iterable of (layer, datatype) to include; default = all found.
        dxf_units_mm: If True, set DXF INSUNITS to millimeters.
        scale: Coordinate scale factor applied when writing DXF.
               Example: if GDS coordinates are microns and you want DXF in mm, use 0.001.
        simplify: Douglas–Peucker simplify tolerance in *DXF units after scaling* (0 disables).
        merge_touching: If True, polygons that touch at edges/points are merged.
    """
    # ---------- helpers ----------
    def _as_polygon(points):
        if not points or len(points) < 3:
            return None
        if points[0][0] == points[-1][0] and points[0][1] == points[-1][1]:
            pts = points[:-1]
        else:
            pts = points
        if len(pts) < 3:
            return None
        return Polygon(pts)

    def _collect_polys(lib):
        out: Dict[Tuple[int, int], List[Polygon]] = {}
        for cell in lib.cells:
            for ply in cell.polygons:
                lay, dt = ply.layer, ply.datatype
                if layers and (lay, dt) not in layers:
                    continue
                poly = _as_polygon([tuple(p) for p in ply.points])
                if poly and not poly.is_empty:
                    out.setdefault((lay, dt), []).append(poly)
            for path in cell.paths:
                lay, dt = path.layer, path.datatype
                if layers and (lay, dt) not in layers:
                    continue
                for pts in path.to_polygons():
                    poly = _as_polygon([tuple(p) for p in pts])
                    if poly and not poly.is_empty:
                        out.setdefault((lay, dt), []).append(poly)
        return out

    def _robust_union(polys: List[Polygon]):
        if not polys:
            return MultiPolygon()
        fixed = []
        for p in polys:
            if not p.is_valid:
                p = make_valid(p)
            p = p.buffer(0)
            if not p.is_empty:
                fixed.append(p)
        if not fixed:
            return MultiPolygon()
        if merge_touching:
            merged = unary_union(fixed)
        else:
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

    # ---------- read GDS ----------
    lib = gdstk.read_gds(input_gds)
    layer_geoms = _collect_polys(lib)
    if not layer_geoms:
        raise ValueError("No polygons/paths found in the requested layers.")

    # ---------- prepare DXF ----------
    doc = ezdxf.new("R2018")
    if dxf_units_mm:
        doc.units = dxf_units.MM
    msp = doc.modelspace()

    # ---------- merge and write ----------
    for (lay, dt), polys in layer_geoms.items():
        merged = _robust_union(polys)
        if merged.is_empty:
            continue

        if isinstance(merged, Polygon):
            poly_iter = [merged]
        else:
            poly_iter = [
                g for g in getattr(merged, "geoms", []) if isinstance(g, Polygon)
            ]

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

    # ---------- save ----------
    doc.saveas(output_dxf)
    print(f"Saved merged DXF with polylines → {output_dxf}")
