#!/usr/bin/env python3
"""Convert a STEP assembly into OBJ files Unity can import.

Unity cannot read STEP, and the individual STL exports are each in their own
local frame -- so neither can be dropped into a scene as-is.  The STEP, on the
other hand, already has every part at its final position, which makes it the
authoritative source for the assembled hand.

What this produces
------------------
* one ``.obj`` holding the whole assembled hand, with an ``o`` object per part
  so Unity splits it into separate meshes, and
* one ``.json`` listing every part's name, position and bounding box, which is
  what the rigging step needs.

Units
-----
The STEP is in **millimetres**.  Unity treats 1 unit as 1 metre, so the model
must be scaled by 0.001 on import.  ``--scale`` does that in the file so the
numbers are right whatever the importer does; the default applies it.

Why tessellation tolerance matters
----------------------------------
A B-rep solid has exact curved surfaces; a mesh approximates them.  Tighter
tolerance means more triangles.  At 0.5 mm the whole hand lands around 80k
triangles, which is comfortable for a laptop; at 0.1 mm it is 250k, which is
not worth it for a part nobody will zoom into.

Usage::

    python tools/step_to_obj.py RUKA_V2_forURDF/RUKA_V2_forURDF.step out/hand

    # writes out/hand.obj and out/hand.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import cadquery as cq
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ShapeTool
    from OCP.TDF import TDF_Label, TDF_LabelSequence
    from OCP.TDataStd import TDataStd_Name
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.IFSelect import IFSelect_RetDone
except ImportError as exc:  # pragma: no cover - environment dependent
    print(f"[FAIL] cadquery/OCP not installed: {exc}")
    print("       pip install cadquery")
    raise SystemExit(1)

DEFAULT_TOLERANCE = 0.5      # mm
DEFAULT_ANGULAR = 0.5        # radians

# Unity works in metres; the STEP is in millimetres.
DEFAULT_SCALE = 0.001


def _label_name(label) -> str:
    attribute = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
        return attribute.Get().ToExtString()
    return "?"


def read_assembly(step_path: Path):
    """Return ``[(name, TopoDS_Shape), ...]`` for every component."""
    app = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("hand"))
    app.NewDocument(TCollection_ExtendedString("hand"), document)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    # ReadFile returns an IFSelect_ReturnStatus enum, not a bool; comparing it
    # to True is always false and silently reports every file as unreadable.
    if reader.ReadFile(str(step_path)) != IFSelect_RetDone:
        raise ValueError(f"could not read {step_path}")
    if not reader.Transfer(document):
        raise ValueError(f"could not transfer {step_path}")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    if roots.Length() == 0:
        raise ValueError("STEP has no root assembly")

    components = TDF_LabelSequence()
    XCAFDoc_ShapeTool.GetComponents_s(roots.Value(1), components)

    parts = []
    for i in range(1, components.Length() + 1):
        component = components.Value(i)
        referred = TDF_Label()
        XCAFDoc_ShapeTool.GetReferredShape_s(component, referred)
        name = _label_name(referred) if not referred.IsNull() else f"part_{i}"
        shape = XCAFDoc_ShapeTool.GetShape_s(component)
        if shape.IsNull():
            continue
        parts.append((name, shape))
    return parts


def tessellate(shape, tolerance: float, angular: float):
    """Return (vertices Nx3, triangles Mx3) for a TopoDS_Shape."""
    wrapped = cq.Shape.cast(shape)
    vertices, triangles = wrapped.tessellate(tolerance, angular)
    return vertices, triangles


def bounds(shape):
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (
        [xmin, ymin, zmin],
        [xmax, ymax, zmax],
        [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2],
    )


def write_single_part(vertices, triangles, scale: float, path: Path) -> None:
    """Write one part as its own OBJ.

    Unity's OBJ importer merges the ``o`` groups of a single file into one mesh
    by default, so a per-part file is the only reliable way to get one
    GameObject per rigid body without touching importer settings.
    """
    with path.open("w", encoding="ascii") as handle:
        handle.write("# one rigid part, mm converted by scale\n")
        for vertex in vertices:
            handle.write(f"v {vertex.x * scale:.6f} {vertex.y * scale:.6f} "
                         f"{vertex.z * scale:.6f}\n")
        for triangle in triangles:
            a, b, c = triangle[0] + 1, triangle[1] + 1, triangle[2] + 1
            handle.write(f"f {a} {b} {c}\n")


def convert(step_path: Path, out_stem: Path, tolerance: float, angular: float,
            scale: float, parts_dir: Path = None) -> int:
    started = time.time()
    parts = read_assembly(step_path)
    print(f"parts: {len(parts)}")

    obj_path = out_stem.with_suffix(".obj")
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    if parts_dir is not None:
        parts_dir.mkdir(parents=True, exist_ok=True)

    total_triangles = 0
    manifest = []

    with obj_path.open("w", encoding="ascii") as handle:
        handle.write(f"# assembled from {step_path.name}\n")
        handle.write(f"# scale {scale} (STEP mm -> output units)\n")
        vertex_offset = 1   # OBJ indices are 1-based

        for index, (name, shape) in enumerate(parts, start=1):
            try:
                vertices, triangles = tessellate(shape, tolerance, angular)
            except Exception as exc:  # a single bad solid must not abort the rest
                print(f"  [{index:>2}] {name:<24} SKIPPED ({exc})")
                continue

            if not triangles:
                print(f"  [{index:>2}] {name:<24} no triangles")
                continue

            safe_name = name.replace(" ", "_")
            handle.write(f"\no {safe_name}_{index}\n")
            for vertex in vertices:
                handle.write(f"v {vertex.x * scale:.6f} {vertex.y * scale:.6f} "
                             f"{vertex.z * scale:.6f}\n")
            for triangle in triangles:
                a, b, c = (vertex_offset + i for i in triangle)
                handle.write(f"f {a} {b} {c}\n")
            vertex_offset += len(vertices)

            # Unity's OBJ importer merges the `o` groups into a single mesh by
            # default, so a 34-object OBJ arrives as one blob.  Writing one
            # file per part sidesteps the importer settings entirely: each file
            # becomes its own GameObject, which is what the rig needs.
            if parts_dir is not None:
                write_single_part(vertices, triangles, scale,
                                  parts_dir / f"{safe_name}_{index}.obj")

            total_triangles += len(triangles)

            low, high, centre = bounds(shape)
            manifest.append({
                "index": index,
                "name": name,
                "triangles": len(triangles),
                # Positions are recorded in millimetres: the rig is easier to
                # reason about in the units the designer used.
                "bounds_min_mm": [round(v, 3) for v in low],
                "bounds_max_mm": [round(v, 3) for v in high],
                "centre_mm": [round(v, 3) for v in centre],
                "part_file": f"{safe_name}_{index}.obj" if parts_dir else None,
            })

            print(f"  [{index:>2}] {name:<24} {len(triangles):>6,} tris")

    json_path = out_stem.with_suffix(".json")
    json_path.write_text(json.dumps({
        "source": step_path.name,
        "units": "millimetres",
        "scale_applied": scale,
        "tolerance_mm": tolerance,
        "angular_rad": angular,
        "total_triangles": total_triangles,
        "parts": manifest,
    }, indent=2), encoding="utf-8")

    elapsed = time.time() - started
    print(f"\n[OK] {obj_path}  ({total_triangles:,} triangles, {elapsed:.1f}s)")
    print(f"[OK] {json_path}  ({len(manifest)} parts)")
    if scale == DEFAULT_SCALE:
        print("     scale 0.001 applied: the model is now in Unity metres")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a STEP assembly to OBJ for Unity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("step", type=Path)
    parser.add_argument("out", type=Path, help="output stem, e.g. out/hand")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help="linear tessellation tolerance, mm (default 0.5)")
    parser.add_argument("--angular", type=float, default=DEFAULT_ANGULAR,
                        help="angular tessellation tolerance, radians (default 0.5)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                        help="metres per STEP unit (default 0.001 for mm->m)")
    parser.add_argument("--no-scale", action="store_true",
                        help="leave the model in millimetres")
    parser.add_argument("--parts-dir", type=Path, default=None,
                        help="also write one OBJ per part into this directory "
                             "(needed for Unity: its OBJ importer merges the "
                             "object groups of a single file)")
    args = parser.parse_args()

    if not args.step.exists():
        print(f"[FAIL] {args.step}: not found")
        return 1

    scale = 1.0 if args.no_scale else args.scale
    try:
        return convert(args.step, args.out, args.tolerance, args.angular, scale,
                       args.parts_dir)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
