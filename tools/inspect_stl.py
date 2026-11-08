#!/usr/bin/env python3
"""Inspect an STL file: size, triangle count, volume, and a sanity check.

Written for the SolidWorks → Unity pipeline.  Before a part goes into the hand
model there are three things worth knowing, and all three are cheap to check:

* **Do the dimensions make sense?**  CAD exports are usually in millimetres but
  sometimes in metres or inches, and a factor-of-1000 error is invisible in a
  viewer until the part is next to something else.
* **How many triangles?**  A tessellated CAD part can carry hundreds of
  thousands of them; twelve of those in one Unity scene will not hold frame
  rate on a laptop.
* **Is it closed?**  A watertight mesh has a well-defined volume.  An open one
  does not, which is a hint that the export is a surface patch rather than the
  solid that was modelled.

Usage::

    python tools/inspect_stl.py part.stl
    python tools/inspect_stl.py parts/*.stl
    python tools/inspect_stl.py part.stl --obj part.obj     # convert for Unity

No third-party dependency: STL is a simple enough format to read directly, and
this way the tool works on a machine that has never installed a 3D library.
"""

import argparse
import struct
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# A hand part is tens of millimetres.  Bounds chosen wide enough to accept a
# palm-sized plate and reject a metres-vs-millimetres mistake.
PLAUSIBLE_MIN_MM = 1.0
PLAUSIBLE_MAX_MM = 400.0

# Above this, a part is likely to hurt frame rate with several of them in scene.
TRIANGLE_WARN = 200_000
TRIANGLE_HIGH = 500_000


def read_stl(path: Path) -> np.ndarray:
    """Read a binary or ASCII STL and return an (N, 3, 3) float array.

    Detecting the format by the ``solid`` keyword alone is unreliable: several
    exporters write binary files whose 80-byte header happens to start with
    "solid".  The size check is authoritative -- a binary file is exactly
    ``84 + 50 * n`` bytes -- so that is tried first.
    """
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path.name}: too small to be an STL file")

    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * triangle_count
    if expected == len(data):
        return _read_binary(data, triangle_count)

    text = data.decode("ascii", errors="replace")
    if "facet" in text and "vertex" in text:
        return _read_ascii(text)

    raise ValueError(
        f"{path.name}: not a recognisable STL "
        f"({len(data)} bytes, header claims {triangle_count} triangles "
        f"which would need {expected} bytes)"
    )


def _read_binary(data: bytes, triangle_count: int) -> np.ndarray:
    # Each record: 3 floats normal, 9 floats vertices, 2 bytes attribute.
    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("vertices", "<f4", (3, 3)),
        ("attribute", "<u2"),
    ])
    records = np.frombuffer(data, dtype=dtype, count=triangle_count, offset=84)
    return records["vertices"].astype(np.float64)


def _read_ascii(text: str) -> np.ndarray:
    vertices: List[Tuple[float, float, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.lower().startswith("vertex"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            raise ValueError(f"malformed ASCII vertex line: {line!r}")
        vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if not vertices or len(vertices) % 3 != 0:
        raise ValueError(
            f"ASCII STL has {len(vertices)} vertices, not a multiple of 3"
        )
    return np.array(vertices, dtype=np.float64).reshape(-1, 3, 3)


def analyse(triangles: np.ndarray) -> dict:
    """Compute the numbers worth reporting about a triangle soup."""
    if triangles.size == 0:
        raise ValueError("mesh has no triangles")

    points = triangles.reshape(-1, 3)
    finite = np.isfinite(points).all()
    if not finite:
        raise ValueError("mesh contains non-finite coordinates")

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    size = maximum - minimum

    # Signed tetrahedron volumes from the origin.  Only meaningful when the
    # surface is closed, which is why the caller gets the raw value plus the
    # triangle count rather than a confident "volume".
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    signed = np.einsum("ij,ij->i", a, np.cross(b, c))
    volume = float(signed.sum() / 6.0)

    area = float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2.0)

    return {
        "triangles": int(len(triangles)),
        "bbox_min": minimum,
        "bbox_max": maximum,
        "size": size,
        # |volume| is correct whichever way the normals face; the sign tells
        # you the winding, not the size.
        "volume": abs(volume),
        "area": area,
        "centroid": points.mean(axis=0),
    }


def _format(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def report(path: Path, stats: dict) -> List[str]:
    """Print a report and return a list of warnings found."""
    warnings: List[str] = []
    size = stats["size"]
    longest = float(size.max())
    smallest = float(size.min())

    print(f"\n{path.name}")
    print(f"  triangles      {stats['triangles']:,}")
    print(f"  size (units)   {_format(size[0])} x {_format(size[1])} x {_format(size[2])}")
    print(f"  longest edge   {_format(longest)}")
    print(f"  volume         {_format(stats['volume'])} cubic units")
    print(f"  surface area   {_format(stats['area'])}")

    # Units check.  A part outside this band is almost certainly in the wrong
    # unit rather than genuinely that size.
    if longest < PLAUSIBLE_MIN_MM:
        warnings.append(
            f"longest dimension is {_format(longest)} -- probably metres or "
            f"inches, not millimetres. Unity will show it 1000x too small."
        )
    elif longest > PLAUSIBLE_MAX_MM:
        warnings.append(
            f"longest dimension is {_format(longest)} -- probably not "
            f"millimetres. Unity will show it far too large."
        )

    # A dimension of exactly zero means the part is flat: either a 2D sketch or
    # a failed export.
    if smallest < 1e-6:
        warnings.append("one dimension is zero -- the part is flat, not solid")

    if stats["triangles"] > TRIANGLE_HIGH:
        warnings.append(
            f"{stats['triangles']:,} triangles is very heavy; twelve such parts "
            f"will not hold frame rate. Decimate before importing."
        )
    elif stats["triangles"] > TRIANGLE_WARN:
        warnings.append(
            f"{stats['triangles']:,} triangles is heavy for a mechanical part; "
            f"consider decimating."
        )

    # Volume smaller than the bounding box by a huge factor suggests a shell or
    # a surface patch rather than a solid.
    bbox_volume = float(np.prod(np.maximum(size, 1e-9)))
    if bbox_volume > 0 and stats["volume"] / bbox_volume < 1e-4:
        warnings.append(
            "volume is negligible against the bounding box -- the mesh may be "
            "an open surface rather than a closed solid"
        )

    for warning in warnings:
        print(f"  [warn] {warning}")
    return warnings


def export_obj(triangles: np.ndarray, path: Path) -> None:
    """Write the mesh as OBJ, which Unity can import directly.

    Unity does not read STL.  OBJ is the least-lossy simple format to hand it:
    plain text, positions and faces only, no materials to go missing.
    """
    points = triangles.reshape(-1, 3)
    # Deduplicate shared vertices: STL repeats every corner, and importing a
    # mesh with three times the necessary vertices bloats it for no reason.
    unique, inverse = np.unique(np.round(points, 6), axis=0, return_inverse=True)

    with path.open("w", encoding="ascii") as handle:
        handle.write(f"# converted from STL, {len(triangles)} triangles\n")
        for x, y, z in unique:
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        faces = inverse.reshape(-1, 3) + 1
        for a, b, c in faces:
            handle.write(f"f {a} {b} {c}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and convert STL files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--obj", type=Path, default=None,
                        help="write an OBJ next to the input (single input only)")
    args = parser.parse_args()

    if args.obj is not None and len(args.files) != 1:
        print("[FAIL] --obj takes exactly one input file")
        return 1

    failures = 0
    for path in args.files:
        if not path.exists():
            print(f"[FAIL] {path}: not found")
            failures += 1
            continue
        try:
            triangles = read_stl(path)
            stats = analyse(triangles)
        except (ValueError, OSError, struct.error) as exc:
            print(f"[FAIL] {path}: {exc}")
            failures += 1
            continue

        report(path, stats)
        if args.obj is not None:
            export_obj(triangles, args.obj)
            print(f"  wrote {args.obj}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
