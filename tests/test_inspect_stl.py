"""Tests for the STL inspection tool.

The meshes are generated here rather than checked in: a known cube has an exact
volume and surface area, so the parser can be verified against arithmetic
instead of against a stored answer.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.inspect_stl import (
    PLAUSIBLE_MAX_MM,
    PLAUSIBLE_MIN_MM,
    TRIANGLE_WARN,
    analyse,
    export_obj,
    read_stl,
    report,
)

CUBE_VERTICES = [
    (0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0),
    (0, 0, 10), (10, 0, 10), (10, 10, 10), (0, 10, 10),
]
CUBE_FACES = [
    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
    (0, 1, 5), (0, 5, 4), (3, 7, 6), (3, 6, 2),
    (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5),
]


def _binary_stl(path: Path, scale: float = 1.0) -> Path:
    buffer = bytearray(b"test cube".ljust(80, b"\0"))
    buffer += struct.pack("<I", len(CUBE_FACES))
    for face in CUBE_FACES:
        buffer += struct.pack("<3f", 0, 0, 0)
        for index in face:
            x, y, z = CUBE_VERTICES[index]
            buffer += struct.pack("<3f", x * scale, y * scale, z * scale)
        buffer += struct.pack("<H", 0)
    path.write_bytes(bytes(buffer))
    return path


def _ascii_stl(path: Path) -> Path:
    lines = ["solid cube"]
    for face in CUBE_FACES:
        lines += ["  facet normal 0 0 0", "    outer loop"]
        lines += [f"      vertex {CUBE_VERTICES[i][0]} {CUBE_VERTICES[i][1]} "
                  f"{CUBE_VERTICES[i][2]}" for i in face]
        lines += ["    endloop", "  endfacet"]
    lines.append("endsolid cube")
    path.write_text("\n".join(lines), encoding="ascii")
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_binary_stl_is_read(tmp_path):
    mesh = read_stl(_binary_stl(tmp_path / "cube.stl"))

    assert mesh.shape == (12, 3, 3)


def test_ascii_stl_is_read(tmp_path):
    mesh = read_stl(_ascii_stl(tmp_path / "cube.stl"))

    assert mesh.shape == (12, 3, 3)


def test_binary_and_ascii_agree(tmp_path):
    """Both encodings describe the same solid; they must decode the same."""
    binary = read_stl(_binary_stl(tmp_path / "b.stl"))
    ascii_ = read_stl(_ascii_stl(tmp_path / "a.stl"))

    assert np.allclose(binary, ascii_)


def test_binary_detection_does_not_trust_the_header_word(tmp_path):
    """Some exporters write binary files starting with 'solid'.

    Only the byte count is authoritative, so a file whose header looks like
    ASCII must still be read as binary.
    """
    path = tmp_path / "sneaky.stl"
    _binary_stl(path)
    data = bytearray(path.read_bytes())
    data[0:16] = b"solid sneaky    "
    path.write_bytes(bytes(data))

    mesh = read_stl(path)

    assert mesh.shape == (12, 3, 3)


def test_a_truncated_file_is_rejected(tmp_path):
    path = _binary_stl(tmp_path / "cube.stl")
    path.write_bytes(path.read_bytes()[:-17])

    with pytest.raises(ValueError):
        read_stl(path)


def test_a_non_stl_file_is_rejected(tmp_path):
    path = tmp_path / "not_a_mesh.stl"
    path.write_text("just some text\n" * 20, encoding="ascii")

    with pytest.raises(ValueError):
        read_stl(path)


def test_an_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.stl"
    path.write_bytes(b"")

    with pytest.raises(ValueError):
        read_stl(path)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def test_a_ten_millimetre_cube_measures_correctly(tmp_path):
    stats = analyse(read_stl(_binary_stl(tmp_path / "cube.stl")))

    assert stats["triangles"] == 12
    assert np.allclose(stats["size"], [10, 10, 10])
    assert stats["volume"] == pytest.approx(1000.0)
    assert stats["area"] == pytest.approx(600.0)


def test_volume_is_reported_positive_whatever_the_winding(tmp_path):
    """A mesh with inverted normals is still 1000 cubic units of material."""
    mesh = read_stl(_binary_stl(tmp_path / "cube.stl"))
    flipped = mesh[:, ::-1, :]     # reverse each triangle's winding

    assert analyse(flipped)["volume"] == pytest.approx(1000.0)


def test_a_scaled_mesh_scales_cubically(tmp_path):
    stats = analyse(read_stl(_binary_stl(tmp_path / "cube.stl", scale=2.0)))

    assert np.allclose(stats["size"], [20, 20, 20])
    assert stats["volume"] == pytest.approx(8000.0)
    assert stats["area"] == pytest.approx(2400.0)


def test_non_finite_coordinates_are_rejected(tmp_path):
    """Parsing accepts the bytes; analysis is what refuses to trust them.

    read_stl is a parser and stays one -- it decodes what is on disk.  The
    rejection belongs in analyse, because a NaN coordinate is a *geometric*
    problem and every caller that wants numbers goes through analyse.
    """
    path = tmp_path / "nan.stl"
    _binary_stl(path)
    data = bytearray(path.read_bytes())
    struct.pack_into("<f", data, 84 + 12, float("nan"))
    path.write_bytes(bytes(data))

    with pytest.raises(ValueError):
        analyse(read_stl(path))


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def test_a_plausible_part_warns_about_nothing(tmp_path, capsys):
    path = _binary_stl(tmp_path / "cube.stl")

    warnings = report(path, analyse(read_stl(path)))

    assert warnings == []


def test_a_metre_scale_part_is_flagged(tmp_path):
    """A 0.01-unit part is almost certainly metres, not millimetres."""
    path = _binary_stl(tmp_path / "tiny.stl", scale=0.001)

    warnings = report(path, analyse(read_stl(path)))

    assert any("metres" in w for w in warnings)


def test_an_oversized_part_is_flagged(tmp_path):
    path = _binary_stl(tmp_path / "huge.stl", scale=100.0)

    warnings = report(path, analyse(read_stl(path)))

    assert any("too large" in w for w in warnings)


def test_a_flat_part_is_flagged(tmp_path):
    path = tmp_path / "flat.stl"
    _binary_stl(path)
    data = bytearray(path.read_bytes())
    # Zero every z coordinate: a flat sheet rather than a solid.
    for triangle in range(12):
        base = 84 + triangle * 50
        for vertex in range(3):
            struct.pack_into("<f", data, base + 12 + vertex * 12 + 8, 0.0)
    path.write_bytes(bytes(data))

    warnings = report(path, analyse(read_stl(path)))

    assert any("flat" in w for w in warnings)


def test_unit_boundaries_are_sane():
    """A finger phalanx is ~20-45 mm and a palm plate ~100 mm."""
    assert PLAUSIBLE_MIN_MM <= 20 <= PLAUSIBLE_MAX_MM
    assert PLAUSIBLE_MIN_MM <= 100 <= PLAUSIBLE_MAX_MM
    # But a 0.02 (metres) or 2000 (microns) reading must fall outside.
    assert not (PLAUSIBLE_MIN_MM <= 0.02 <= PLAUSIBLE_MAX_MM)
    assert not (PLAUSIBLE_MIN_MM <= 2000 <= PLAUSIBLE_MAX_MM)


def test_triangle_warning_threshold_is_below_the_hard_limit():
    assert 0 < TRIANGLE_WARN


# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------

def test_obj_export_writes_a_readable_mesh(tmp_path):
    mesh = read_stl(_binary_stl(tmp_path / "cube.stl"))
    obj = tmp_path / "cube.obj"

    export_obj(mesh, obj)

    text = obj.read_text(encoding="ascii")
    vertices = [l for l in text.splitlines() if l.startswith("v ")]
    faces = [l for l in text.splitlines() if l.startswith("f ")]
    assert len(faces) == 12
    # Shared corners are deduplicated: a cube has 8 unique vertices, not 36.
    assert len(vertices) == 8


def test_obj_export_indexes_from_one(tmp_path):
    """OBJ indices are 1-based; 0 would silently corrupt the mesh."""
    mesh = read_stl(_binary_stl(tmp_path / "cube.stl"))
    obj = tmp_path / "cube.obj"

    export_obj(mesh, obj)

    for line in obj.read_text(encoding="ascii").splitlines():
        if line.startswith("f "):
            indices = [int(part) for part in line.split()[1:]]
            assert all(i >= 1 for i in indices)
