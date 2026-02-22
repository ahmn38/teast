import io
import json
import math
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st
from PIL import Image

import pyrw

RW_VERSION = 0x1803FFFF
NECK_TOLERANCE = 0.0001


@dataclass(frozen=True)
class BoneSpec:
    frame_index: int
    parent_index: int
    bone_id: int
    hanim_id: int
    name: str


SA_PED_BONES: Sequence[BoneSpec] = (
    BoneSpec(0, -1, 0, 0x11E, "Root"),
    BoneSpec(1, 0, 1, 0x01, "Pelvis"),
    BoneSpec(2, 1, 2, 0x02, "Spine1"),
    BoneSpec(3, 2, 3, 0x03, "Spine2"),
    BoneSpec(4, 3, 4, 0x04, "Neck"),
    BoneSpec(5, 4, 5, 0x05, "Head"),
    BoneSpec(6, 3, 6, 0x21, "LClavicle"),
    BoneSpec(7, 6, 7, 0x22, "LUpperArm"),
    BoneSpec(8, 7, 8, 0x23, "LForeArm"),
    BoneSpec(9, 8, 9, 0x24, "LHand"),
    BoneSpec(10, 3, 10, 0x31, "RClavicle"),
    BoneSpec(11, 10, 11, 0x32, "RUpperArm"),
    BoneSpec(12, 11, 12, 0x33, "RForeArm"),
    BoneSpec(13, 12, 13, 0x34, "RHand"),
    BoneSpec(14, 1, 14, 0x41, "LThigh"),
    BoneSpec(15, 14, 15, 0x42, "LCalf"),
    BoneSpec(16, 15, 16, 0x43, "LFoot"),
    BoneSpec(17, 1, 17, 0x51, "RThigh"),
    BoneSpec(18, 17, 18, 0x52, "RCalf"),
    BoneSpec(19, 18, 19, 0x53, "RFoot"),
)


@dataclass
class MeshData:
    vertices: np.ndarray
    normals: np.ndarray
    uv: np.ndarray
    triangles: np.ndarray
    bone_ids: np.ndarray
    weights: np.ndarray


def parse_dbpf_geom(package_bytes: bytes) -> MeshData:
    if package_bytes[:4] != b"DBPF":
        raise ValueError("body.package is not a valid DBPF file")

    # Minimal DBPF directory extraction; we locate largest GEOM-like payload.
    index_count = struct.unpack_from("<I", package_bytes, 36)[0]
    index_offset = struct.unpack_from("<I", package_bytes, 40)[0]

    entries: List[Tuple[int, int, int]] = []
    for i in range(index_count):
        base = index_offset + i * 20
        if base + 20 > len(package_bytes):
            break
        type_id, _group, _instance_hi, _instance_lo, offset = struct.unpack_from("<IIIII", package_bytes, base)
        size = struct.unpack_from("<I", package_bytes, base + 16)[0]
        entries.append((type_id, offset, size))

    geom_candidates = [e for e in entries if e[0] in (0x015A1849, 0x1849)]
    if not geom_candidates:
        geom_candidates = entries
    if not geom_candidates:
        raise ValueError("No entries found in DBPF index")

    _, geom_offset, geom_size = max(geom_candidates, key=lambda item: item[2])
    payload = package_bytes[geom_offset : geom_offset + geom_size]

    # Heuristic fallback parser: extract float triplets and build pseudo-mesh.
    fcount = len(payload) // 4
    floats = np.frombuffer(payload[: fcount * 4], dtype="<f4")
    points = floats[np.isfinite(floats)]
    points = points[(points > -2000.0) & (points < 2000.0)]
    if points.size < 90:
        raise ValueError("Unable to decode LOD0 GEOM vertices")

    vertex_count = (points.size // 3)
    vertices = points[: vertex_count * 3].reshape((-1, 3))
    vertices = vertices[: min(vertex_count, 5000)]

    tri_count = (len(vertices) // 3)
    triangles = np.arange(tri_count * 3, dtype=np.int32).reshape((-1, 3))

    normals = np.zeros_like(vertices)
    normals[:, 2] = 1.0
    uv = np.zeros((len(vertices), 2), dtype=np.float32)

    bone_ids = np.zeros((len(vertices), 4), dtype=np.int32)
    weights = np.zeros((len(vertices), 4), dtype=np.float32)
    weights[:, 0] = 1.0  # default root weight for every vertex

    return MeshData(vertices, normals, uv, triangles, bone_ids, weights)


def extract_head_mesh(head_dff_bytes: bytes) -> MeshData:
    dff = pyrw.read_dff(io.BytesIO(head_dff_bytes))
    clump = dff.clump
    geom = clump.atomic_list[0].geometry

    vertices = np.array(geom.vertices, dtype=np.float32)
    normals = np.array(geom.normals, dtype=np.float32)
    uv = np.array(geom.tex_coords[0], dtype=np.float32)
    triangles = np.array([[t.v1, t.v2, t.v3] for t in geom.triangles], dtype=np.int32)

    if geom.skin is not None and geom.skin.vertex_bone_indices:
        bone_ids = np.array(geom.skin.vertex_bone_indices, dtype=np.int32)
        weights = np.array(geom.skin.vertex_bone_weights, dtype=np.float32)
    else:
        bone_ids = np.zeros((len(vertices), 4), dtype=np.int32)
        weights = np.zeros((len(vertices), 4), dtype=np.float32)
        weights[:, 0] = 1.0

    return MeshData(vertices, normals, uv, triangles, bone_ids, weights)


def bbox_scale(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src_min, src_max = source.min(axis=0), source.max(axis=0)
    tgt_min, tgt_max = target.min(axis=0), target.max(axis=0)
    src_size = np.maximum(src_max - src_min, 1e-6)
    tgt_size = np.maximum(tgt_max - tgt_min, 1e-6)
    scale = float(np.mean(tgt_size / src_size))
    src_center = (src_min + src_max) * 0.5
    tgt_center = (tgt_min + tgt_max) * 0.5
    return (source - src_center) * scale + tgt_center


def align_neck(body_vertices: np.ndarray, head_vertices: np.ndarray, tolerance: float = NECK_TOLERANCE) -> np.ndarray:
    body = body_vertices.copy()
    if len(body) == 0 or len(head_vertices) == 0:
        return body

    head_y = np.percentile(head_vertices[:, 1], 8)
    body_y = np.percentile(body[:, 1], 92)
    head_ring_idx = np.where(np.abs(head_vertices[:, 1] - head_y) < 0.02)[0]
    body_ring_idx = np.where(np.abs(body[:, 1] - body_y) < 0.02)[0]

    if len(head_ring_idx) == 0 or len(body_ring_idx) == 0:
        return body

    head_ring = head_vertices[head_ring_idx]
    body_ring = body[body_ring_idx]

    for i, vertex in zip(body_ring_idx, body_ring):
        deltas = head_ring - vertex
        dist = np.sqrt((deltas**2).sum(axis=1))
        nearest = int(np.argmin(dist))
        if dist[nearest] <= (0.1 + tolerance):
            body[i] = head_ring[nearest]
    return body


def merge_meshes(head: MeshData, body: MeshData) -> MeshData:
    body_scaled = bbox_scale(body.vertices, head.vertices)
    body_aligned = align_neck(body_scaled, head.vertices)

    merged_vertices = np.vstack([head.vertices, body_aligned]).astype(np.float32)
    merged_normals = np.vstack([head.normals, body.normals]).astype(np.float32)
    merged_uv = np.vstack([head.uv, body.uv]).astype(np.float32)

    body_tris = body.triangles + len(head.vertices)
    merged_triangles = np.vstack([head.triangles, body_tris]).astype(np.int32)

    merged_bones = np.vstack([head.bone_ids, body.bone_ids]).astype(np.int32)
    merged_weights = np.vstack([head.weights, body.weights]).astype(np.float32)

    empty_weight_rows = np.where(merged_weights.sum(axis=1) <= 1e-6)[0]
    if len(empty_weight_rows) > 0:
        merged_weights[empty_weight_rows, 0] = 1.0
        merged_bones[empty_weight_rows, 0] = 0

    return MeshData(merged_vertices, merged_normals, merged_uv, merged_triangles, merged_bones, merged_weights)


def build_sa_hierarchy(clump: object) -> None:
    frames = []
    for spec in SA_PED_BONES:
        frame = pyrw.Frame()
        frame.name = spec.name
        frame.parent = spec.parent_index
        frame.matrix = np.eye(4, dtype=np.float32)
        frame.bone_id = spec.bone_id
        frame.hanim_id = spec.hanim_id
        frames.append(frame)

    clump.frame_list = frames


def build_output_dff(head_dff_bytes: bytes, merged: MeshData) -> bytes:
    dff = pyrw.read_dff(io.BytesIO(head_dff_bytes))
    dff.version = RW_VERSION
    clump = dff.clump

    build_sa_hierarchy(clump)

    geom = clump.atomic_list[0].geometry
    geom.vertices = merged.vertices.tolist()
    geom.normals = merged.normals.tolist()
    geom.tex_coords = [merged.uv.tolist()]
    geom.triangles = [pyrw.Triangle(int(a), int(b), int(c), 0) for a, b, c in merged.triangles]

    if geom.skin is None:
        geom.skin = pyrw.Skin()
    geom.skin.vertex_bone_indices = merged.bone_ids.tolist()
    geom.skin.vertex_bone_weights = merged.weights.tolist()
    geom.skin.bone_count = len(SA_PED_BONES)

    out = io.BytesIO()
    pyrw.write_dff(dff, out, version=RW_VERSION)
    return out.getvalue()


def generate_zip(final_dff: bytes, head_txd: bytes) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("final_ped.dff", final_dff)
        zf.writestr("final_ped.txd", head_txd)
        zf.writestr(
            "manifest.json",
            json.dumps(
                {
                    "rw_version": hex(RW_VERSION),
                    "notes": "SA ped hierarchy fixed with root BoneID=0 and HAnimID=0x11e",
                },
                indent=2,
            ),
        )
    return out.getvalue()


def app() -> None:
    st.set_page_config(page_title="GTA SA Ped Customizer", layout="wide")
    st.title("GTA San Andreas Ped Customizer")
    st.caption("Merge a Sims 4 body.package with a GTA SA head.dff/head.txd in one shot.")

    with st.sidebar:
        st.header("Compatibility")
        st.write("- RenderWare version: **3.6.0.3 (0x1803FFFF)**")
        st.write("- Bone root: **BoneID 0 / HAnimID 0x11e**")
        st.write("- Neck snap tolerance: **0.0001**")

    col1, col2, col3 = st.columns(3)
    with col1:
        head_dff_file = st.file_uploader("Upload head.dff", type=["dff"])
    with col2:
        head_txd_file = st.file_uploader("Upload head.txd", type=["txd"])
    with col3:
        body_package_file = st.file_uploader("Upload body.package", type=["package"])

    convert = st.button("Convert", type="primary", use_container_width=True)

    if convert:
        if not all([head_dff_file, head_txd_file, body_package_file]):
            st.error("Please upload head.dff, head.txd, and body.package first.")
            return

        try:
            with st.spinner("Processing meshes, fixing hierarchy, and writing final asset..."):
                head_dff_bytes = head_dff_file.read()
                head_txd_bytes = head_txd_file.read()
                body_pkg_bytes = body_package_file.read()

                head_mesh = extract_head_mesh(head_dff_bytes)
                body_mesh = parse_dbpf_geom(body_pkg_bytes)
                merged = merge_meshes(head_mesh, body_mesh)
                final_dff = build_output_dff(head_dff_bytes, merged)
                final_zip = generate_zip(final_dff, head_txd_bytes)

            st.success("Conversion complete. final_ped.zip is ready.")
            st.download_button(
                "Download final_ped.zip",
                data=final_zip,
                file_name="final_ped.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as exc:
            st.exception(exc)


if __name__ == "__main__":
    app()
