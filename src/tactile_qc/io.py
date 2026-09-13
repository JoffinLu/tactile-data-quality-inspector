"""RCT tactile data loading.

The RCT dataset (Robotic Contact Tactile) is organised as::

    rct_dataset/
    ├── material_categories.json          # material id -> top-level category
    └── materials/
        └── material_<id>/
            ├── tactile_data/
            │   └── position_<P>/
            │       └── sensor_<S>/
            │           ├── contact_frames/depth_<value>.png
            │           └── metadata.json
            ├── force_data/
            │   └── position_<P>_sensor_<S>.txt   # 6-axis F/T trace
            └── frame/
                └── sensor_<S>.png                # per-sensor background frame

A *sequence* (a.k.a. trajectory) is one press of a (material, position,
sensor) combination. Expected totals (per the dataset README): 122
materials, 1,832 sequences, 29,279 contact frames and 1,827 force traces.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

PathLike = Union[str, Path]

#: ``depth_106.0.png`` and ``depth_106.png`` are both valid
DEPTH_RE = re.compile(r"^depth_(\d+(?:\.\d+)?)\.png$")

#: One line of a ``force_data/*.txt`` trace file
FORCE_LINE_RE = re.compile(
    r"z_position-\[([-\d.eE+]+)\].*?"
    r"Position:\s*\[([^\]]*)\].*?"
    r"External Force:\s*\[([^\]]*)\].*?"
    r"Raw Force:\s*\[([^\]]*)\]"
)

FORCE_COLUMNS = ("fx", "fy", "fz", "tx", "ty", "tz")


def _parse_force_file(path: Path):
    """Parse a force trace text file.

    Returns ``(z_positions, robot_poses, ext_forces, raw_forces)`` as numpy
    arrays of shape ``(T,)``, ``(T, 6)``, ``(T, 6)`` and ``(T, 6)``, or
    ``None`` when nothing could be parsed.
    """
    z, poses, exts, raws = [], [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = FORCE_LINE_RE.search(line)
        if m is None:
            continue
        z.append(float(m.group(1)))
        poses.append([float(v) for v in m.group(2).split(",")])
        exts.append([float(v) for v in m.group(3).split(",")])
        raws.append([float(v) for v in m.group(4).split(",")])
    if not z:
        return None
    return (
        np.asarray(z, dtype=np.float64),
        np.asarray(poses, dtype=np.float64),
        np.asarray(exts, dtype=np.float64),
        np.asarray(raws, dtype=np.float64),
    )


@dataclass
class TactileSequence:
    """One press trajectory: (material, position, sensor)."""

    sequence_id: str
    material_id: str
    material_label: str  # top-level category, e.g. "Plastic_Rubber"
    position: int
    sensor: int
    frame_paths: List[Path]  # contact frames, sorted by depth value
    depth_values: np.ndarray  # (N,) tactile depth per frame
    background_path: Optional[Path] = None
    forces: Optional[np.ndarray] = None  # (T, 6) external F/T trace
    raw_forces: Optional[np.ndarray] = None  # (T, 6) raw F/T trace
    z_positions: Optional[np.ndarray] = None  # (T,) robot z during trace
    robot_poses: Optional[np.ndarray] = None  # (T, 6) robot pose during trace
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame_paths)

    @property
    def has_force(self) -> bool:
        return self.forces is not None and len(self.forces) > 0

    def force_magnitude(self) -> Optional[np.ndarray]:
        """External force magnitude ||(fx, fy, fz)|| per time step."""
        if not self.has_force:
            return None
        return np.linalg.norm(self.forces[:, :3], axis=1)

    def load_frames(self, max_frames: Optional[int] = None) -> np.ndarray:
        """Load contact frames as a stacked RGB uint8 array (N, H, W, 3).

        Frames that cannot be decoded (corrupt/truncated PNGs) are skipped
        silently, so the returned array may contain fewer frames than
        ``len(self)``. Use :attr:`frame_paths` for the authoritative count.
        """
        from PIL import Image

        paths = self.frame_paths
        if max_frames is not None:
            paths = paths[:max_frames]
        arrays = []
        for p in paths:
            try:
                arrays.append(np.asarray(Image.open(p).convert("RGB")))
            except Exception:
                continue  # skip unreadable frame
        if not arrays:
            return np.empty((0, 0, 0, 3), dtype=np.uint8)
        return np.stack(arrays)

    def load_background(self) -> Optional[np.ndarray]:
        """Load the sensor background frame as an RGB uint8 array."""
        from PIL import Image

        if self.background_path is None or not self.background_path.exists():
            return None
        return np.asarray(Image.open(self.background_path).convert("RGB"))


@dataclass
class RctDataset:
    """A loaded RCT dataset: an ordered collection of sequences."""

    data_dir: Path
    sequences: List[TactileSequence]

    def __len__(self) -> int:
        return len(self.sequences)

    def __iter__(self):
        return iter(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def summary_dataframe(self):
        """Per-sequence summary as a ``pandas.DataFrame``."""
        import pandas as pd

        rows = []
        for s in self.sequences:
            force_mag = s.force_magnitude()
            rows.append(
                {
                    "sequence_id": s.sequence_id,
                    "material_id": s.material_id,
                    "material_label": s.material_label,
                    "position": s.position,
                    "sensor": s.sensor,
                    "n_frames": len(s),
                    "depth_min": float(np.min(s.depth_values)),
                    "depth_max": float(np.max(s.depth_values)),
                    "has_force": s.has_force,
                    "force_trace_len": 0 if s.z_positions is None else len(s.z_positions),
                    "force_mag_max": None if force_mag is None else float(np.max(force_mag)),
                }
            )
        return pd.DataFrame(rows)


def load_material_categories(data_dir: PathLike) -> Dict[int, str]:
    """Map material id (as int, leading zeros stripped) -> top category."""
    path = Path(data_dir) / "material_categories.json"
    mapping: Dict[int, str] = {}
    if not path.exists():
        return mapping
    for entry in json.loads(path.read_text(encoding="utf-8")):
        label = entry.get("top_category", "Unknown")
        for key in ("id", "id_short"):
            if key in entry and entry[key]:
                try:
                    mapping[int(entry[key])] = label
                except ValueError:
                    continue
    return mapping


def load_rct_sequences(
    data_dir: PathLike,
    materials: Optional[List[str]] = None,
    skip_missing_force: bool = False,
) -> RctDataset:
    """Load all RCT contact sequences under ``data_dir``.

    Parameters
    ----------
    data_dir : str | Path
        Path to the extracted ``rct_dataset`` directory (the one that
        contains ``materials/`` and ``material_categories.json``).
    materials : list of str, optional
        Restrict loading to these material folder names
        (e.g. ``["material_0710380"]``). ``None`` loads everything.
    skip_missing_force : bool, default False
        Drop sequences without a force trace instead of keeping them
        (``forces=None``).

    Returns
    -------
    RctDataset
        Dataset object; each :class:`TactileSequence` carries
        ``frame_paths``/``depth_values`` (tactile frames), ``forces``
        (6-axis external force/torque trace), ``material_label`` (top
        category) and ``sequence_id``.
    """
    data_dir = Path(data_dir)
    materials_root = data_dir / "materials"
    if not materials_root.is_dir():
        raise FileNotFoundError(f"materials/ not found under: {data_dir}")

    categories = load_material_categories(data_dir)

    material_dirs = sorted(
        d for d in materials_root.iterdir()
        if d.is_dir() and d.name.startswith("material_")
    )
    if materials is not None:
        wanted = set(materials)
        material_dirs = [d for d in material_dirs if d.name in wanted]

    sequences: List[TactileSequence] = []
    for mat_dir in material_dirs:
        material_id = mat_dir.name.removeprefix("material_")
        try:
            material_key = int(material_id)
        except ValueError:
            material_key = -1
        material_label = categories.get(material_key, "Unknown")

        tactile_root = mat_dir / "tactile_data"
        if not tactile_root.is_dir():
            continue

        for pos_dir in sorted(tactile_root.iterdir()):
            m_pos = re.match(r"^position_(\d+)$", pos_dir.name)
            if not m_pos or not pos_dir.is_dir():
                continue
            position = int(m_pos.group(1))

            for sensor_dir in sorted(pos_dir.iterdir()):
                m_sensor = re.match(r"^sensor_(\d+)$", sensor_dir.name)
                if not m_sensor or not sensor_dir.is_dir():
                    continue
                sensor = int(m_sensor.group(1))

                frames_dir = sensor_dir / "contact_frames"
                if not frames_dir.is_dir():
                    continue

                depth_items = []
                for f in frames_dir.iterdir():
                    m_frame = DEPTH_RE.match(f.name)
                    if m_frame:
                        depth_items.append((float(m_frame.group(1)), f))
                if not depth_items:
                    continue
                depth_items.sort(key=lambda item: item[0])

                force_path = (
                    mat_dir / "force_data" / f"position_{position}_sensor_{sensor}.txt"
                )
                force = _parse_force_file(force_path) if force_path.exists() else None

                meta_path = sensor_dir / "metadata.json"
                metadata = {}
                if meta_path.exists():
                    try:
                        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        metadata = {}

                background = mat_dir / "frame" / f"sensor_{sensor}.png"

                seq = TactileSequence(
                    sequence_id=(
                        f"{mat_dir.name}_position_{position}_sensor_{sensor}"
                    ),
                    material_id=material_id,
                    material_label=material_label,
                    position=position,
                    sensor=sensor,
                    frame_paths=[f for _, f in depth_items],
                    depth_values=np.asarray([d for d, _ in depth_items]),
                    background_path=background if background.exists() else None,
                    forces=None if force is None else force[2],
                    raw_forces=None if force is None else force[3],
                    z_positions=None if force is None else force[0],
                    robot_poses=None if force is None else force[1],
                    metadata=metadata,
                )
                if skip_missing_force and not seq.has_force:
                    continue
                sequences.append(seq)

    return RctDataset(data_dir=data_dir, sequences=sequences)
