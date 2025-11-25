"""Python port of examples/3_duck.rs using the floorplan_py bindings."""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

import floorplan as fp

try:
    from svgpathtools import parse_path
except ImportError as exc:  # pragma: no cover - helper dependency
    raise SystemExit(
        "svgpathtools is required for examples/3_duck.py. Install via 'pip install svgpathtools'."
    ) from exc

Point = Tuple[float, float]
DUCK_PATH = (
    "M7920 11494 c-193 -21 -251 -29 -355 -50 -540 -105 -1036 -366 -1442 "
    "-758 -515 -495 -834 -1162 -904 -1891 -15 -154 -6 -563 15 -705 66 -440 220 "
    "-857 442 -1203 24 -37 44 -69 44 -71 0 -2 -147 -3 -327 -4 -414 -1 -765 -23 "
    "-1172 -72 -97 -12 -167 -17 -170 -11 -3 5 -33 52 -66 106 -231 372 -633 798 "
    "-1040 1101 -309 229 -625 409 -936 532 -287 113 -392 130 -500 79 -65 -32 "
    "-118 -81 -249 -237 -627 -745 -1009 -1563 -1170 -2505 -54 -320 -77 -574 -86 "
    "-965 -28 -1207 238 -2308 785 -3242 120 -204 228 -364 270 -397 84 -67 585 "
    "-319 901 -454 1197 -511 2535 -769 3865 -744 983 19 1875 166 2783 458 334 "
    "108 918 340 1013 404 99 65 407 488 599 824 620 1080 835 2329 614 3561 -75 "
    "415 -226 892 -401 1262 -39 82 -54 124 -47 133 5 7 42 58 82 114 41 55 77 99 "
    "81 96 4 -2 68 -8 142 -14 766 -53 1474 347 1858 1051 105 192 186 439 228 693 "
    "27 167 24 487 -6 660 -33 189 -64 249 -150 289 -46 21 -51 21 -846 21 -440 0 "
    "-828 -3 -861 -7 l-62 -7 -32 86 c-54 143 -194 412 -289 554 -479 720 -1201 "
    "1178 -2040 1295 -101 14 -496 27 -571 18z"
)


def write_duck_path_svg(output: Path) -> None:
        path = parse_path(DUCK_PATH)
        xmin, xmax, ymin, ymax = path.bbox()
        width = xmax - xmin
        height = ymax - ymin
        stroke_width = max(width, height) * 0.001
        y_reflect = ymin + ymax
        svg = f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"{xmin} {ymin} {width} {height}\">
    <path d=\"{DUCK_PATH}\" fill=\"#FFD700\" stroke=\"black\" stroke-width=\"{stroke_width}\" transform=\"matrix(1 0 0 -1 0 {y_reflect})\" />
</svg>
"""
        output.write_text(svg)


def sample_svg_path(num_samples: int) -> List[Point]:
    path = parse_path(DUCK_PATH)
    total_length = path.length()
    samples = []
    for i in range(num_samples):
        #pos = path.point(i / num_samples * path.length() / total_length)
        pos = path.point(i / num_samples)
        samples.append((pos.real, pos.imag))
    return samples


def resample_polyloop(points: Sequence[Point], target_count: int) -> List[Point]:
    def segment_lengths(seq: Sequence[Point]) -> List[float]:
        lengths = []
        for i in range(len(seq)):
            x0, y0 = seq[i]
            x1, y1 = seq[(i + 1) % len(seq)]
            lengths.append(math.hypot(x1 - x0, y1 - y0))
        return lengths

    lengths = segment_lengths(points)
    perimeter = sum(lengths)
    spacing = perimeter / target_count
    print( "perimeter", perimeter, "spacing", spacing )
    resampled: List[Point] = []
    acc = 0.0
    idx = 0
    cur = points[0]
    for _ in range(target_count):
        while acc + lengths[idx] < spacing:
            acc += lengths[idx]
            idx = (idx + 1) % len(points)
        remain = spacing - acc
        x0, y0 = points[idx]
        x1, y1 = points[(idx + 1) % len(points)]
        seg_len = lengths[idx]
        t = remain / seg_len if seg_len else 0.0
        cur = (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
        resampled.append(cur)
        if remain > 0 :
            idx = (idx + 1) % len(points)
            acc = seg_len - remain
        else :
            acc = 0.0
    return resampled


def normalize_loop(points: Sequence[Point]) -> List[Point]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) * 0.5
    center_y = (min_y + max_y) * 0.5
    scale = max(max_x - min_x, max_y - min_y)
    if scale == 0:
        scale = 1.0
    return [((x - center_x) / scale + 0.5, (y - center_y) / scale + 0.5) for x, y in points]


def polygon_area(points: Sequence[Point]) -> float:
    area = 0.0
    for i in range(len(points)):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % len(points)]
        area += x0 * y1 - x1 * y0
    return abs(area) * 0.5


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    for i in range(len(polygon)):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % len(polygon)]
        if ((y0 > y) != (y1 > y)) and (x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-9) + x0):
            inside = not inside
    return inside



def problem(seed: int):
    num_room = 6
    room2color = [fp.random_room_color(seed=7 + i) for i in range(num_room)]
    
    print("Sampling duck path...")
    path_samples = sample_svg_path(400)
    print( path_samples )
    
    print("Resampling duck path...")
    loop = resample_polyloop(path_samples, 100)
    def mylength(p0, p1) -> float:
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    for i, p in enumerate(loop):
        print( "Segment", i, "point", p )
        print( mylength( loop[i], loop[(i+1)%len(loop)] ) )
    
    print("Normalizing duck path...")
    loop = normalize_loop(loop)
    
    print("Flattening duck path...")
    loop_flat: List[float] = []
    for x, y in loop:
        loop_flat.extend([x, y])
    area_ratio = [0.4, 0.2, 0.2, 0.2, 0.2, 0.1]
    total_area = polygon_area(loop)
    sum_ratio = sum(area_ratio)
    room2area_trg = [val / sum_ratio * total_area for val in area_ratio]
    
    print("Generating Poisson disk samples...")
    poisson_points = fp.poisson_disk_sampling(loop, 0.03, 50, seed)
    
    site2xy2flag = [0.0] * len(poisson_points)
    site2room = fp.site2room(len(poisson_points) // 2, room2area_trg[:-1])
    poisson_points.extend([0.48, 0.06, 0.52, 0.06])
    site2xy2flag.extend([1.0, 1.0, 1.0, 1.0])
    site2room.extend([len(room2area_trg) - 1, len(room2area_trg) - 1])
    room_connections = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    return (
        loop_flat,
        poisson_points,
        site2room,
        site2xy2flag,
        room2area_trg,
        room2color,
        room_connections,
    )


def main() -> None:
    write_duck_path_svg(Path("duck_path.svg"))
    (
        vtxl2xy,
        site2xy,
        site2room,
        site2xy2flag,
        room2area_trg,
        room2color,
        room_connections,
    ) = problem(0)
    palette = [0xFFFFFF, 0x000000] + room2color
    target_dir = Path("target")
    target_dir.mkdir(exist_ok=True)
    canvas = fp.CanvasGif(str(target_dir / "3_duck.gif"), 1024, 1024, palette)
    fp.optimize(
        canvas,
        vtxl2xy,
        site2xy,
        site2room,
        site2xy2flag,
        room2area_trg,
        room2color,
        room_connections, 1000
    )


if __name__ == "__main__":
    main()
