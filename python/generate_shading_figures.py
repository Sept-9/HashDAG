from io import BytesIO
from math import pi
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "image"
FONT_DIR = Path(r"C:\Windows\Fonts")
pdfmetrics.registerFont(TTFont("Times", FONT_DIR / "times.ttf"))
pdfmetrics.registerFont(TTFont("Times-Bold", FONT_DIR / "timesbd.ttf"))
pdfmetrics.registerFont(TTFont("Times-Italic", FONT_DIR / "timesi.ttf"))

INK = HexColor("#1B2732")
MID = HexColor("#536371")
LINE = HexColor("#C8D2DA")

SUN_IRRADIANCE = 2.6
SKY_IRRADIANCE = 1.26
F0 = 0.04
ALPHA2_BASE = 0.5 ** 4
NORMAL_VARIANCE_SCALE = 1.0
FINE_RESOLUTION = 128
LOD_BLOCK_SIZE = 4
DIRECTIONS = np.array([
    [-1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)
BIN_MAX = np.array([31.0, 31.0, 31.0, 31.0, 15.0, 15.0], dtype=np.float32)

_SCENE = None


def normalise(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def ggx_distribution(ndh, alpha2):
    d = ndh * ndh * (alpha2 - 1.0) + 1.0
    return alpha2 / (pi * np.maximum(d * d, 1e-8))


def smith_visibility(ndl, ndv, alpha2):
    lv = ndl * np.sqrt(ndv * ndv * (1.0 - alpha2) + alpha2)
    ll = ndv * np.sqrt(ndl * ndl * (1.0 - alpha2) + alpha2)
    return 0.5 / np.maximum(lv + ll, 1e-8)


def fresnel(vdh):
    m = np.clip(1.0 - vdh, 0.0, 1.0)
    return F0 + (1.0 - F0) * m ** 5


def specular_lobe(n, v, l, h, alpha2):
    ndl = np.sum(n * l, axis=-1)
    ndv = np.sum(n * v, axis=-1)
    ndh = np.maximum(np.sum(n * h, axis=-1), 0.0)
    value = ggx_distribution(ndh, alpha2) * smith_visibility(ndl, ndv, alpha2) * ndl
    return np.where((ndl > 0.0) & (ndv > 0.0), value, 0.0)


def combine(albedo, diffuse, specular, vdh):
    diffuse_brdf = albedo / pi
    f = fresnel(vdh)[..., None]
    color = diffuse_brdf * SKY_IRRADIANCE
    color += diffuse_brdf * ((1.0 - f) * diffuse[..., None] * SUN_IRRADIANCE)
    color += f * specular[..., None] * SUN_IRRADIANCE
    return color


def shade_standard(n, v, l, albedo):
    h = normalise(l + v)
    diffuse = np.maximum(np.sum(n * l, axis=-1), 0.0)
    specular = specular_lobe(n, v, l, h, ALPHA2_BASE)
    return combine(albedo, diffuse, specular, np.maximum(np.sum(v * h, axis=-1), 0.0))


def shade_relief(box_n, v, l, albedo, relief, variance):
    h = normalise(l + v)
    view_cosine = np.maximum(np.einsum("...k,dk->...d", v, DIRECTIONS), 0.0)
    weights = relief * view_cosine
    weight_sum = np.sum(weights, axis=-1)
    flat = np.maximum(0.0, 1.0 - weight_sum)
    total = np.maximum(weight_sum + flat, 1e-8)
    directional_diffuse = np.maximum(DIRECTIONS @ l, 0.0)
    diffuse = np.sum(weights * directional_diffuse, axis=-1)
    diffuse += flat * np.maximum(np.sum(box_n * l, axis=-1), 0.0)
    diffuse /= total

    if variance:
        alignment = np.einsum("...k,dk->...d", box_n, DIRECTIONS)
        moment = np.sum(weights * alignment * alignment, axis=-1) + flat
        s_axis = np.clip(moment / total, 0.0, 1.0)
        alpha2 = np.clip(
            ALPHA2_BASE + NORMAL_VARIANCE_SCALE * (1.0 - s_axis),
            ALPHA2_BASE,
            1.0,
        )
        specular = specular_lobe(box_n, v, l, h, alpha2)
    else:
        specular = np.zeros_like(weight_sum)
        for direction in range(6):
            n = np.broadcast_to(DIRECTIONS[direction], v.shape)
            specular += weights[..., direction] * specular_lobe(n, v, l, h, ALPHA2_BASE)
        specular += flat * specular_lobe(box_n, v, l, h, ALPHA2_BASE)
        specular /= total

    return combine(albedo, diffuse, specular, np.maximum(np.sum(v * h, axis=-1), 0.0))


def voxel_tree():
    n = FINE_RESOLUTION
    scale = n / 24.0
    x, y, z = np.ogrid[:n, :n, :n]
    x = x.astype(np.float32) + 0.5
    y = y.astype(np.float32) + 0.5
    z = z.astype(np.float32) + 0.5

    def scaled(point):
        return np.asarray(point, dtype=np.float32) * scale

    def capsule(a, b, radius):
        a = scaled(a)
        b = scaled(b)
        radius *= scale
        ab = b - a
        px = x - a[0]
        py = y - a[1]
        pz = z - a[2]
        t = np.clip((px * ab[0] + py * ab[1] + pz * ab[2]) / np.dot(ab, ab), 0.0, 1.0)
        dx = px - t * ab[0]
        dy = py - t * ab[1]
        dz = pz - t * ab[2]
        return dx * dx + dy * dy + dz * dz <= radius * radius

    wood = capsule((12, 12, 0), (12.2, 12, 14.5), 1.7)
    primary_branches = [
        ((12, 12, 8), (5.2, 9.3, 17.2), 1.0),
        ((12, 12, 10), (18.7, 12.5, 18.5), 1.0),
        ((12, 12, 11), (11.3, 19.0, 18.0), 0.95),
        ((12, 12, 9), (17.4, 6.5, 16.5), 0.9),
        ((12, 12, 13), (8.2, 15.2, 20.8), 0.8),
        ((12, 12, 13), (15.4, 16.8, 21.1), 0.8),
    ]
    for a, b, radius in primary_branches:
        wood |= capsule(a, b, radius)

    rng = np.random.default_rng(20260826)
    shoots = []
    crown_anchors = []
    for branch_start, branch_end, _ in primary_branches:
        branch_start = np.asarray(branch_start, dtype=np.float32)
        branch_end = np.asarray(branch_end, dtype=np.float32)
        for along in (0.52, 0.76, 1.0):
            crown_anchors.append(branch_start + along * (branch_end - branch_start))
    crown_anchors.append(np.array([12.0, 12.0, 20.5], dtype=np.float32))
    for anchor_index, tip in enumerate(crown_anchors):
        outward = normalise(tip - np.array([12.0, 12.0, 13.5], dtype=np.float32))
        shoot_count = 4 if anchor_index == len(crown_anchors) - 1 else 3
        for _ in range(shoot_count):
            jitter = np.array([
                rng.normal(0.0, 0.72),
                rng.normal(0.0, 0.72),
                rng.uniform(-0.05, 0.85),
            ], dtype=np.float32)
            direction = normalise(0.70 * outward + jitter)
            direction[2] = np.clip(direction[2], -0.18, 0.82)
            direction = normalise(direction)
            start = tip + outward * rng.uniform(-0.30, 0.35)
            end = start + direction * rng.uniform(2.3, 4.2)
            end = np.clip(end, np.array([1.0, 1.0, 13.0]), np.array([23.0, 23.0, 23.2]))
            wood |= capsule(start, end, rng.uniform(0.13, 0.23))
            shoots.append((start, end))

    color = np.zeros((n, n, n, 3), dtype=np.float32)
    material = np.zeros((n, n, n), dtype=np.uint8)
    material[wood] = 1
    variation = 0.80 + 0.20 * (0.5 + 0.5 * np.sin(x * 0.37 + y * 0.19 + z * 0.29))
    color[..., 0][wood] = (0.50 * variation)[wood]
    color[..., 1][wood] = (0.19 * variation)[wood]
    color[..., 2][wood] = (0.045 * variation)[wood]

    occupied = wood.copy()
    leaf = np.zeros_like(occupied)
    for shoot_start, shoot_end in shoots:
        shoot_axis = normalise(shoot_end - shoot_start)
        for _ in range(11):
            along = rng.uniform(0.22, 1.08)
            radial = rng.normal(0.0, 0.46, 3).astype(np.float32)
            radial -= shoot_axis * np.dot(radial, shoot_axis)
            center = scaled(shoot_start + along * (shoot_end - shoot_start) + radial)
            normal = normalise(np.array([
                rng.normal(0.0, 0.72),
                rng.normal(0.0, 0.72),
                abs(rng.normal(0.52, 0.48)) + 0.10,
            ], dtype=np.float32))
            tangent_seed = normalise(rng.normal(0.0, 1.0, 3).astype(np.float32))
            tangent_u = normalise(np.cross(normal, tangent_seed))
            if np.linalg.norm(tangent_u) < 0.2:
                tangent_u = normalise(np.cross(normal, np.array([1.0, 0.0, 0.0], dtype=np.float32)))
            tangent_v = normalise(np.cross(normal, tangent_u))
            radius_u = rng.uniform(0.52, 0.92) * scale
            radius_v = rng.uniform(0.22, 0.42) * scale
            thickness = rng.uniform(0.075, 0.135) * scale
            extent = int(np.ceil(max(radius_u, radius_v) + thickness + 1))
            low = np.maximum(np.floor(center - extent).astype(int), 0)
            high = np.minimum(np.ceil(center + extent).astype(int) + 1, n)
            lx, ly, lz = np.mgrid[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
            points = np.stack([lx + 0.5, ly + 0.5, lz + 0.5], axis=-1) - center
            du = points @ tangent_u
            dv = points @ tangent_v
            dn = points @ normal
            shape = (du / radius_u) ** 2 + (dv / radius_v) ** 2 + (dn / thickness) ** 2
            taper = 1.0 - 0.18 * np.clip((du / radius_u + 1.0) * 0.5, 0.0, 1.0)
            local_leaf = shape <= taper
            region = np.s_[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
            leaf[region] |= local_leaf
            hue = 0.72 + 0.25 * rng.random()
            local_material = material[region]
            assign = local_leaf & (local_material != 1)
            local_material[assign] = 2
            material[region] = local_material
            local_color = color[region]
            local_color[..., 0][assign] = 0.030 * hue
            local_color[..., 1][assign] = 0.64 * hue
            local_color[..., 2][assign] = 0.055 * hue
            color[region] = local_color

    leaf &= ~wood
    occupied |= leaf
    material[wood] = 1
    return occupied, material, color


def block_statistics(blocks):
    boundary = np.zeros((*blocks.shape[:3], 6), dtype=np.float32)
    area = np.zeros_like(boundary)
    boundary[..., 0] = np.sum(blocks[..., 0, :, :], axis=(-2, -1))
    boundary[..., 1] = np.sum(blocks[..., -1, :, :], axis=(-2, -1))
    boundary[..., 2] = np.sum(blocks[..., :, 0, :], axis=(-2, -1))
    boundary[..., 3] = np.sum(blocks[..., :, -1, :], axis=(-2, -1))
    boundary[..., 4] = np.sum(blocks[..., :, :, 0], axis=(-2, -1))
    boundary[..., 5] = np.sum(blocks[..., :, :, -1], axis=(-2, -1))
    area[..., 0] = boundary[..., 0] + np.sum(blocks[..., 1:, :, :] & ~blocks[..., :-1, :, :], axis=(-3, -2, -1))
    area[..., 1] = boundary[..., 1] + np.sum(blocks[..., :-1, :, :] & ~blocks[..., 1:, :, :], axis=(-3, -2, -1))
    area[..., 2] = boundary[..., 2] + np.sum(blocks[..., :, 1:, :] & ~blocks[..., :, :-1, :], axis=(-3, -2, -1))
    area[..., 3] = boundary[..., 3] + np.sum(blocks[..., :, :-1, :] & ~blocks[..., :, 1:, :], axis=(-3, -2, -1))
    area[..., 4] = boundary[..., 4] + np.sum(blocks[..., :, :, 1:] & ~blocks[..., :, :, :-1], axis=(-3, -2, -1))
    area[..., 5] = boundary[..., 5] + np.sum(blocks[..., :, :, :-1] & ~blocks[..., :, :, 1:], axis=(-3, -2, -1))
    raw = np.clip((area - boundary) / float(LOD_BLOCK_SIZE ** 2), 0.0, 1.0)
    quantised = np.floor(raw * BIN_MAX + 0.5)
    decoded = quantised / BIN_MAX
    return area, boundary, decoded.astype(np.float32)


def build_scene():
    global _SCENE
    if _SCENE is not None:
        return _SCENE
    occupied, material, color = voxel_tree()
    m = FINE_RESOLUTION // LOD_BLOCK_SIZE
    blocks = occupied.reshape(m, LOD_BLOCK_SIZE, m, LOD_BLOCK_SIZE, m, LOD_BLOCK_SIZE)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5)
    material_blocks = material.reshape(m, LOD_BLOCK_SIZE, m, LOD_BLOCK_SIZE, m, LOD_BLOCK_SIZE)
    material_blocks = material_blocks.transpose(0, 2, 4, 1, 3, 5)
    color_blocks = color.reshape(m, LOD_BLOCK_SIZE, m, LOD_BLOCK_SIZE, m, LOD_BLOCK_SIZE, 3)
    color_blocks = color_blocks.transpose(0, 2, 4, 1, 3, 5, 6)
    counts = np.sum(blocks, axis=(-3, -2, -1))
    coarse = counts > 0
    coarse_color = np.sum(color_blocks, axis=(-4, -3, -2)) / np.maximum(counts[..., None], 1)
    area, boundary, relief = block_statistics(blocks)
    _SCENE = {
        "occupied": occupied,
        "material": material,
        "color": color,
        "blocks": blocks,
        "material_blocks": material_blocks,
        "color_blocks": color_blocks,
        "coarse": coarse,
        "coarse_color": coarse_color,
        "area": area,
        "boundary": boundary,
        "relief": relief,
        "counts": counts,
    }
    return _SCENE


def tone_rgb(rgb):
    rgb = np.clip(rgb, 0.0, 1.5) / 1.5
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)
    return np.uint8(np.clip((rgb - 0.42) * 1.28 + 0.42, 0.0, 1.0) * 255.0)


def neighbour_count(grid):
    padded = np.pad(grid.astype(np.uint8), 1)
    count = np.zeros_like(grid, dtype=np.uint8)
    sx, sy, sz = grid.shape
    for dx in range(3):
        for dy in range(3):
            for dz in range(3):
                if dx == 1 and dy == 1 and dz == 1:
                    continue
                count += padded[dx:dx + sx, dy:dy + sy, dz:dz + sz]
    return count


def exposed_mask(grid, delta):
    neighbour = np.zeros_like(grid)
    if delta == (1, 0, 0):
        neighbour[:-1] = grid[1:]
    elif delta == (0, -1, 0):
        neighbour[:, 1:] = grid[:, :-1]
    else:
        neighbour[:, :, :-1] = grid[:, :, 1:]
    return grid & ~neighbour


def render_grid(grid, colors, relief, mode, cell_size, size=700, shadow=True):
    view = normalise(np.array([1.15, -1.35, 0.95], dtype=np.float32))
    right = normalise(np.array([view[1], -view[0], 0.0], dtype=np.float32))
    up = normalise(np.cross(right, view))
    light = normalise(np.array([0.30, 1.00, 0.50], dtype=np.float32))
    faces = [
        ((1, 0, 0), np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)])),
        ((0, -1, 0), np.array([0.0, -1.0, 0.0], dtype=np.float32), np.array([(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)])),
        ((0, 0, 1), np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])),
    ]
    ao = neighbour_count(grid)
    projected_groups = []
    depth_groups = []
    color_groups = []
    for delta, normal, corners in faces:
        coords = np.argwhere(exposed_mask(grid, delta))
        origins = coords.astype(np.float32) * cell_size
        verts = origins[:, None, :] + corners[None, :, :] * cell_size
        projected_groups.append(np.stack([verts @ right, verts @ up], axis=-1))
        depth_groups.append(np.mean(verts @ view, axis=1))
        albedo = colors[tuple(coords.T)]
        count = len(coords)
        n = np.broadcast_to(normal, (count, 3))
        v = np.broadcast_to(view, (count, 3))
        if mode == "reference" or mode == "average":
            rgb = shade_standard(n, v, light, albedo)
        else:
            node_relief = relief[tuple(coords.T)]
            rgb = shade_relief(n, v, light, albedo, node_relief, mode == "variance")
        rgb = tone_rgb(rgb).astype(np.float32)
        local_ao = ao[tuple(coords.T)]
        factor = np.maximum(0.62, 1.0 - local_ao * (0.012 if mode == "reference" else 0.016))
        color_groups.append(np.uint8(rgb * factor[:, None]))

    projected = np.concatenate(projected_groups)
    depths = np.concatenate(depth_groups)
    face_colors = np.concatenate(color_groups)
    points = projected.reshape(-1, 2)
    pmin, pmax = np.min(points, axis=0), np.max(points, axis=0)
    margin = 34
    scale = min(
        (size - 2 * margin) / max(pmax[0] - pmin[0], 1e-8),
        (size - 2 * margin) / max(pmax[1] - pmin[1], 1e-8),
    )
    offset_x = (size - (pmax[0] - pmin[0]) * scale) / 2 - pmin[0] * scale
    offset_y = (size - (pmax[1] - pmin[1]) * scale) / 2 + pmax[1] * scale
    image = Image.new("RGBA", (size, size), (245, 248, 249, 255))
    if shadow:
        shadow_layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_layer)
        shadow_draw.ellipse(
            (int(size * 0.18), int(size * 0.75), int(size * 0.84), int(size * 0.92)),
            fill=(24, 43, 34, 98),
        )
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(size * 0.025))
        image = Image.alpha_composite(image, shadow_layer)
    draw = ImageDraw.Draw(image)
    outline = (30, 48, 58)
    outline_width = 1 if mode == "reference" else 2
    for index in np.argsort(depths):
        polygon = [
            (float(u * scale + offset_x), float(offset_y - v * scale))
            for u, v in projected[index]
        ]
        draw.polygon(polygon, fill=tuple(face_colors[index]), outline=outline, width=outline_width)
    return image.convert("RGB")


def render_scene(mode, size=700):
    scene = build_scene()
    render_size = max(size, 900)
    if mode == "reference":
        image = render_grid(
            scene["occupied"],
            scene["color"],
            None,
            mode,
            1,
            render_size,
            True,
        )
    else:
        image = render_grid(
            scene["coarse"],
            scene["coarse_color"],
            scene["relief"],
            mode,
            LOD_BLOCK_SIZE,
            render_size,
            True,
        )
    crop = (
        int(render_size * 0.35),
        int(render_size * 0.68),
        int(render_size * 0.65),
        int(render_size * 0.98),
    )
    return image.crop(crop).resize((size, size), Image.Resampling.LANCZOS)


def representative_node():
    scene = build_scene()
    target = np.array([2.07, 1.95, 2.09], dtype=np.float32)
    above = np.zeros_like(scene["coarse"])
    above[..., :-1] = scene["coarse"][..., 1:]
    top_surface = scene["coarse"] & ~above
    has_wood = np.any(scene["material_blocks"] == 1, axis=(-3, -2, -1))
    has_leaf = np.any(scene["material_blocks"] == 2, axis=(-3, -2, -1))
    valid = top_surface & has_leaf & ~has_wood & (scene["counts"] >= 6) & (scene["counts"] <= 56)
    valid &= scene["boundary"][..., 5] > 0
    valid &= scene["blocks"][..., 2, 1, 2]
    angles = np.linspace(-0.49 * pi, 0.49 * pi, 121, dtype=np.float32)
    meridian_x = np.stack([np.sin(angles), np.zeros_like(angles), np.cos(angles)], axis=-1)
    meridian_y = np.stack([np.zeros_like(angles), np.sin(angles), np.cos(angles)], axis=-1)
    azimuth = np.linspace(0.0, 2.0 * pi, 144, endpoint=False, dtype=np.float32)
    ring = np.stack([
        np.sin(1.05) * np.cos(azimuth),
        np.sin(1.05) * np.sin(azimuth),
        np.full_like(azimuth, np.cos(1.05)),
    ], axis=-1)
    probe_view = np.concatenate([meridian_x, meridian_y, ring], axis=0)
    best_score = -np.inf
    index = None
    for candidate in np.argwhere(valid):
        candidate = tuple(candidate)
        mask = scene["blocks"][candidate]
        coords = np.argwhere(mask).astype(np.float32)
        normal, hit, hit_valid = ray_entry_normals(probe_view, target, coords, coords + 1.0)
        axis = np.argmax(np.abs(normal), axis=-1)
        face = axis * 2 + (normal[np.arange(len(normal)), axis] > 0.0)
        signature = hit * 6 + face
        signature[~hit_valid] = -1
        first = len(angles)
        second = first * 2
        transitions = np.count_nonzero(signature[1:first] != signature[:first - 1])
        transitions += np.count_nonzero(signature[first + 1:second] != signature[first:second - 1])
        transitions += np.count_nonzero(signature[second + 1:] != signature[second:-1])
        transitions += signature[second] != signature[-1]
        distinct = len(np.unique(signature[hit_valid]))
        hit_ratio = np.mean(hit_valid)
        if hit_ratio < 0.55:
            continue
        node_score = transitions * 0.22 + distinct * 0.32 + hit_ratio * 2.0
        node_score += scene["relief"][candidate][5] * 1.5
        node_score += np.std(scene["color_blocks"][candidate][mask], axis=0).sum() * 2.0
        if node_score > best_score:
            best_score = node_score
            index = candidate
    if index is None:
        score = np.sum(scene["relief"], axis=-1)
        score[~valid] = -np.inf
        index = np.unravel_index(np.argmax(score), score.shape)
    mask = scene["blocks"][index]
    return {
        "index": index,
        "mask": mask,
        "colors": scene["color_blocks"][index],
        "albedo": scene["coarse_color"][index],
        "area": scene["area"][index],
        "boundary": scene["boundary"][index],
        "relief": scene["relief"][index],
        "target": target,
        "complexity": best_score,
    }


def ray_entry_normals(view, target, lower, upper):
    origin = target[None, :] + view * 8.0
    direction = -view
    safe_direction = np.where(
        np.abs(direction) < 1e-7,
        np.where(direction < 0.0, -1e-7, 1e-7),
        direction,
    )
    near_axis = np.minimum(
        (lower[None, ...] - origin[:, None, :]) / safe_direction[:, None, :],
        (upper[None, ...] - origin[:, None, :]) / safe_direction[:, None, :],
    )
    far_axis = np.maximum(
        (lower[None, ...] - origin[:, None, :]) / safe_direction[:, None, :],
        (upper[None, ...] - origin[:, None, :]) / safe_direction[:, None, :],
    )
    near = np.max(near_axis, axis=-1)
    far = np.min(far_axis, axis=-1)
    valid = far >= np.maximum(near, 0.0)
    hit = np.argmin(np.where(valid, near, np.inf), axis=1)
    row = np.arange(len(view))
    chosen_near = near_axis[row, hit]
    axis = np.argmax(chosen_near, axis=-1)
    normal = np.zeros_like(view)
    normal[row, axis] = np.sign(view[row, axis])
    return normal, hit, valid[row, hit]


def ray_response(view, node):
    coords = np.argwhere(node["mask"]).astype(np.float32)
    fine_normal, hit, fine_valid = ray_entry_normals(view, node["target"], coords, coords + 1.0)
    hit_coords = coords[hit].astype(np.int32)
    fine_albedo = node["colors"][tuple(hit_coords.T)]
    coarse_normal, _, coarse_valid = ray_entry_normals(
        view,
        node["target"],
        np.zeros((1, 3), dtype=np.float32),
        np.full((1, 3), LOD_BLOCK_SIZE, dtype=np.float32),
    )
    return fine_normal, fine_albedo, coarse_normal, fine_valid, coarse_valid


def tone_response(rgb, disk):
    mapped = 1.0 - np.exp(-1.8 * np.maximum(rgb, 0.0))
    mapped = np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * np.power(mapped, 1.0 / 2.4) - 0.055)
    out = np.ones_like(mapped) * np.array([0.96, 0.97, 0.98])
    out[disk] = np.clip(mapped[disk], 0.0, 1.0)
    edge = disk & ~(
        np.roll(disk, 1, 0)
        & np.roll(disk, -1, 0)
        & np.roll(disk, 1, 1)
        & np.roll(disk, -1, 1)
    )
    out[edge] = np.array([0.20, 0.27, 0.33])
    return out


def response_images(size=560):
    npx = 360
    yy, xx = np.mgrid[0:npx, 0:npx]
    x = (xx + 0.5) / npx * 2.0 - 1.0
    y = 1.0 - (yy + 0.5) / npx * 2.0
    radius2 = x * x + y * y
    disk = radius2 <= 1.0
    view = np.stack([x, y, np.sqrt(np.maximum(0.0, 1.0 - radius2))], axis=-1)
    light = normalise(np.array([0.30, 1.00, 0.50], dtype=np.float32))
    node = representative_node()
    sample_view = view[disk].astype(np.float32)
    fine_normal, fine_albedo, box_normal, fine_valid, coarse_valid = ray_response(sample_view, node)
    albedo = np.broadcast_to(node["albedo"], sample_view.shape)
    relief = np.broadcast_to(node["relief"], (len(sample_view), 6))
    samples = [
        shade_standard(fine_normal, sample_view, light, fine_albedo),
        shade_standard(box_normal, sample_view, light, albedo),
        shade_relief(box_normal, sample_view, light, albedo, relief, False),
        shade_relief(box_normal, sample_view, light, albedo, relief, True),
    ]
    linear = []
    sample_validity = [fine_valid, coarse_valid, coarse_valid, coarse_valid]
    for sample, sample_valid in zip(samples, sample_validity):
        field = np.zeros((*disk.shape, 3), dtype=np.float32)
        field[disk] = np.where(sample_valid[:, None], sample, 0.0)
        linear.append(field)
    responses = [tone_response(value, disk) for value in linear]
    differences = [np.linalg.norm(value - linear[0], axis=-1) for value in linear[1:]]
    shared_max = max(np.percentile(value[disk], 98) for value in differences)
    false_color = []
    for difference in differences:
        t = np.clip(difference / max(shared_max, 1e-8), 0.0, 1.0)
        image = np.ones((*t.shape, 3), dtype=np.float32) * np.array([0.96, 0.97, 0.98])
        mapped = np.stack([
            np.clip(1.35 * t, 0.0, 1.0),
            np.clip(0.85 * t ** 1.55, 0.0, 1.0),
            np.clip(0.18 * t ** 2.4, 0.0, 1.0),
        ], axis=-1)
        image[disk] = mapped[disk]
        false_color.append(image)

    def to_image(array):
        image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0), "RGB")
        return image.resize((size, size), Image.Resampling.LANCZOS)

    micro_colors = np.where(node["mask"][..., None], node["colors"], 0.0)
    micro = render_grid(
        node["mask"],
        micro_colors,
        None,
        "reference",
        1,
        size,
        False,
    )
    return [to_image(value) for value in responses], [to_image(value) for value in false_color], micro, node


def image_reader(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    stream.seek(0)
    return ImageReader(stream)


def paragraph(c, value, x, y, w, h, size=9, leading=None, color=INK):
    p = Paragraph(
        value,
        ParagraphStyle(
            "p",
            fontName="Times",
            fontSize=size,
            leading=leading or size * 1.2,
            alignment=TA_CENTER,
            textColor=color,
            spaceAfter=0,
            spaceBefore=0,
        ),
    )
    _, ph = p.wrap(w, h)
    p.drawOn(c, x, y + (h - ph) / 2)


def title(c, value, cx, y, size=10, color=INK):
    c.setFillColor(color)
    c.setFont("Times-Bold", size)
    c.drawCentredString(cx, y, value)


def panel(c, x, y, w, h):
    c.setFillColor(white)
    c.setStrokeColor(LINE)
    c.setLineWidth(0.7)
    c.rect(x, y, w, h, fill=1, stroke=1)


def shading_comparison():
    c = Canvas(str(OUT / "shading_relief_comparison.pdf"), pagesize=(12 * inch, 3.6 * inch))
    w, h = 12 * inch, 3.6 * inch
    margin = 28
    gap = 12
    panel_w = (w - 2 * margin - 3 * gap) / 4
    panel_h = h - 48
    y = 18
    modes = ["reference", "average", "relief", "variance"]
    headings = [
        "(a) Fine 128x128x128 lower trunk",
        "(b) 4x4x4 lower-trunk LoD",
        "(c) Node-local relief at same LoD",
        "(d) Relief + GGX variance",
    ]
    subtitles = [
        "Full leafy tree retained; bottom 30% enlarged",
        "Same LoD level; box normals only",
        "Per-node quantised descriptors",
        "Same descriptors; modulated roughness",
    ]
    for index, (mode, heading, subtitle) in enumerate(zip(modes, headings, subtitles)):
        x = margin + index * (panel_w + gap)
        panel(c, x, y, panel_w, panel_h)
        title(c, heading, x + panel_w / 2, h - 20, 8.7)
        image = render_scene(mode)
        image_size = panel_w - 16
        c.drawImage(
            image_reader(image),
            x + 8,
            y + 30,
            image_size,
            image_size,
            preserveAspectRatio=True,
            mask="auto",
        )
        paragraph(c, f"<i>{subtitle}</i>", x + 8, y + 3, panel_w - 16, 23, 8.0, 9.5, MID)
    c.showPage()
    c.save()


def roughness_response():
    c = Canvas(str(OUT / "roughness_variance_response.pdf"), pagesize=(12 * inch, 6.2 * inch))
    w, h = 12 * inch, 6.2 * inch
    margin = 28
    gap = 12
    panel_w = (w - 2 * margin - 3 * gap) / 4
    top_y = 221
    top_h = 193
    responses, differences, micro, node = response_images()
    headings = [
        "(a) Fine-voxel first-hit reference",
        "(b) LoD box first-hit response",
        "(c) Relief weighted",
        "(d) Relief + GGX variance",
    ]
    for index, (response, heading) in enumerate(zip(responses, headings)):
        x = margin + index * (panel_w + gap)
        panel(c, x, top_y, panel_w, top_h)
        title(c, heading, x + panel_w / 2, h - 23, 9.1)
        c.drawImage(
            image_reader(response),
            x + 12,
            top_y + 11,
            panel_w - 24,
            panel_w - 24,
            preserveAspectRatio=True,
            mask="auto",
        )

    lower_y = 28
    lower_h = 159
    for index in range(4):
        x = margin + index * (panel_w + gap)
        panel(c, x, lower_y, panel_w, lower_h)
        if index == 0:
            title(c, "Selected 4x4x4 leaf node", x + panel_w / 2, lower_y + lower_h - 22, 8.8)
            image = micro
        else:
            label = chr(97 + index)
            title(c, f"Absolute response change: |{label} - a|", x + panel_w / 2, lower_y + lower_h - 22, 8.8)
            image = differences[index - 1]
        c.drawImage(
            image_reader(image),
            x + 40,
            lower_y + 9,
            panel_w - 80,
            panel_w - 80,
            preserveAspectRatio=True,
            mask="auto",
        )
    c.showPage()
    c.save()
    print(
        "Selected response node",
        node["index"],
        "relief",
        np.array2string(node["relief"], precision=3),
    )


if __name__ == "__main__":
    build_scene()
    shading_comparison()
    roughness_response()
    print(f"Generated shading figures in {OUT}")
