# bs_placement.py
# GA/NSGA-II for placing BS on building rooftops (osm.poly.xml).

from __future__ import annotations
import argparse, json, math, random, os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from shapely.geometry import Polygon
import lxml.etree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ----------------------------- Fixed scenario bounds -----------------------------
# Area ≈ 1113 m × 856 m ≈ 953,000 m² (0.95 km²)
SCENARIO_MIN_X = 154.76061138609438
SCENARIO_MAX_X = 1267.8307381998957
SCENARIO_MIN_Y = 0.4836547144772948
SCENARIO_MAX_Y = 856.969029905406

SCENARIO_BBOX = (SCENARIO_MIN_X, SCENARIO_MAX_X, SCENARIO_MIN_Y, SCENARIO_MAX_Y)

# ----------------------------- Default parameters -----------------------------
DEFAULT_R_COVER = 200.0      # coverage radius (m)
DEFAULT_MIN_CAND_SEP = 60.0  # minimum separation between candidate sites (m)
DEFAULT_POP = 120
DEFAULT_GEN = 150
DEFAULT_PMUT = 0.25
DEFAULT_PCX  = 0.85
MAX_SITES_CAP_LIMIT = 10     # upper cap limit

VALID_TYPES = {
    'building.roof', 'building.university', 'building.yes', 'building.school',
    'man_made.tower', 'man_made.water_tower', 'amenity.library',
    'amenity.restaurant', 'amenity.fast_food', 'amenity.clinic', 'amenity.pub',
    'building.public', 'man_made.yes', 'amenity.bank'
}

@dataclass
class Site:
    idx: int
    xy: Tuple[float, float]
    poly_idx: int

# ----------------------------- Parse polygons from osm.poly.xml -----------------------------
def parse_osm_polygons(osm_poly_xml: str, min_area: float = 100.0) -> Tuple[List[Polygon], List[str]]:
    tree = ET.parse(osm_poly_xml)
    root = tree.getroot()

    polys: List[Polygon] = []
    types: List[str] = []
    for poly in root.findall(".//poly"):
        tp = poly.get("type", "")
        if tp not in VALID_TYPES:
            continue
        shape = poly.get("shape")
        if not shape:
            continue
        coords = []
        for pair in shape.split():
            x, y = pair.split(",")
            coords.append((float(x), float(y)))
        if len(coords) < 3:
            continue
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        p = Polygon(coords)
        if (not p.is_valid) or (p.area < min_area):
            continue
        polys.append(p)
        types.append(tp)
    return polys, types

def build_sites_from_vertices(polys: List[Polygon],
                              bbox: Tuple[float,float,float,float],
                              margin: float = 0.0) -> List[Site]:
    min_x, max_x, min_y, max_y = bbox
    x0, x1 = min_x - margin, max_x + margin
    y0, y1 = min_y - margin, max_y + margin

    sites: List[Site] = []
    s_idx = 0
    for i, p in enumerate(polys):
        coords = list(p.exterior.coords)
        if len(coords) <= 1:
            continue
        for (x, y) in coords[:-1]:  # skip closing point
            if x0 <= x <= x1 and y0 <= y <= y1:
                sites.append(Site(idx=s_idx, xy=(float(x), float(y)), poly_idx=i))
                s_idx += 1
    return sites

def deduplicate_by_distance(cands: List[Site], dmin: float) -> List[Site]:
    kept: List[Site] = []
    for s in cands:
        ok = True
        for t in kept:
            dx = s.xy[0] - t.xy[0]
            dy = s.xy[1] - t.xy[1]
            if (dx*dx + dy*dy) < dmin*dmin:
                ok = False
                break
        if ok:
            kept.append(s)
    return kept

# ----------------------------- Objective geometry -----------------------------
def circle_circle_overlap_area(d: float, r: float) -> float:
    if d >= 2*r:
        return 0.0
    if d <= 1e-12:
        return math.pi * r * r
    alpha = 2*math.acos(d/(2*r))
    r2 = r*r
    return r2*(alpha - math.sin(alpha))

def overlap_penalty(xys: np.ndarray, r: float) -> float:
    n = len(xys)
    if n < 2:
        return 0.0
    A = math.pi * r * r
    tot = 0.0
    pairs = 0
    for i in range(n):
        xi, yi = xys[i]
        for j in range(i+1, n):
            xj, yj = xys[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < 2*r:
                ov = circle_circle_overlap_area(d, r) / A
                tot += min(1.0, ov)
                pairs += 1
    if pairs == 0:
        return 0.0
    return tot / pairs

def circle_rect_intersection_area(cx: float, cy: float, r: float,
                                  xmin: float, xmax: float, ymin: float, ymax: float,
                                  S: int = 120) -> float:
    # sample within the bounding square of the circle
    x0, x1 = cx - r, cx + r
    y0, y1 = cy - r, cy + r
    cnt = 0
    cnt_circle = 0
    for i in range(S):
        x = x0 + (i + 0.5) * (2*r / S)
        for j in range(S):
            y = y0 + (j + 0.5) * (2*r / S)
            if (x - cx)**2 + (y - cy)**2 <= r*r:
                cnt_circle += 1
                if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                    cnt += 1
    if cnt_circle == 0:
        return 0.0
    # fraction × actual circle area
    frac = cnt / float(cnt_circle)
    return frac * (math.pi * r * r)

def outside_penalty(xys: np.ndarray, r: float,
                    bbox: Tuple[float,float,float,float]) -> float:
    xmin, xmax, ymin, ymax = bbox
    A = math.pi * r * r
    vals = []
    for (cx, cy) in xys:
        inside_area = circle_rect_intersection_area(cx, cy, r, xmin, xmax, ymin, ymax, S=120)
        waste = 1.0 - max(0.0, min(1.0, inside_area / A))
        vals.append(waste)
    return float(np.mean(vals)) if vals else 0.0

# ----------------------------- NSGA-II minimal -----------------------------
def non_dom_sort(fvals: List[Tuple[float,float]]) -> List[List[int]]:
    N = len(fvals)
    S = [[] for _ in range(N)]
    n = [0]*N
    rank = [0]*N
    fronts: List[List[int]] = [[]]

    for p in range(N):
        for q in range(N):
            if p == q: continue
            fp = fvals[p]; fq = fvals[q]
            if (fp[0] <= fq[0] and fp[1] <= fq[1]) and (fp[0] < fq[0] or fp[1] < fq[1]):
                S[p].append(q)
            elif (fq[0] <= fp[0] and fq[1] <= fp[1]) and (fq[0] < fp[0] or fq[1] < fp[1]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        Q: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i+1
                    Q.append(q)
        i += 1
        fronts.append(Q)
    if not fronts[-1]:
        fronts.pop()
    return fronts

def crowding_distance(front: List[int], fvals: List[Tuple[float,float]]) -> List[float]:
    m = 2
    L = len(front)
    if L == 0: return []
    if L == 1: return [float("inf")]
    if L == 2: return [float("inf"), float("inf")]
    dist = [0.0]*L
    for k in range(m):
        idx = sorted(range(L), key=lambda i: fvals[front[i]][k])
        dist[idx[0]] = dist[idx[-1]] = float("inf")
        fmin = fvals[front[idx[0]]][k]
        fmax = fvals[front[idx[-1]]][k]
        denom = max(1e-12, (fmax - fmin))
        for ii in range(1, L-1):
            a = fvals[front[idx[ii-1]]][k]
            b = fvals[front[idx[ii+1]]][k]
            dist[idx[ii]] += (b - a) / denom
    return dist

# ----------------------------- GA helpers -----------------------------
def random_solution(cand_idx: np.ndarray, n_sites: int) -> np.ndarray:
    return np.sort(np.random.choice(cand_idx, size=n_sites, replace=False))

def mutate(sol: np.ndarray, cand_idx: np.ndarray, pmut: float) -> np.ndarray:
    s = sol.copy()
    for i in range(len(s)):
        if random.random() < pmut:
            choices = np.setdiff1d(cand_idx, s)
            if len(choices) > 0:
                s[i] = np.random.choice(choices)
    return np.sort(s)

def crossover(a: np.ndarray, b: np.ndarray, pcx: float) -> Tuple[np.ndarray, np.ndarray]:
    if random.random() >= pcx:
        return a.copy(), b.copy()
    k = len(a)
    cut = random.randint(1, k-1)
    left = set(a[:cut])
    child1 = list(a[:cut]) + [x for x in b if x not in left]
    left = set(b[:cut])
    child2 = list(b[:cut]) + [x for x in a if x not in left]
    return np.sort(np.array(child1[:k])), np.sort(np.array(child2[:k]))

# ----------------------------- Evaluation -----------------------------
def evaluate(sol: np.ndarray, cand_sites: List[Site], r: float,
            bbox: Tuple[float,float,float,float]) -> Tuple[float,float]:
    pts = np.array([cand_sites[i].xy for i in sol], dtype=float)
    f1 = overlap_penalty(pts, r)
    f2 = outside_penalty(pts, r, bbox)
    return (f1, f2)

# ----------------------------- Plot -----------------------------
def plot_selection(polys: List[Polygon],
                   sites_xy: List[Tuple[float,float]],
                   radius: float,
                   bbox: Tuple[float,float,float,float],
                   out_png: str):
    xmin, xmax, ymin, ymax = bbox

    fig, ax = plt.subplots(figsize=(10, 7))
    # Building polygons
    patches_poly = []
    for p in polys:
        patches_poly.append(MplPolygon(np.array(p.exterior.coords), closed=True))
    pc = PatchCollection(patches_poly, facecolor="#e6e6e6", edgecolor="#9e9e9e", linewidths=0.6, alpha=0.8)
    ax.add_collection(pc)

    # Coverage discs
    patches_circ = []
    for (x, y) in sites_xy:
        patches_circ.append(Circle((x, y), radius, fill=False, linewidth=1.2))
    cc = PatchCollection(patches_circ, facecolor="none", edgecolor="#1f77b4", linewidths=1.2, alpha=0.9)
    ax.add_collection(cc)

    # BS positions
    xs = [x for (x, _) in sites_xy]
    ys = [y for (_, y) in sites_xy]
    ax.scatter(xs, ys, s=40, c="#d62728", marker="o", zorder=5, label="BS")

    # Numbered labels
    for i, (x, y) in enumerate(sites_xy):
        ax.text(x, y, f"{i+1}", fontsize=9, color="#d62728", ha="left", va="bottom")

    # Scenario bounding box
    ax.add_patch(MplPolygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]],
                            closed=True, fill=False, edgecolor="#333333", linewidth=1.2, linestyle="--"))

    ax.set_xlim(xmin - 20, xmax + 20)
    ax.set_ylim(ymin - 20, ymax + 20)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Selected BS (N={len(sites_xy)})  R={radius:.1f} m", fontsize=12)
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

# ----------------------------- Main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poly", type=str, default="data/osm.poly.xml")
    ap.add_argument("--radius", type=float, default=DEFAULT_R_COVER, help="Coverage radius (m)")
    ap.add_argument("--min_cand_sep", type=float, default=DEFAULT_MIN_CAND_SEP, help="Minimum separation between candidate sites (m)")
    ap.add_argument("--n_sites", type=int, default=7, help="Desired number of sites (capped by cap_by_area)")
    ap.add_argument("--pop", type=int, default=DEFAULT_POP)
    ap.add_argument("--gen", type=int, default=DEFAULT_GEN)
    ap.add_argument("--pmut", type=float, default=DEFAULT_PMUT)
    ap.add_argument("--pcx", type=float, default=DEFAULT_PCX)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_json", type=str, default="placement/w")
    ap.add_argument("--out_png", type=str, default="placement/selected_sites_plot.png")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1) Polygons and candidate sites
    polys, _types = parse_osm_polygons(args.poly)
    cand_sites = build_sites_from_vertices(polys, SCENARIO_BBOX, margin=0.0)
    cand_sites = deduplicate_by_distance(cand_sites, args.min_cand_sep)

    # 2) Capacity by area using fixed BBox
    bbox_w = SCENARIO_MAX_X - SCENARIO_MIN_X
    bbox_h = SCENARIO_MAX_Y - SCENARIO_MIN_Y
    bbox_area = max(1e-9, bbox_w * bbox_h)
    cap_by_area = max(1, int(math.floor(bbox_area / (math.pi * args.radius * args.radius))))
    max_sites_cap = 9
    n_sites = 9

    print(f"[INFO] Candidates (after dmin filter): {len(cand_sites)} | Sites cap: {max_sites_cap} | n_sites={n_sites}")
    print(f"[INFO] Radius R={args.radius:.1f} m | Fixed BBox = "
          f"({SCENARIO_MIN_X:.3f},{SCENARIO_MIN_Y:.3f})–({SCENARIO_MAX_X:.3f},{SCENARIO_MAX_Y:.3f})")

    if len(cand_sites) < n_sites:
        raise RuntimeError(f"Number of candidates ({len(cand_sites)}) < n_sites={n_sites}")

    cand_idx = np.arange(len(cand_sites), dtype=int)

    # 3) Initial population
    pop = [np.sort(np.random.choice(cand_idx, size=n_sites, replace=False)) for _ in range(args.pop)]
    fvals = [evaluate(ind, cand_sites, args.radius, SCENARIO_BBOX) for ind in pop]

    # 4) Minimal NSGA-II evolution
    for g in range(args.gen):
        # selection via fronts + crowding distance
        fronts = non_dom_sort(fvals)
        pool = []
        for fr in fronts:
            if not fr: continue
            cd = crowding_distance(fr, fvals)
            order = [fr[i] for i in np.argsort([-c for c in cd])]
            pool.extend(order)
            if len(pool) >= 12:
                break

        children = []
        while len(children) < args.pop:
            p1 = pop[random.choice(pool)]
            p2 = pop[random.choice(pool)]
            c1, c2 = crossover(p1, p2, args.pcx)
            c1 = mutate(c1, cand_idx, args.pmut)
            c2 = mutate(c2, cand_idx, args.pmut)
            children.extend([c1, c2])

        pop_all = pop + children[:args.pop]
        f_all  = [evaluate(ind, cand_sites, args.radius, SCENARIO_BBOX) for ind in pop_all]
        fronts = non_dom_sort(f_all)

        next_pop = []
        for fr in fronts:
            if len(next_pop) + len(fr) <= args.pop:
                next_pop.extend(fr)
            else:
                cd = crowding_distance(fr, f_all)
                order = [fr[i] for i in np.argsort([-c for c in cd])]
                next_pop.extend(order[:(args.pop - len(next_pop))])
            if len(next_pop) == args.pop:
                break

        pop = [pop_all[i] for i in next_pop]
        fvals = [f_all[i] for i in next_pop]

        if (g+1) % 10 == 0:
            best_idx = int(np.argmin([fv[0]+fv[1] for fv in fvals]))
            b1, b2 = fvals[best_idx]
            print(f"[Gen {g+1:03d}] best overlap={b1:.4f}  outside={b2:.4f}")

    # 5) Final selection: best sum (f1+f2)
    best_k = int(np.argmin([fv[0]+fv[1] for fv in fvals]))
    best_sol = pop[best_k]
    best_overlap, best_outside = fvals[best_k]
    best_sites_idx = [int(i) for i in best_sol]
    best_sites_xy = [tuple(map(float, cand_sites[i].xy)) for i in best_sol]

    # 6) Output
    sel_final = {
        "max_sites_cap": int(max_sites_cap),
        "feasible": True,
        "best_num_sites": int(n_sites),
        # outside_penalty used as proxy for best_frac_uncovered
        "best_frac_uncovered": float(best_outside),
        "best_sites_idx": best_sites_idx,
        "best_sites_xy": best_sites_xy,
        "best_azimuth_base_deg": 0.0,
        # explicit additional fields
        "best_overlap_penalty": float(best_overlap),
        "best_outside_penalty": float(best_outside),
        "radius_m": float(args.radius)
    }

    print("\n[FINAL SELECTION]")
    print(json.dumps(sel_final, indent=2))

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(sel_final, f, ensure_ascii=False, indent=2)

    # 7) PNG
    plot_selection(polys, best_sites_xy, args.radius, SCENARIO_BBOX, args.out_png)
    print(f"[OK] Files saved:\n - JSON: {os.path.abspath(args.out_json)}\n - PNG : {os.path.abspath(args.out_png)}")

if __name__ == "__main__":
    main()
