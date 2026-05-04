import argparse
import json
from typing import List, Dict, Tuple
import numpy as np


# 8x8 S-box construction from 4x4 components over GF(2^4)
# Based on the construction described in:
# https://www.mdpi.com/2410-387X/9/4/67

# Configuration python nonlin.py --mode s01
# All configuration python nonlin.py --mode both

# Representative 4x4 S-boxes reported for f1(x)=x^4+x+1
S6 = [0x0, 0x1, 0xE, 0x9, 0xB, 0xD, 0x7, 0x6, 0x8, 0x3, 0xA, 0x4, 0xC, 0x5, 0x2, 0xF]
S7 = [0x0, 0x1, 0xD, 0xB, 0xE, 0x9, 0x6, 0x7, 0xA, 0x4, 0xF, 0x2, 0x8, 0x3, 0x5, 0xC]
S8 = [0x0, 0x1, 0x9, 0xE, 0xD, 0xB, 0x7, 0x6, 0xF, 0x2, 0xC, 0x5, 0xA, 0x4, 0x3, 0x8]

POLY_F1 = 0x13  # x^4 + x + 1
SIZE = 256
MASKS = list(range(1, 256))
PARITY = np.array([bin(i).count('1') & 1 for i in range(256)], dtype=np.uint8)


def gf16_mul(a: int, b: int, poly: int = POLY_F1) -> int:
    """Multiply two 4-bit values in GF(2^4) modulo the irreducible polynomial."""
    res = 0
    x = a & 0xF
    y = b & 0xF
    for _ in range(4):
        if y & 1:
            res ^= x
        y >>= 1
        carry = x & 0x8
        x = (x << 1) & 0xF
        if carry:
            x ^= (poly & 0xF)
    return res & 0xF


def build_sbox(sa: List[int], sb: List[int], sc: List[int], sd: List[int], poly: int = POLY_F1) -> List[int]:
    sbox = [0] * 256
    for x in range(256):
        xh = (x >> 4) & 0xF
        xl = x & 0xF

        if xh != 0:
            yh = gf16_mul(sa[xl], xh, poly)
        else:
            yh = sb[xl]

        if yh != 0:
            yl = sc[gf16_mul(xh, yh, poly)]
        else:
            yl = sd[xh]

        y = ((yh << 4) | yl) ^ 0x01
        sbox[x] = y
    return sbox


def is_bijective(sbox: List[int]) -> bool:
    return sorted(sbox) == list(range(256))


def fwht_int(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.int32).copy()
    h = 1
    n = out.shape[0]
    while h < n:
        for i in range(0, n, h * 2):
            x = out[i:i+h].copy()
            y = out[i+h:i+2*h].copy()
            out[i:i+h] = x + y
            out[i+h:i+2*h] = x - y
        h <<= 1
    return out


def bool_nl(bits01: np.ndarray) -> int:
    pm = 1 - 2 * bits01.astype(np.int16)
    spec = fwht_int(pm)
    return int(128 - np.max(np.abs(spec)) // 2)


def coordinate_nls(sbox: List[int]) -> List[int]:
    vals = np.array(sbox, dtype=np.uint8)
    return [bool_nl(((vals >> b) & 1).astype(np.uint8)) for b in range(8)]


def vectorial_nl(sbox: List[int]) -> int:
    vals = np.array(sbox, dtype=np.uint8)
    best = 999
    for mask in MASKS:
        bits = PARITY[np.bitwise_and(vals, mask)]
        nl = bool_nl(bits)
        if nl < best:
            best = nl
    return int(best)


def fixed_points(sbox: List[int]) -> List[int]:
    return [x for x in range(256) if sbox[x] == x]


def opposite_fixed_points(sbox: List[int]) -> List[int]:
    return [x for x in range(256) if sbox[x] == (x ^ 0xFF)]


def sac_matrix(sbox: List[int]) -> List[List[float]]:
    sac = [[0.0] * 8 for _ in range(8)]
    for i in range(8):
        dx = 1 << i
        for j in range(8):
            changed = 0
            for x in range(256):
                if ((sbox[x] ^ sbox[x ^ dx]) >> j) & 1:
                    changed += 1
            sac[i][j] = changed / 256.0
    return sac


def sac_average(sac: List[List[float]]) -> float:
    return sum(sum(row) for row in sac) / 64.0


def bic_sac(sbox: List[int]) -> Tuple[List[Dict], float]:
    rows = []
    avgs = []
    for input_bit in range(8):
        dx = 1 << input_bit
        delta_bits = np.array([sbox[x] ^ sbox[x ^ dx] for x in range(256)], dtype=np.uint8)
        for j in range(8):
            for k in range(j + 1, 8):
                pair_func = (((delta_bits >> j) & 1) ^ ((delta_bits >> k) & 1)).astype(np.uint8)
                p = float(np.mean(pair_func))
                rows.append({"input_bit": input_bit, "out_pair": [j, k], "value": p})
                avgs.append(p)
    return rows, float(sum(avgs) / len(avgs)) if avgs else 0.0


def bic_nl(sbox: List[int]) -> Tuple[List[Dict], float]:
    rows = []
    vals = np.array(sbox, dtype=np.uint8)
    all_vals = []
    for j in range(8):
        bj = ((vals >> j) & 1).astype(np.uint8)
        for k in range(j + 1, 8):
            bk = ((vals >> k) & 1).astype(np.uint8)
            f = (bj ^ bk).astype(np.uint8)
            nl = bool_nl(f)
            rows.append({"out_pair": [j, k], "nl": nl})
            all_vals.append(nl)
    return rows, float(sum(all_vals) / len(all_vals)) if all_vals else 0.0


def ddt(sbox: List[int]) -> List[List[int]]:
    table = [[0] * 256 for _ in range(256)]
    for da in range(256):
        for x in range(256):
            db = sbox[x] ^ sbox[x ^ da]
            table[da][db] += 1
    return table


def differential_uniformity_and_dap(sbox: List[int]) -> Tuple[int, float]:
    table = ddt(sbox)
    du = 0
    for da in range(1, 256):
        row_max = max(table[da])
        if row_max > du:
            du = row_max
    return du, du / 256.0


def lat_bias_max(sbox: List[int]) -> int:
    vals = np.array(sbox, dtype=np.uint8)
    best = 0
    xs = np.arange(256, dtype=np.uint16)
    for a in range(1, 256):
        xa = PARITY[np.bitwise_and(xs, a)]
        for b in range(1, 256):
            yb = PARITY[np.bitwise_and(vals, b)]
            eq = (xa == yb)
            bias = abs(int(np.sum(eq)) - 128)
            if bias > best:
                best = bias
    return best


def lap(sbox: List[int]) -> float:
    # |Pr[a·x = b·S(x)] - 1/2| = bias/256
    return lat_bias_max(sbox) / 256.0


def algebraic_degree_coordinate_truth(bits01: np.ndarray) -> int:
    anf = bits01.astype(np.uint8).copy()
    n = 8
    for i in range(n):
        step = 1 << i
        for mask in range(256):
            if mask & step:
                anf[mask] ^= anf[mask ^ step]
    deg = 0
    for idx, coef in enumerate(anf):
        if coef:
            wt = bin(idx).count('1')
            if wt > deg:
                deg = wt
    return deg


def algebraic_degree_sbox(sbox: List[int]) -> int:
    vals = np.array(sbox, dtype=np.uint8)
    degs = [algebraic_degree_coordinate_truth(((vals >> b) & 1).astype(np.uint8)) for b in range(8)]
    return min(degs)


def analyze_sbox(sbox: List[int]) -> Dict:
    coord = coordinate_nls(sbox)
    vec_nl = vectorial_nl(sbox)
    sac = sac_matrix(sbox)
    bic_sac_rows, bic_sac_avg = bic_sac(sbox)
    bic_nl_rows, bic_nl_avg = bic_nl(sbox)
    du, dap = differential_uniformity_and_dap(sbox)
    lp = lap(sbox)
    fps = fixed_points(sbox)
    ofps = opposite_fixed_points(sbox)
    return {
        "is_bijective_8x8": is_bijective(sbox),
        "coord_nls": coord,
        "min_coord_nl": int(min(coord)),
        "max_coord_nl": int(max(coord)),
        "avg_coord_nl": float(sum(coord) / 8.0),
        "vectorial_nl": int(vec_nl),
        "algebraic_degree_min_over_coordinates": int(algebraic_degree_sbox(sbox)),
        "fixed_points_count": len(fps),
        "fixed_points": fps,
        "opposite_fixed_points_count": len(ofps),
        "opposite_fixed_points": ofps,
        "sac_matrix": sac,
        "sac_avg": float(sac_average(sac)),
        "bic_nl_avg": float(bic_nl_avg),
        "bic_nl_rows": bic_nl_rows,
        "bic_sac_avg": float(bic_sac_avg),
        "bic_sac_rows": bic_sac_rows,
        "differential_uniformity": int(du),
        "dap": float(dap),
        "lap": float(lp),
        "sbox": sbox,
    }


def print_human_readable(title: str, analysis: Dict):
    print(f"=== {title} ===")
    print(f"bijective 8x8: {analysis['is_bijective_8x8']}")
    print(f"coord NLs     : {analysis['coord_nls']}")
    print(f"min/max/avg NL: {analysis['min_coord_nl']} / {analysis['max_coord_nl']} / {analysis['avg_coord_nl']:.2f}")
    print(f"vectorial NL  : {analysis['vectorial_nl']}")
    print(f"alg degree    : {analysis['algebraic_degree_min_over_coordinates']}")
    print(f"SAC avg       : {analysis['sac_avg']:.4f}")
    print(f"BIC-NL avg    : {analysis['bic_nl_avg']:.4f}")
    print(f"BIC-SAC avg   : {analysis['bic_sac_avg']:.4f}")
    print(f"DU / DAP      : {analysis['differential_uniformity']} / {analysis['dap']:.4f}")
    print(f"LAP           : {analysis['lap']:.4f}")
    print(f"FP / OFP      : {analysis['fixed_points_count']} / {analysis['opposite_fixed_points_count']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Construct and analyze 8x8 S-boxes with high coordinate nonlinearity.")
    parser.add_argument("--mode", choices=["s01", "s02", "both"], default="both")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    outputs = {}

    if args.mode in ("s01", "both"):
        s01 = build_sbox(S8, S8, S8, S8, POLY_F1)
        outputs["s01"] = analyze_sbox(s01)
    if args.mode in ("s02", "both"):
        s02 = build_sbox(S6, S8, S7, S8, POLY_F1)
        outputs["s02"] = analyze_sbox(s02)

    if args.json:
        print(json.dumps(outputs, ensure_ascii=False))
    else:
        for name, analysis in outputs.items():
            print_human_readable(name.upper(), analysis)


if __name__ == "__main__":
    main()