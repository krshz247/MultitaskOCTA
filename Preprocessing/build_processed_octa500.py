"""Build Datasets/processed_OCTA500 for the joint RV + FAZ pipeline from the raw OCTA-500 release.

Python port of Preprocessing/pre_dis.m (boundary heatmap + signed distance map), plus the
folder assembly that dataset.py expects but no script in the repo produced:

    <out>/image_surface/<N|D|M>_<3|6>_<id>.bmp        OCTA projection map (--projection, default ILM_OPL = B5)
    <out>/gt_full/mask/<name>.bmp                     FAZ U vessel (only used to derive paths)
    <out>/gt_vessel/mask/<name>.bmp                   GT_LargeVessel [U GT_Capillary, --vessel] (minus FAZ pixels)
    <out>/gt_faz/mask/<name>.bmp                      GT_FAZ
    <out>/gt_{vessel,faz}/contour_img/<name>.png      Gaussian boundary heatmap (pre_dis.m `contour`)
    <out>/gt_{vessel,faz}/dist_signed11_img/<name>.png  signed distance map (pre_dis.m `s_dis11`, (x+1)/2)

Only Healthy (NORMAL -> N), DR (-> D) and AMD (-> M) eyes are kept (364 of 500 scans).
Usage: python Preprocessing/build_processed_octa500.py --octa500 ../OCTA500 --out Datasets/processed_OCTA500/combined
"""
import argparse
import os
import re
import zipfile
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

CLASS_PREFIX = {"NORMAL": "N", "DR": "D", "AMD": "M"}
SIGMA = 1.6       # pre_dis.m
AMPLITUDE = 0.06  # Gauss2D amplitude in the 2023 pre_dis.m
KERNEL_RADIUS = 12
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_labels(xlsx_path):
    """Return {id: disease} from an OCTA-500 'Text labels' sheet (stdlib-only xlsx reader)."""
    with zipfile.ZipFile(xlsx_path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
        rows = ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter("{%s}row" % NS["m"])
        out = {}
        for row in list(rows)[1:]:
            cells = {}
            for c in row.findall("m:c", NS):
                v = c.find("m:v", NS)
                if v is None:
                    continue
                val = shared[int(v.text)] if c.get("t") == "s" else v.text
                cells[re.match(r"[A-Z]+", c.get("r")).group()] = val
            if "A" in cells and "E" in cells:
                out[int(float(cells["A"]))] = cells["E"].strip().upper()
        return out


def rescale(a, lo=0.0, hi=1.0):
    """MATLAB rescale(): min -> lo, max -> hi (constant input -> lo)."""
    mn, mx = a.min(), a.max()
    if mx == mn:
        return np.full_like(a, lo, dtype=np.float64)
    return lo + (a - mn) / (mx - mn) * (hi - lo)


def boundary_heatmap(mask):
    """pre_dis.m `contour`: noisy-or sum of Gaussians centred on 4-connected perimeter pixels.

    1 - prod_j(1 - A*g_j) = 1 - exp(sum_j log(1 - A*g_j)), and the inner sum is a convolution of the
    perimeter indicator with log(1 - A*g(d)), so the per-pixel MATLAB loop becomes one filter2D call.
    """
    m = mask > 0
    eroded = cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), borderValue=0)
    perim = (m & (eroded == 0)).astype(np.float64)  # bwperim(mask, 4)
    r = np.arange(-KERNEL_RADIUS, KERNEL_RADIUS + 1)
    d2 = r[None, :] ** 2 + r[:, None] ** 2
    kernel = np.log1p(-AMPLITUDE * np.exp(-d2 / SIGMA ** 2))
    s = cv2.filter2D(perim, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    zsum = 1.0 - np.exp(s)
    heat = rescale(zsum, 0, 1)
    heat[heat < 0.001] = 0
    return perim, heat


def signed_distance(mask, perim):
    """pre_dis.m `s_dis11` (thesis Eq. 3.8: negative inside, ~0 on the boundary, positive outside).

    s_dis11 lies in [-1, 1]. pre_dis.m passed it straight to imwrite(), which clips doubles to [0, 1] and
    discards the whole inside half, so it is shifted to [0, 1] here instead: 0.5 = boundary, <0.5 inside.
    """
    fg = mask > 0
    dist_to_fg = distance_transform_edt(~fg)             # bwdist(mask)
    dist_to_contour = distance_transform_edt(perim == 0)  # bwdist(contour00)
    dis = rescale(dist_to_fg)
    s_dis = rescale(-(dist_to_contour - dist_to_fg), -1, 0)
    return (dis + s_dis + 1.0) / 2.0


def to_u8(x):
    return np.round(np.clip(x, 0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--octa500", required=True, help="raw OCTA500 folder (contains Label/, OCTA_3mm/, OCTA_6mm/)")
    ap.add_argument("--out", required=True, help="output root, e.g. Datasets/processed_OCTA500/combined")
    ap.add_argument("--projection", default="ILM_OPL", choices=["ILM_OPL", "FULL", "OPL_BM"],
                    help="OCTA en-face projection used as input: ILM_OPL = B5 superficial (thesis), FULL = B4, OPL_BM = B6 deep")
    ap.add_argument("--vessel", default="large_capillary", choices=["large_capillary", "large"],
                    help="vessel ground truth: GT_LargeVessel U GT_Capillary, or GT_LargeVessel only")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N scans (debug)")
    args = ap.parse_args()

    label_dir = os.path.join(args.octa500, "Label")
    diseases = {}
    for f in ("Text labels.xlsx", "Text labels copy.xlsx"):  # 3mm (10301-10500), 6mm (10001-10300)
        diseases.update(read_labels(os.path.join(label_dir, "TextLabels", f)))

    for d in ("image_surface", "gt_full/mask", "gt_vessel/mask", "gt_faz/mask",
              "gt_vessel/contour_img", "gt_faz/contour_img",
              "gt_vessel/dist_signed11_img", "gt_faz/dist_signed11_img"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    ids = sorted(i for i, dz in diseases.items() if dz in CLASS_PREFIX)
    if args.limit:
        ids = ids[:args.limit]
    overlap_total, overlap_scans, written = 0, 0, 0
    for i in ids:
        area = "3" if i >= 10301 else "6"
        name = f"{CLASS_PREFIX[diseases[i]]}_{area}_{i}"
        img = cv2.imread(os.path.join(args.octa500, f"OCTA_{area}mm", "Projection Maps", f"OCTA({args.projection})", f"{i}.bmp"), cv2.IMREAD_GRAYSCALE)
        faz = cv2.imread(os.path.join(label_dir, "GT_FAZ", f"{i}.bmp"), cv2.IMREAD_GRAYSCALE) > 127
        large = cv2.imread(os.path.join(label_dir, "GT_LargeVessel", f"{i}.bmp"), cv2.IMREAD_GRAYSCALE) > 127
        vessel = large
        if args.vessel == "large_capillary":
            vessel = large | (cv2.imread(os.path.join(label_dir, "GT_Capillary", f"{i}.bmp"), cv2.IMREAD_GRAYSCALE) > 127)
        # dataset.py builds background = 1 - (faz + vessel); overlapping pixels would make it negative.
        overlap = int((vessel & faz).sum())
        if overlap:
            overlap_total += overlap
            overlap_scans += 1
            vessel &= ~faz
        assert faz.any() and vessel.any(), f"empty mask for {name}"

        cv2.imwrite(os.path.join(args.out, "image_surface", name + ".bmp"), img)
        cv2.imwrite(os.path.join(args.out, "gt_full/mask", name + ".bmp"), ((faz | vessel) * 255).astype(np.uint8))
        for key, m in (("faz", faz), ("vessel", vessel)):
            m8 = m.astype(np.uint8) * 255
            perim, heat = boundary_heatmap(m8)
            cv2.imwrite(os.path.join(args.out, f"gt_{key}/mask", name + ".bmp"), m8)
            cv2.imwrite(os.path.join(args.out, f"gt_{key}/contour_img", name + ".png"), to_u8(heat))
            cv2.imwrite(os.path.join(args.out, f"gt_{key}/dist_signed11_img", name + ".png"), to_u8(signed_distance(m8, perim)))
        written += 1
    print(f"wrote {written} scans to {args.out}; FAZ/vessel overlap removed in {overlap_scans} scans ({overlap_total} px)")


if __name__ == "__main__":
    main()
