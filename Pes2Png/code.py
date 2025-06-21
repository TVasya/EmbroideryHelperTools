#!/usr/bin/env python3
import os
import sys
from pyembroidery import EmbPattern
from pyembroidery.EmbConstant import STITCH, SEW_TO, NEEDLE_AT, COMMAND_MASK
from PIL import Image, ImageDraw

def pes_to_png_layers(pes_path, output_dir,
                      padding=10, stroke_width=1, scale=1.0):
    """
    Reads a .pes file and writes out one PNG per color-block,
    all aligned to a global canvas, then returns the scale factor.

    :param pes_path: Path to the input .pes
    :param output_dir: Where to save your layer PNGs
    :param padding: Pixels of white border around design (pre‐scale)
    :param stroke_width: Thickness of stitch lines (pre‐scale)
    :param scale:  Final image will be `scale`× size (e.g. 0.5 = half)
    :return: the scale factor actually used
    """
    os.makedirs(output_dir, exist_ok=True)
    pattern = EmbPattern(pes_path)
    blocks  = list(pattern.get_as_colorblocks())

    # Pass 1: global bounds
    all_x, all_y = [], []
    for stitches, _thread in blocks:
        for x, y, cmd in stitches:
            if (cmd & COMMAND_MASK) in (STITCH, SEW_TO, NEEDLE_AT):
                all_x.append(x)
                all_y.append(y)
    if not all_x:
        raise ValueError("No sew commands found in .pes")

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    raw_w  = max_x - min_x + 2*padding
    raw_h  = max_y - min_y + 2*padding
    canvas_w = int(raw_w * scale)
    canvas_h = int(raw_h * scale)

    # Compute a scaled stroke width (at least 1px)
    sw = max(1, int(round(stroke_width * scale)))

    # Pass 2: draw each block
    for idx, (stitches, _thread) in enumerate(blocks, start=1):
        segments, prev = [], None
        for x, y, cmd in stitches:
            if (cmd & COMMAND_MASK) in (STITCH, SEW_TO, NEEDLE_AT):
                if prev is not None:
                    segments.append((prev, (x, y)))
                prev = (x, y)
            else:
                prev = None
        if not segments:
            continue

        img  = Image.new("RGB", (canvas_w, canvas_h), "white")
        draw = ImageDraw.Draw(img)
        for (x1, y1), (x2, y2) in segments:
            p1 = ((x1 - min_x + padding)*scale, (y1 - min_y + padding)*scale)
            p2 = ((x2 - min_x + padding)*scale, (y2 - min_y + padding)*scale)
            draw.line([p1, p2], fill="black", width=sw)

        out_path = os.path.join(output_dir, f"layer_{idx:02d}.png")
        img.save(out_path)
        print(f"→ Saved layer {idx:02d} at {out_path}")

    return scale

if __name__ == "__main__":


    pes_file     = input("Enter path to the .pes file: ")
    if not os.path.isfile(pes_file):
        print(f"File '{pes_file}' not found.")
        sys.exit(1)
    output_folder = 'output_folder'
    scale_arg     = 0.2

    used_scale = pes_to_png_layers(pes_file, output_folder, scale=scale_arg)
    print(f"\nDone! Scale factor returned: {used_scale}")

