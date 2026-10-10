#!/usr/bin/env python3
"""Combine the dense strike with its longer solver tail and a staged lead-in."""
import csv
import sys

VOXEL_COUNT = 12 * 12 * 30


def read_capture(path):
    frames = {}
    with open(path, newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            frame = int(row[1])
            if row[0] == "F":
                frames[frame] = [row]
            elif row[0] == "V" and frame in frames:
                frames[frame].append(row)
    return {
        frame: rows for frame, rows in frames.items()
        if len(rows) == VOXEL_COUNT + 1
    }


def main(event_path, tail_path, output_path):
    event = read_capture(event_path)
    tail = read_capture(tail_path)
    if not event or 0 not in event or 60 not in event:
        raise SystemExit("event capture must contain complete frames 0 through 60")
    if not tail:
        raise SystemExit("tail capture contains no complete solver frames")
    merged = {frame: rows for frame, rows in event.items() if 0 <= frame <= 60}
    merged.update({frame: rows for frame, rows in tail.items() if frame > 60})
    if max(merged) < 1060:
        raise SystemExit("tail must simulate through at least 1.06 seconds")

    with open(output_path, "w", newline="") as stream:
        writer = csv.writer(stream)
        for frame in sorted(merged):
            writer.writerows(merged[frame])
    print(f"Wrote {len(merged)} solver states from {min(merged)} to {max(merged)} ms")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} EVENT.csv TAIL.csv OUTPUT.csv")
    main(*sys.argv[1:])
