#####
# Original Sources: 
# https://github.com/michaeltrs/DeepSatModels/blob/782f7363e04b9e9d3dfc18a1ce6609157613b5cf/data/MTLCC/tfrecord2tif.py
# https://github.com/MarcCoru/MTLCC/blob/master/tfrecord2tif.py
# Adaptation: Joseph Metcalfe, converting output from .tif to .npy
# Purpose: Create numpy versions of MTLCC dataset files
#####

"""
tfrecord2npy.py
---------------
Convert MTLCC .tfrecord.gz files to numpy arrays.

Produces three .npy files per tile:
  <id>_sits.npy    (T, B, H, W)  int16  -- all bands concatenated, bands ordered [x10 | x20_up | x60_up]
  <id>_labels.npy  (H, W)        int8   -- label map
  <id>_dates.npy   (T, 2)        int16  -- columns: [year, doy]  (real calendar DOY, not an index)

Usage:
  # Single file
  python tfrecord2npy.py path/to/file.tfrecord.gz --outdir npy_output

  # Whole directory
  python tfrecord2npy.py path/to/data16/ --outdir npy_output

  # With dates CSV to resolve real calendar dates
  python tfrecord2npy.py path/to/data16/ --outdir npy_output --dates path/to/dates.csv
"""

from S2parserUpdated import S2parser
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime


# ---------------------------------------------------------------------------
# Dates CSV loading
# ---------------------------------------------------------------------------

def load_dates_csv(csv_path):
    """
    Load the acquisition dates CSV.

    The CSV has columns: [index, date, sat, sat_id]
    where the row index corresponds to the time-step index stored in the
    tfrecord doy field.

    Returns a dict mapping  index (int) -> (year: int, doy: int)
    where doy is the real calendar day-of-year.
    """
    df = pd.read_csv(csv_path, index_col=0)
    index_to_date = {}
    for idx, row in df.iterrows():
        dt = datetime.strptime(str(row["date"]), "%Y-%m-%d")
        index_to_date[int(idx)] = (dt.year, dt.timetuple().tm_yday)
    return index_to_date


def resolve_dates(doy_indices, index_to_date):
    """
    NON FUNCTIONAL
    Convert an array of date indices into a (T, 2) array of [year, doy].

    Parameters
    ----------
    doy_indices   : np.ndarray, shape (T,) -- integer indices into the dates CSV
    index_to_date : dict from load_dates_csv

    Returns
    -------
    np.ndarray, shape (T, 2), dtype int16 -- columns: [year, real_doy]
    """
    resolved = []
    for idx in doy_indices:
        if int(idx) in index_to_date:
            resolved.append(index_to_date[int(idx)])
        else:
            # Fallback: store index as-is with year=0 to flag unresolved entries
            resolved.append((0, int(idx)))

    exit() # make whoever runs this find this function to fix it if used

    return np.array(resolved, dtype=np.int16)


# ---------------------------------------------------------------------------
# TFRecord reading (TF2)
# ---------------------------------------------------------------------------

def tfrecord2npy(path):
    """
    Read a single .tfrecord.gz and return raw numpy arrays.
    Padded time steps (doy == 0) are removed.
    """
    parser = S2parser()

    dataset = tf.data.TFRecordDataset([path], compression_type="GZIP")
    dataset = dataset.map(parser.parse_example, num_parallel_calls=1)

    x10, x20, x60, doy, year, labels = next(iter(dataset))
    x10, x20, x60, doy, year, labels = [t.numpy() for t in (x10, x20, x60, doy, year, labels)]

    # Remove padded (-1 doy) time steps
    mask   = doy > 0
    x10    = x10[mask]
    x20    = x20[mask]
    x60    = x60[mask]
    doy    = doy[mask]
    year   = year[mask]
    labels = labels[mask]

    return x10, x20, x60, doy, year, labels


# ---------------------------------------------------------------------------
# Band concatenation
# ---------------------------------------------------------------------------

def build_sits(x10, x20, x60):
    """
    Upsample x20 and x60 to the x10 spatial resolution via nearest-neighbour
    repeat, then concatenate all bands into a single SITS array.

    Input shapes:
        x10 : (T, H,   W,   B10)
        x20 : (T, H/2, W/2, B20)
        x60 : (T, H/6, W/6, B60)

    Returns
    -------
    sits : np.ndarray, shape (T, B_total, H, W), dtype int16
        Band order: [x10 | x20_upsampled | x60_upsampled]
    """
    T, H, W, _ = x10.shape

    x20_up = np.repeat(np.repeat(x20, 2, axis=1), 2, axis=2)[:, :H, :W, :]
    x60_up = np.repeat(np.repeat(x60, 6, axis=1), 6, axis=2)[:, :H, :W, :]

    sits = np.concatenate([x10, x20_up, x60_up], axis=-1).transpose(0, 3, 1, 2)
    return sits.astype(np.int16)


# ---------------------------------------------------------------------------
# Single-file conversion
# ---------------------------------------------------------------------------

def convert(tfrecord_path, outdir, index_to_date=None):
    """
    Convert one .tfrecord.gz to three .npy files.

    If index_to_date is provided, doy values in the tfrecord are treated as
    indices into the dates CSV and resolved to real calendar [year, doy].
    Otherwise, [year, doy] are stored as-is from the tfrecord.
    """
    os.makedirs(outdir, exist_ok=True)
    record_id = os.path.basename(tfrecord_path).replace(".tfrecord.gz", "")

    print(f"Reading  {tfrecord_path} ...")
    x10, x20, x60, doy, year, labels = tfrecord2npy(tfrecord_path)

    # SITS: (T, B, H, W)
    sits = build_sits(x10, x20, x60)

    # Labels: (H, W) — static per tile, use first time step
    lbl = labels[0]
    if lbl.ndim == 3:
        lbl = lbl[:, :, 0]
    lbl = lbl.astype(np.int8)

    # Dates: (T, 2) columns [year, doy]
    index_to_date = None # disable use of observations csv dates, have all info already
    if index_to_date is not None:
        dates = resolve_dates(doy, index_to_date)
    else:
        dates = np.stack([year, doy], axis=-1).astype(np.int16)

    sits_path   = os.path.join(outdir, f"{record_id}_sits.npy")
    labels_path = os.path.join(outdir, f"{record_id}_labels.npy")
    dates_path  = os.path.join(outdir, f"{record_id}_dates.npy")

    np.save(sits_path,   sits)
    np.save(labels_path, lbl)
    np.save(dates_path,  dates)

    print(f"  SITS   -> {sits_path}    shape={sits.shape}  dtype={sits.dtype}")
    print(f"  Labels -> {labels_path}  shape={lbl.shape}   dtype={lbl.dtype}")
    print(f"  Dates  -> {dates_path}   shape={dates.shape} dtype={dates.dtype}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Convert MTLCC .tfrecord.gz files to numpy arrays."
    )
    ap.add_argument(
        "--indir", default="D:/Data/MTLCC/data_IJGI18/datasets/full/480/data16",
        help="Path to a single .tfrecord.gz file, or a directory of files."
    )
    ap.add_argument(
        "--outdir", default="D:/Data/MTLCC/data_IJGI18/datasets/fullNpy/480/data16",
        help="Output directory for .npy files (default: npy_output)."
    )
    ap.add_argument(
        "--dates", default="D:/Data/MTLCC/data_IJGI18/observations.csv",
        help="Path to acquisition dates CSV. If provided, doy values are "
             "resolved to real calendar dates via the CSV index."
    )
    args = ap.parse_args()

    index_to_date = None
    if args.dates:
        print(f"Loading dates CSV from {args.dates} ...")
        index_to_date = load_dates_csv(args.dates)
        print(f"  Loaded {len(index_to_date)} date entries.")

    if os.path.isdir(args.indir):
        files = sorted([
            os.path.join(args.indir, f)
            for f in os.listdir(args.indir)
            if f.endswith(".tfrecord.gz")
        ])
        if not files:
            print(f"No .tfrecord.gz files found in {args.indir}")
            return
        print(f"Found {len(files)} file(s) to convert.")
        for f in files:
            try:
                convert(f, args.outdir, index_to_date)
            except Exception as e:
                print(f"  ERROR on {f}: {e} — skipping.")
    else:
        convert(args.indir, args.outdir, index_to_date)

    print("\nDone.")


if __name__ == "__main__":
    main()