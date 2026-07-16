"""Download Stanford2D3D noXYZ area archives from Redivis.

This requires access to the Stanford 2D-3D-Semantics Dataset on Redivis.
Accept the dataset license in a browser first, then authenticate either with
the Redivis OAuth flow or by setting REDIVIS_API_TOKEN.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TARGET_FILES = {
    "area_1_no_xyz.tar",
    "area_2_no_xyz.tar",
    "area_3_no_xyz.tar",
    "area_4_no_xyz.tar",
    "area_5a_no_xyz.tar",
    "area_5b_no_xyz.tar",
    "area_6_no_xyz.tar",
}


def properties(obj):
    props = getattr(obj, "properties", None)
    if isinstance(props, dict):
        return props
    if isinstance(obj, dict):
        return obj
    try:
        got = obj.get()
        if isinstance(got, dict):
            return got
        props = getattr(got, "properties", None)
        if isinstance(props, dict):
            return props
    except Exception:
        pass
    return {}


def prop(obj, *names, default=""):
    props = properties(obj)
    for name in names:
        if name in props:
            return props[name]
    return default


def file_name(file_obj) -> str:
    value = prop(
        file_obj,
        "name",
        "fileName",
        "file_name",
        "filename",
        "path",
        "label",
        "qualifiedReference",
        default="",
    )
    return str(value)


def find_dataset(redivis):
    org = redivis.organization("sdss")
    guesses = (
        "stanford_2d_3d_semantics_dataset_2d_3d_s:f304",
        "stanford_2d_3d_semantics_dataset:f304",
        "2d_3d_s:f304",
        "f304-a3vhsvcaf",
    )
    for guess in guesses:
        try:
            dataset = org.dataset(guess)
            dataset.get()
            return dataset
        except Exception:
            continue

    for dataset in org.list_datasets():
        text = str(properties(dataset)).lower()
        if "f304" in text or "stanford 2d" in text or "2d-3d-s" in text:
            return dataset
    raise RuntimeError(
        "Could not locate Stanford 2D-3D-S dataset under Redivis organization 'sdss'. "
        "Make sure you accepted the license at https://sdss.redivis.com/datasets/f304-a3vhsvcaf"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Stanford2D3D noXYZ archives from Redivis.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("downloads/Stanford2D3D"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only list matching files; do not download.",
    )
    parser.add_argument(
        "--dump-files",
        action="store_true",
        help="Print all visible Redivis table/file names for debugging.",
    )
    args = parser.parse_args()

    import redivis

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = find_dataset(redivis)
    print(f"[info] dataset: {prop(dataset, 'name', 'label', 'qualifiedReference', default=str(dataset))}")

    found = {}
    for table in dataset.list_tables():
        table_name = prop(table, "name", "qualifiedReference", default=str(table))
        try:
            files = table.list_files()
        except Exception as exc:
            print(f"[warn] cannot list files for table {table_name}: {exc}")
            continue
        if args.dump_files:
            print(f"[table] {table_name}")
        for file_obj in files:
            name = file_name(file_obj)
            if args.dump_files:
                print(f"  [file] {name}")
            basename = Path(name).name
            if name in TARGET_FILES or basename in TARGET_FILES:
                found[name] = file_obj
                print(f"[match] {name} from table {table_name}")

    missing = sorted(TARGET_FILES - set(found))
    if missing:
        print("[warn] missing expected files:")
        for name in missing:
            print(f"  {name}")

    if args.list_only:
        return 0 if found else 1

    for name in sorted(found):
        target = output_dir / name
        if target.exists() and not args.overwrite:
            print(f"[skip] {target}")
            continue
        print(f"[download] {name} -> {output_dir}")
        found[name].download(str(output_dir), overwrite=args.overwrite)

    print(f"[done] downloaded {len(found)} files to {output_dir}")
    return 0 if not missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
