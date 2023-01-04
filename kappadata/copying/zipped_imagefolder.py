import os
import shutil
from pathlib import Path
import zipfile
import joblib

def create_zipped_imagefolder_classwise(src, dst):
    src_path = Path(src).expanduser()
    assert src_path.exists(), f"src_path '{src_path}' doesn't exist"
    dst_path = Path(dst).expanduser()
    dst_path.mkdir(exist_ok=True, parents=True)

    for item in os.listdir(src_path):
        src_uri = src_path / item
        if not src_uri.is_dir():
            continue
        shutil.make_archive(
            base_name=dst_path / item,
            format="zip",
            root_dir=src_uri,
        )

def _unzip(src, dst):
    with zipfile.ZipFile(src) as f:
        f.extractall(dst)

def unzip_imagefolder_classwise(src, dst, num_workers=0):
    src_path = Path(src).expanduser()
    assert src_path.exists(), f"src_path '{src_path}' doesn't exist"
    dst_path = Path(dst).expanduser()
    dst_path.mkdir(exist_ok=True, parents=True)

    # compose jobs
    jobargs = []
    for item in os.listdir(src_path):
        assert item.endswith(".zip")
        dst_uri = (dst_path / item).with_suffix("")
        src_uri = src_path / item
        jobargs.append((src_uri, dst_uri))

    # run jobs
    if num_workers <= 1:
        for src, dst in jobargs:
            _unzip(src, dst)
    else:
        jobs = joblib.delayed(_unzip)(src, dst)
        pool = joblib.Parallel(n_jobs=num_workers)
        pool(jobs)
