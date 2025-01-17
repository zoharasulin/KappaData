import os
import shutil
import zipfile
from collections import namedtuple
from pathlib import Path

import joblib

from kappadata.utils.logging import log

CopyImageFolderResult = namedtuple("CopyFolderResult", "was_copied was_deleted was_zip was_zip_classwise")


def _check_src_path(src_path):
    if src_path.exists() and src_path.is_dir():
        return True
    if src_path.with_suffix(".zip").exists():
        return True
    return False


def copy_imagefolder_from_global_to_local(global_path, local_path, relative_path=None, num_workers=0, log_fn=None):
    if not isinstance(global_path, Path):
        global_path = Path(global_path).expanduser()
    if not isinstance(local_path, Path):
        local_path = Path(local_path).expanduser()
    if relative_path is not None and not isinstance(relative_path, Path):
        # relative path can be .zip -> relative path is always without .zip as dst_path is never .zip
        relative_path = Path(relative_path).with_suffix("")

    # check src_path exists (src_path can be folder or .zip)
    src_path = global_path / relative_path if relative_path is not None else global_path
    assert _check_src_path(src_path), f"invalid src_path (can be folder or folder of zips or zip) '{src_path}'"

    # if dst_path exists:
    # - autocopy start/end file exists -> already copied -> do nothing
    # - autocopy start file exists && autocopy end file doesn't exist -> incomplete copy -> delete and copy again
    # - autocopy start file doesn't exists -> manually copied dataset -> do nothing
    dst_path = local_path / relative_path if relative_path is not None else local_path
    start_copy_file = dst_path / "autocopy_start.txt"
    end_copy_file = dst_path / "autocopy_end.txt"
    was_deleted = False
    if dst_path.exists():
        if start_copy_file.exists():
            if end_copy_file.exists():
                # already automatically copied -> do nothing
                log(log_fn, f"dataset was already automatically copied '{dst_path}'")
                return CopyImageFolderResult(
                    was_copied=False,
                    was_deleted=False,
                    was_zip=False,
                    was_zip_classwise=False,
                )
            else:
                # incomplete copy -> delete and copy again
                log(log_fn, f"found incomplete automatic copy in '{dst_path}' -> deleting folder")
                shutil.rmtree(dst_path)
                was_deleted = True
                dst_path.mkdir()
        else:
            log(log_fn, f"using manually copied dataset '{dst_path}'")
            return CopyImageFolderResult(was_copied=False, was_deleted=False, was_zip=False, was_zip_classwise=False)
    else:
        dst_path.mkdir(parents=True)

    # create start_copy_file
    with open(start_copy_file, "w") as f:
        f.write("this file indicates that an attempt to copy the dataset automatically was started")

    # copy
    was_zip = False
    was_zip_classwise = False
    if src_path.exists() and src_path.is_dir():
        # check if subfolders are zips (allow files such as a README inside the folder)
        items = os.listdir(src_path)
        zips = [item for item in items if item.endswith(".zip")]
        if len(zips) > 0 and len(zips) >= len(items) // 2:
            # extract all zip folders into dst (e.g. imagenet1k/train/n01558993.zip
            was_zip_classwise = True
            log(log_fn, f"extracting {len(zips)} zips from '{src_path}' to '{dst_path}' using {num_workers} workers")
            unzip_imagefolder_classwise(src=src_path, dst=dst_path, num_workers=num_workers)
        else:
            # copy folders which contain the raw files (not zipped or anything)
            log(log_fn, f"copying folders of '{src_path}' to '{dst_path}'")
            # copy folder (dirs_exist_ok=True because dst_path is created for start_copy_file)
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
    elif src_path.with_suffix(".zip").exists():
        log(log_fn, f"extracting '{src_path.with_suffix('.zip')}' to '{dst_path}'")
        # extract zip
        was_zip = True
        with zipfile.ZipFile(src_path.with_suffix(".zip")) as f:
            f.extractall(dst_path)
    else:
        raise NotImplementedError

    # create end_copy_file
    with open(end_copy_file, "w") as f:
        f.write("this file indicates that the copying the dataset automatically was successful")

    log(log_fn, "finished copying data from global to local")
    return CopyImageFolderResult(
        was_copied=True,
        was_deleted=was_deleted,
        was_zip=was_zip,
        was_zip_classwise=was_zip_classwise,
    )


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
        jobs = [joblib.delayed(_unzip)(src, dst) for src, dst in jobargs]
        pool = joblib.Parallel(n_jobs=num_workers)
        pool(jobs)
