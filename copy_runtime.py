import os
import shutil
import site
from pathlib import Path


def copy_ryzenai_onnx_runtime(targets):
    installation_path = os.environ["RYZEN_AI_INSTALLATION_PATH"]
    # Copy onnxruntime bin files
    src_path = Path(installation_path, "onnxruntime", "bin")

    for target in targets:
        dest_dir = f"./target/{target}"
        print("Copy:", src_path, "->", dest_dir)
        shutil.copytree(src_path, dest_dir, dirs_exist_ok=True)

        # Copy vaip_config.json
        src_file = Path(installation_path, "voe-4.0-win_amd64",
                        "vaip_config.json")
        print("Copy:", src_file, "->", dest_dir)
        shutil.copy2(src_file, dest_dir)


def copy_python_runtime(targets):
    site_packages = site.getsitepackages()
    for path in site_packages:
        # if path.find("onnxruntime") != -1:
        onnx_lib_path = Path(path, "onnxruntime/capi")
        if onnx_lib_path.exists():
            for target in targets:
                dest_dir = f"./target/{target}"
                for fname in os.listdir(onnx_lib_path):
                    if fname.endswith(".dll"):
                        fpath = Path(onnx_lib_path, fname)
                        shutil.copy2(fpath, dest_dir)
                        print("Copy:", fpath, "->", dest_dir)
            return True


def main():
    targets = ["debug", "release"]
    if copy_python_runtime(targets):
        return
    copy_ryzenai_onnx_runtime(targets)


if __name__ == "__main__":
    main()
