#!/usr/bin/env python3

import os
import platform
import subprocess
import sys
from pathlib import Path

COMFYVOXELS_ROOT_ABS_PATH = os.path.dirname(__file__)
OPENVDB_REPO_URL = "https://github.com/AcademySoftwareFoundation/openvdb.git"
OPENVDB_SOURCE_DIR = os.path.join(COMFYVOXELS_ROOT_ABS_PATH, "openvdb")
OPENVDB_BUILD_DIR = os.path.join(OPENVDB_SOURCE_DIR, "build")

def run_command(cmd, shell=True, cwd=None):
    """
    Helper function to run a command in a subprocess, print output, and raise
    if there's an error.
    """
    print(f"Running command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        result = subprocess.run(cmd, shell=shell, check=True, cwd=cwd)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with return code {e.returncode}")
        sys.exit(e.returncode)


def install_dependencies():
    """
    Install the dependencies needed for OpenVDB + NanoVDB
    on macOS (via Homebrew), Linux (via apt-get),
    and Windows (via vcpkg).
    """
    current_os = platform.system().lower()
    print(f"Detected OS: {current_os}")

    if current_os.startswith("linux"):
        # Ubuntu/Debian example using apt-get. 
        # Adjust if you have a different package manager (e.g., yum or dnf).
        deps = [
            "libboost-iostreams-dev",
            "libtbb-dev",
            "libblosc-dev",
            # Add any extras for building. 
            # For the core OpenVDB, these are usually enough.
        ]
        cmd = f"sudo apt-get update && sudo apt-get install -y {' '.join(deps)}"
        run_command(cmd)

    elif current_os == "darwin":  # macOS
        # Using Homebrew
        deps = ["boost", "tbb", "c-blosc"]
        for dep in deps:
            cmd = f"brew install {dep}"
            run_command(cmd)

    elif current_os.startswith("win"):
        print("Ensure vcpkg is installed and in your PATH.")
        # The list from OpenVDB documentation
        deps = [
            "zlib:x64-windows",
            "blosc:x64-windows",
            "tbb:x64-windows",
            "boost-iostreams:x64-windows",
            "boost-any:x64-windows",
            "boost-algorithm:x64-windows",
            "boost-interprocess:x64-windows",
        ]
        for dep in deps:
            cmd = f"vcpkg install {dep}"
            run_command(cmd)
    else:
        print(
            "Your OS is not recognized by this script. "
            "Please install build dependencies manually."
        )


def clone_openvdb():
    """
    Clones the OpenVDB repository if it doesn't exist already.
    """
    if not os.path.exists(OPENVDB_SOURCE_DIR):
        cmd = f"git clone {OPENVDB_REPO_URL} {OPENVDB_SOURCE_DIR}"
        run_command(cmd)
    else:
        print("OpenVDB repository already exists, pulling latest changes...")
        run_command("git pull", cwd=OPENVDB_SOURCE_DIR)


def build_openvdb():
    """
    Builds OpenVDB with NanoVDB support. 
    For Windows, uses the default generator if run from a Visual Studio shell
    or you can specify the -A x64 and vcpkg toolchain manually.
    For macOS/Linux, uses makefiles by default.
    """
    if not os.path.exists(OPENVDB_BUILD_DIR):
        os.mkdir(OPENVDB_BUILD_DIR)

    current_os = platform.system().lower()
    cmake_cmd = ["cmake", ".."]

    # Use some standard config flags to enable NanoVDB:
    #   -D OPENVDB_BUILD_NANOVDB=ON -> enable NanoVDB
    #   -D NANOVDB_USE_OPENVDB=ON   -> link NanoVDB with OpenVDB
    #   -D OPENVDB_BUILD_AX=OFF     -> example if you do NOT need AX
    build_flags = [
        "-D", "OPENVDB_BUILD_NANOVDB=ON",
        "-D", "NANOVDB_USE_OPENVDB=ON",
        # Add any other flags you might need:
        # e.g. "-D", "OPENVDB_BUILD_AX=ON", 
        # or setting your install prefix with "-DCMAKE_INSTALL_PREFIX=..."
    ]
    cmake_cmd.extend(build_flags)

    # On Windows, if using vcpkg and Visual Studio
    if current_os.startswith("win"):
        # Adjust vcpkg path as needed. For example:
        # vcpkg_path = r"C:\path\to\vcpkg\scripts\buildsystems\vcpkg.cmake"
        # cmake_cmd.append(f"-DCMAKE_TOOLCHAIN_FILE={vcpkg_path}")
        
        # Example for x64:
        # cmake_cmd.append("-DVCPKG_TARGET_TRIPLET=x64-windows")
        # cmake_cmd.append("-A x64")
        pass

    print(f"CMake configure command: {' '.join(cmake_cmd)}")

    # Configure
    run_command(cmake_cmd, shell=False, cwd=OPENVDB_BUILD_DIR)
    
    # Build + install
    # On Windows, we might want to specify `--config Release`
    # On Linux/Mac, `make -j` is typical
    if current_os.startswith("win"):
        # Parallel build, release config
        run_command(["cmake", "--build", ".", "--parallel", "4", "--config", "Release", "--target", "install"],
                    shell=False, cwd=OPENVDB_BUILD_DIR)
    else:
        run_command(["make", "-j4"], shell=False, cwd=OPENVDB_BUILD_DIR)
        run_command(["make", "install"], shell=False, cwd=OPENVDB_BUILD_DIR)

    print("OpenVDB (with NanoVDB) build and installation complete!")


def main():
    print("==== ComfyUI Voxel: Installing OpenVDB & NanoVDB ====")
    install_dependencies()
    clone_openvdb()
    build_openvdb()
    print("==== Install script completed successfully ====")


if __name__ == "__main__":
    main()
