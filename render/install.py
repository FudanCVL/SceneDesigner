from pathlib import Path
import sys
import platform
import urllib.request
import shutil
import tarfile
import zipfile
import subprocess
from tqdm import tqdm
def download_file(url, dest):
    dest_path = Path(dest)
    
    with urllib.request.urlopen(url) as response:
        remote_size = int(response.headers.get('content-length', 0))
    
    if dest_path.exists():
        local_size = dest_path.stat().st_size
        if local_size == remote_size:
            print(f"{dest_path.name} already exists, skipping download.")
            return
        else:
            print(f"{dest_path.name} exists but incomplete (local: {local_size}, remote: {remote_size}), re-downloading...")
            dest_path.unlink()
    
    with urllib.request.urlopen(url) as response:
        with open(dest, 'wb') as out_file, tqdm(
            desc=f"download {dest_path.name}",
            total=remote_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))

def extract_file(archive_path, extract_to):
    print("Extracting", archive_path.name)
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        with tarfile.open(archive_path, 'r:xz') as tar_ref:
            tar_ref.extractall(extract_to)

if __name__ == "__main__":
    system = platform.system().lower()
    current_dir = Path(__file__).parent

    if system == "windows":
        archive_name = "blender-4.2.8-windows-x64.zip"
        blender_path = current_dir / "blender-4.2.8-windows-x64"
    elif system == "linux":
        archive_name = "blender-4.2.8-linux-x64.tar.xz"
        blender_path = current_dir / "blender-4.2.8-linux-x64"
    else:
        print("not support system:", system)
        sys.exit(1)

    url = "https://download.blender.org/release/Blender4.2/" + archive_name
    archive_file = current_dir / archive_name

    download_file(url, archive_file)
    

    extract_file(archive_file, current_dir)
    blender_path.rename("blender")
    archive_file.unlink()

    python_path = Path("blender/4.2/python/bin").glob("python*")
    python_path = next(python_path)
    subprocess.run([str(python_path), "-m", "pip", "install", "-r", "blender_requirements.txt"])

