from pathlib import Path
from setuptools import find_packages, setup


def load_reqs(path):
    reqs = []
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith("-r"):
                reqs += load_reqs(line.split(" ")[1].strip())
            elif line.startswith("-e") or line.startswith("git"):
                continue
            else:
                req = line.strip()
                if req and not req.startswith("#"):
                    reqs.append(req)
    return reqs


project_path = Path(__file__).parent
req_path = project_path / "requirements.txt"
requirements = load_reqs(req_path)

setup(
    name="zamba",
    packages=find_packages(),
    version="1.0",
    description="Zamba identifies species in camera trap videos.",
    author="DrivenData",
    license="",
    install_requires=requirements,
)
