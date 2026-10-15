# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Dependency metadata for both PEP 517/editable and legacy setuptools installs.
import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()


def read_requirements(filename):
    """Share the release pins with pip instead of keeping conflicting lists."""
    lines = (Path(__file__).parent / filename).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


install_requires = read_requirements("requirements.txt")

# These extras describe the study's vLLM profile, not inherited, untested
# SGLang/Megatron/TRL environments. Install CUDA extensions after Torch.
extras_require = {
    "test": ["pytest", "pre-commit", "py-spy==0.4.2"],
    "gpu": read_requirements("requirements-kernels.txt"),
    "vllm": ["vllm==0.19.1"],
    "retrieval": ["faiss-gpu==1.15.0"],
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    # Distribution name; Python import paths such as `verl` stay unchanged.
    name="self-distillation-agentrl",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    license="Apache 2.0",
    author="Self-Distillation-AgentRL contributors",
    description="Self-Distillation-AgentRL: Performance, Stability, and Cost of Agentic Reinforcement Learning",
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.10,<3.11",
    package_data={
        "": ["version/*"],
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
