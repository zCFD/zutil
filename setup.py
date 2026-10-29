#!/usr/bin/env python
import os
import setuptools

version = os.environ.get("RELEASE_VERSION", "0.0.0")

if __name__ == "__main__":
    setuptools.setup(
        version=version,
        install_requires=[
            "pydantic>=2.12,<3",
            "colorama",
            "numpy",
            "voluptuous",
            "click>=8.0.0",
            "scipy",
            "pandas",
            "networkx",
            "PyYAML",
            "pytransform3d",
            "trimesh",
        ],
        entry_points={
            "console_scripts": [
                "validate_input=zutil.validate.cli:main",
                "update_input=zutil.update_input.cli:main",
                "generate_paraview_helpers=zutil.paraview_state:main",
            ],
        },
    )
