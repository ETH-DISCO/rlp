from setuptools import Extension, setup

setup(
    name="rlp",
    description="Package containing a Gymnasium-compatible version of Simon Tatham's Portable Puzzle Collection.",
    version="1.0.0",
    install_requires=["gymnasium>=0.28.0", "pygame>=2.1.0"],
    packages=["rlp", "rlp.envs", "rlp.constants"],
    package_dir={
        "rlp": "rlp",
        "rlp.puzzle": "rlp",
        "rlp.api": "rlp",
        "rlp.specific_api": "rlp",
        "rlp.envs": "rlp/envs",
    },
    package_data={"rlp": ["lib/lib*.so", "lib/icons/*-96d24.png"]},
    include_package_data=True,
    ext_modules=[Extension("rlp.constants", ["rlp/constants/puzzle_constants.c"])],
)
