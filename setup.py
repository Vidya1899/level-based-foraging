from setuptools import setup, find_packages

setup(
    name="level-based-foraging",
    version="1.0.15",
    description="Level Based Foraging Environment",
    author="",
    url="https://github.com/Vidya1899/level-based-foraging",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numpy", "gym>=0.12", "pyglet"],
    extras_require={"test": ["pytest"]},
    include_package_data=True,
)
