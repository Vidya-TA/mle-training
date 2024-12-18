from setuptools import setup, find_packages

setup(
    name="housing price prediction",
    version="0.1.0",
    packages=find_packages(where="src"), 
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "six",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "main=notebooks.main:main",  
        ],
    },
)
