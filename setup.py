from setuptools import setup, find_packages

setup(
    name="traffic_sign_recognition",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "PyYAML>=5.4.0",
        "opendatasets>=0.1.22",
    ],
    python_requires=">=3.7",
)