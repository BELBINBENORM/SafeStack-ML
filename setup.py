from setuptools import setup, find_packages

setup(
    name="SafeStack-ML",
    version="0.1.0",
    description="A robust Stacking Classifier with artifact persistence and memory safety.",
    author="Belbin Beno R M",
    author_email="belbin.datascientist@gmail.com",
    packages=find_packages(),
    py_modules=["safe_stack"],
    install_requires=[
        "numpy",
        "scikit-learn",
        "joblib",
        "cloudpickle",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    license="Apache-2.0",
)
