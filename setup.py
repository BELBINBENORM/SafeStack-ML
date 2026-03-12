from setuptools import setup

setup(
    name="safe-stack-ml",
    version="0.1.1",  
    author="Belbin Beno R M",
    author_email="belbin.datascientist@gmail.com",
    description="A memory-efficient stacking classifier with automated checkpointing.",
    py_modules=["safe_stack"],
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn>=1.4.0",  
        "joblib",
        "cloudpickle",
        "xgboost"
    ],
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.9',
)
