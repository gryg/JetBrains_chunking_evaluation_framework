from setuptools import setup, find_packages

setup(
    name="chunking-eval",
    version="0.1.7",
    description="A modular framework for evaluating text chunking strategies",
    author="Filoftei-Andrei Grigore",
    author_email="filoftei.grigore@protonmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "sentence-transformers": ["sentence-transformers>=2.2.0"],
        "huggingface": ["transformers>=4.20.0", "torch>=1.10.0"],
        "tokenizers": ["tiktoken>=0.3.0"],
        "sentence-chunker": ["spacy>=3.4.0"],
        "all": [
            "sentence-transformers>=2.2.0",
            "transformers>=4.20.0",
            "torch>=1.10.0",
            "tiktoken>=0.3.0",
            "spacy>=3.4.0",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)