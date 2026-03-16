from setuptools import setup, find_packages

setup(
    name="chainlink-memory",
    version="0.1.0",
    description="Find connections your vector search misses. Chain-aware memory for AI agents.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Akeel Mohammed",
    author_email="akeel.m96@gmail.com",
    url="https://github.com/akeelm96/chainlink",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "anthropic>=0.40.0",
        "sentence-transformers>=2.2.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "server": [
            "fastapi==0.115.0",
            "uvicorn==0.30.0",
            "pydantic>=2.0.0",
        ],
        "mcp": [
            "mcp>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chainlink-mcp=chainlink.mcp_server:cli_entry",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="ai memory agent chain reasoning vector search rag llm mcp",
)
