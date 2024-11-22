from setuptools import setup, find_packages

setup(
    name="chitu",
    version="0.2",
    author="Jiarui Fang, SageAttention Team, Int8_flash_attn Team",
    author_email="fangjiarui@gmail.com",
    packages=find_packages(),
    description="8-bit plug-and-play attention suite.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    license="MIT",
    python_requires=">=3.9",
    install_requires=[
        "sageattn==1.0.6",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
)
