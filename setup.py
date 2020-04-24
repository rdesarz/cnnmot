import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cnnmot",
    version="0.0.1",
    author="Romain Desarzens",
    author_email="rdesarz@protonmail.com",
    description="A multiple object tracking library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rdesarz/cnnmot",
    packages=setuptools.find_packages(),
    scripts=['bin/webcam_object_detection.py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
