import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="objdetect",
    version="0.0.1",
    author="Romain Desarzens",
    author_email="rdesarz@protonmail.com",
    description="A small package for simple object detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rdesarz/multi_object_tracking",
    packages=setuptools.find_packages(),
    scripts=['bin/webcam_object_detection.py'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
