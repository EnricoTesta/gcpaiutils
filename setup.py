import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gcpaiutils",
    version="0.0.1",
    author="Enrico Testa",
    author_email="enrico@mailtesta.com",
    description="GCP AI Platforms helper classes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EnricoTesta/gcpaiutils",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
