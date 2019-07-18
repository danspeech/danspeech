import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="danspeech",
    version="0.0.1",
    author="Rasmus Arpe Fogh Jensen, Martin Carsten Nielsen",
    author_email="rasmus.arpe@gmail.com, mcnielsen4270@gmail.com,",
    description="Speech recognition for Danish",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rasmusafj/danspeech",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        'Development Status :: 3 - Alpha',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)