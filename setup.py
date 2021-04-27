import setuptools

import os
import stat
from setuptools.command.install import install

from distutils import log

with open("README.md", "r") as fh:
    long_description = fh.read()

"""
The below code is taken from https://github.com/Uberi/speech_recognition
See README.md for licence information
"""
FILES_TO_MARK_EXECUTABLE = ["flac-linux-x86", "flac-linux-x86_64", "flac-mac", "flac-win32.exe"]


class InstallWithExtraSteps(install):
    def run(self):
        install.run(self)  # do the original install steps

        # mark the FLAC executables as executable by all users (this fixes occasional issues when file permissions get messed up)
        for output_path in self.get_outputs():
            if os.path.basename(output_path) in FILES_TO_MARK_EXECUTABLE:
                log.info("setting executable permissions on {}".format(output_path))
                stat_info = os.stat(output_path)
                os.chmod(
                    output_path,
                    stat_info.st_mode |
                    stat.S_IRUSR | stat.S_IXUSR |  # owner can read/execute
                    stat.S_IRGRP | stat.S_IXGRP |  # group can read/execute
                    stat.S_IROTH | stat.S_IXOTH  # everyone else can read/execute
                )


"""
Below is DanSpeech licence
"""

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setuptools.setup(
    name="danspeech",
    version="1.0.4",
    author="Rasmus Arpe Fogh Jensen, Martin Carsten Nielsen",
    author_email="rasmus.arpe@gmail.com, mcnielsen4270@gmail.com,",
    description="Speech recognition for Danish",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danspeech/danspeech",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license_file="LICENCE.txt",
    classifiers=[
        "Programming Language :: Python :: 3",
        'Development Status :: 5 - Production/Stable',
        "Operating System :: OS Independent",
    ],
)
