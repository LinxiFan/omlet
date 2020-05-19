import os
from setuptools import setup, find_packages


def read_file(fname):
    this_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(this_dir, fname)) as f:
        return f.read()


pkg_name = 'omlet'


setup(
    name=pkg_name,
    version='0.0.1a',
    author='Lab',
    # url='http://github.com/',
    description='research project',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    keywords=['Deep Learning',
              'Machine Learning'],
    license='MIT',
    packages=[
        package for package in find_packages() if package.startswith(pkg_name)
    ],
    entry_points={
        'console_scripts': [
            # 'cmd_tool=mylib.subpkg.module:main',
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=False
)
