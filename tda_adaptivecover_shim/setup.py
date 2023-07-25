import setuptools

setuptools.setup(
    name='adaptivecover_kmapper',
    version='0.1.0',    
    description='A small shim to use adaptive cover with Kepler Mapper',
    author='Hazel Brenner',
    author_email='hnb29@cornell.edu',
    license='BSD 2-clause',
    packages=setuptools.find_packages(),
    install_requires=['mapper_xmean_cover',
                      'numpy',                     
                      ],
)