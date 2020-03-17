import setuptools

setuptools.setup(
    name='hbd',
    version='1.6.1',
    author='Daniel Roberto Cassar',
    author_email='daniel.r.cassar@gmail.com',
    description='hbd',
    url="https://github.com/drcassar/hbd",
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.1', 'pandas>=0.24.0', 'deap', 'tensorflow'],
    python_requires='>=3.6',
)
