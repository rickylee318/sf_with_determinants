import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='4SFwD', # 
    version="0.0.2",
    author='Ruei-Chi Lee',
    author_email='axu3bjo4fu6@gmail.com',
    description='four component stochastic frontier model with determinants',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://https://github.com/rickylee318/sf_with_determinants',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)