from setuptools import setup, find_packages

setup(
    name="langchain_bytez",  # Name of your package
    version="0.0.0",
    packages=find_packages(),
    description="Bytez langchain integration",
    long_description=open("./README.md").read(),
    long_description_content_type="text/markdown",
    author="Bytez",
    url="https://github.com/bytez-com/docs",
    install_requires=["langchain==0.3.17"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
)
