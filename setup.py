from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="SemantiCache a RAG Assitant",
    description="A Retrieval-Augmented Generation (RAG) assistant that uses semantic caching to improve response times and cost optimization",
    version="0.1",
    author="Renswick Delvar",
    author_email="renswick.delver@gmail.com",
    packages=find_packages(include=['src', 'src.*']),
    install_requires = requirements,
)