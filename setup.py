from setuptools import setup

setup(
    name="gfw_raster-analysis-lambda",
    version="2.0.0",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["raster_analysis"],
    author="Justin Terry/Thomas Maschler",
    license="MIT",
    install_requires=["aws-xray-sdk", "requests", "geobuf"],
)
