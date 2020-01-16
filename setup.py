from setuptools import setup

setup(
    name="gfw_raster-analysis-lambda",
    version="0.2.0",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["raster_analysis"],
    author="Thomas Maschler",
    license="MIT",
    install_requires=["lambda-decorators~=0.3.0", "aws-xray-sdk", "requests"],
)
