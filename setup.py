from setuptools import setup

setup(
    name="gfw_raster-analysis-lambda",
    version="1.0.0",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["raster_analysis", "raster_analysis.layer"],
    author="Thomas Maschler",
    license="MIT",
    install_requires=["lambda-decorators~=0.3.0", "aws-xray-sdk", "requests"],
)
