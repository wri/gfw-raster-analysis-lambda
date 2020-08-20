from setuptools import setup

setup(
    name="gfw_raster-analysis-lambda",
    version="1.0.0",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["raster_analysis", "raster_analysis.layer", "lambdas"],
    author="Justin Terry/Thomas Maschler",
    license="MIT",
    install_requires=["aws-xray-sdk", "requests"],
)
