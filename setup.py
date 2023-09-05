from setuptools import setup

setup(
    name="gfw_raster-analysis-lambda",
    version="2.0.0",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["raster_analysis"],
    author="Justin Terry/Thomas Maschler",
    license="MIT",
    install_requires=[
        "aws-xray-sdk~=2.8.0",
        "requests~=2.25.1",
        "geobuf==1.1.1",
        "protobuf==3.20.0",
        "pydantic~=1.7.3",
        "moz_sql_parser~=4.40.21126",
    ],
)
