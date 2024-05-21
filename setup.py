from setuptools import setup

setup(
    name="gfw_raster-analysis-lambda",
    version="2.0.0",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["raster_analysis"],
    author="Justin Terry/Thomas Maschler",
    license="MIT",
    install_requires=[
        "aws-xray-sdk==2.12.0",
        "requests==2.32.0",
        "geobuf==1.1.1",
        "protobuf==3.20.3",
        "pydantic==1.10.12",
        "mo_sql_parsing==9.436.23241",
    ],
)
