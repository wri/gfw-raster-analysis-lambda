from setuptools import setup

setup(
    name="gfw_raster-analysis-lambda",
    version="0.1.0",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["raster_analysis"],
    author="Thomas Maschler",
    license="MIT",
    install_requires=[
        "rasterio[s3]~=1.0.28",
        "shapely~=1.6.4.post2",
        "pyproj~=2.1.3",
        # "requests~=2.20.1",
        # "urllib3~=1.24.3",
        # "awscli~=1.16.169",
        # "aws-sam-cli~=0.16.1",
        # "click~=7.0",
        # "botocore<1.13.0,>=1.12.164",
        "pandas~=0.25.1",
    ],
)
