from setuptools import setup

setup(
    name="gfw_raster-analysis-lambda",
    version="0.1.0",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["raster_analysis"],
    author="Thomas Maschler",
    license="MIT",
    install_requires=[
        "rasterio[s3]==1.0.23",
        "shapely==1.6.4.post2",
        "pyproj==2.1.3",
        # "Pandas==0.24.2",
        "requests==2.20.1",
        "urllib3==1.24.3",
        "awscli==1.16.169",
        "aws-sam-cli==0.16.1",
        "click==6.7"]

)
