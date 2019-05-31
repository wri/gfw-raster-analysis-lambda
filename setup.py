from setuptools import setup

setup(
    name="gfw_raster-analysis-lambda",
    version="0.1.0",
    description="Lambda function to run serverless on the fly raster analysis",
    packages=["raster_analysis"],
    author="Thomas Maschler",
    license="MIT",
    install_requires=["rasterio[s3]==1.0.23",
                    "shapely==1.6.4.post2",
                    "pyproj==2.1.3",
                    "nose==1.3.7",
                    "Flask==1.0.3",
                    "Pandas==0.24.2"]
)
