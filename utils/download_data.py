from roboflow import Roboflow

# You still need your own API key to use the library
rf = Roboflow(api_key="aJbb6tEq1kmfwLrFpyeR")

project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
version = project.version(4)

# Download in "yolov8" format to a folder named "license-plate-dataset"
version.download(
    model_format="yolov8",
    location="license-plate-dataset"
)

print("Download complete!")