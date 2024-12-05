import subprocess
import os

if __name__ == "__main__":
    # sudo docker run --gpus=all -it -v /home/aerotract/software/myria3d_pipeline:/app/myria3d_pipeline -v /home/aerotract/Desktop/LiDAR_Myria3D_Data_Dir/prediction_output/runs:/home/aerotract/Desktop/LiDAR_Myria3D_Data_Dir/prediction_output/runs -v /home/aerotract/Desktop/LiDAR_Myria3D_Data_Dir/HDF5/:/home/aerotract/Desktop/LiDAR_Myria3D_Data_Dir/HDF5/ --shm-size=2g myria3d
    command = ["sudo", "docker", "run", "--gpus", "all", "-it",
            "-v","/home/aerotract/software/PointTransformerV3/:/app/PointTransformerV3" , 
            "-v","/home/aerotract/software/AeroTract_LiDAR_Pipeline/:/app/AeroTract_LiDAR_Pipeline" , 
            "--shm-size=2g", "myria3d_seth:latest"]
    subprocess.run(command)