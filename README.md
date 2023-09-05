# FRS-OpenCv

This project has various files which are used to regconize face and facail analysis using deepface and other libraries.
The entire project is made as docker image.

How to Run docker image
--
Pull the docker image
```
docker pull pavan0077/deepface:latest
```
Run the image (Needs webcam)
```
sudo docker run --device=/dev/video0 --rm -it -P --user $(id root -u):$(id root -g) -p 5901:5901 -p 6080:6080 pavan0077/deepface:latest
```

Enter this commands to export display
```
ps aux | grep Xorg
export DISPLAY=:1
export XAUTHORITY=/path/to/your/xauthority/file
```
To Start Vnc server
```
start_vnc
```
It show url in the docker container after running above command click the link and enter password ``` cms.cern ```

Incase broswer not opened automatically then use the below command and 
open in web broswer and type ``` http://127.0.0.1:6080/vnc.html ``` and password ``` cms.cern ```

If you want to use vnc viewer or any other application to view like Remmina and others
```VNC viewer address: 127.0.0.1::5901 ``` and password ``` cms.cern ```

How to Run python code for face-Reg
--
Right click on empty space and go to Applications --- Shells -- Bash


![image](https://github.com/pavankumar0077/FRS-OpenCv/assets/40380941/9c0c6874-da88-46c5-b503-18a49ef86d76)

Open project folder
--
```
ls
```
```
cd FRS-OpenCv
```

Run deepface.py file
--
```
python3 my_deepface.py
```

If you want to run any other python script
then use 

```
python3 <file-name.py>
```
