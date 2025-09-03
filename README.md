# Python based framework for indoor flight of Crazyflie using OptiTrack
This repository includes the Python codes needed to achieve controlled and stabilised flight of a Crazyflie drone in an indoor flight arena, using OptiTrack as a method for external localisation in GPS-denied environments.

To start with, NatNet SDK from OptiTrack website (https://optitrack.com/software/natnet-sdk/) will need to be downloaded onto the host PC.
Once extracted, the folderf "Samples -> PythonClient" will have 3 python codes named "DataDescriptions.py", "MoCapData.py", "NatNetClient.py" (also included in the repository). These python codes must be in the same folder as the main control code.

Practical setup:
1. Place 4 reflective markers on the drone in an asymmetric pattern to track the drone using the OptiTrack cameras
2. Place reflective markers/tape at the desired positions around the capture volume
Once the code is run, it will automatically prompt the user to choose the markers and the desired manouever.

(To be updated)








