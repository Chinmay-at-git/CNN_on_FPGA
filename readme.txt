-------------------------------------------------------
Convolutional neural network on FPGA

There are two folders, cnn_parallel and cnn_pipe
These are two approches we tried in this project.
How to run?
./cnnparallel rgbim.raw out.raw 732 486 3
./cnnpipe rgbim.raw out.raw 732 486 3
Emulate:
./run.sh rgbim.raw out.raw 732 486 3
Note that rgbim.raw should be 732*486*3*sizeof(float) size file.
We have provided it in OpenCL folder. 
Thank you!
