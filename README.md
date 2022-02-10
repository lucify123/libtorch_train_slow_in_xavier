# libtorch_train_slow_in_xavier

# environment
- Jetson Xavier NX
- Test on pytorch 1.7 1.8 1.9；opencv 4.1.1
- 15W 2CORE

# results
||  pytorch_origin   | pytoch_jit  | libtorch_jit |
|----|  ----  | ----  |----|
|infer 50x| 8  | 7.76 |7.8|
|train 50x| 30.02  | 27.2 |**41.62**|
			
We found: 
Libtorch’s inferring speed is a little bit slower than Pytorch’s Jit.
Libtorch’s training speed is **53% slower** than Pytorch’s Jit.

# how to run
- please install opencv and libtorch manually
- cd libtorch_train_slow_in_xaiver
- python3 pytorch_origin.py
- python3 pytorch_jit.py
- cd build
- find / -name "TorchConfig.cmake" # get the TorchConfig.cmake DIR
- export Torch_DIR=DIR_above
- cmake ..
- make
- ./main
