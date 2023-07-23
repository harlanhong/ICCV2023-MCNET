# OpenFace for Pytorch

* Disclaimer: This codes require the input face-images that are aligned and cropped in the same way of the original OpenFace. *


I made a dirty code to use OpenFace in PyTorch.
I converted '<a href="https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7">nn4.small2.v1.t7</a>' to a .hdf5 file using '<a href="https://github.com/deepmind/torch-hdf5">torch-hdf5</a>'.
Then I read layer informations from .hdf5 file, which can be displayed as follows:
```
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> (13) -> (14) -> (15) -> (16) -> (17) -> (18) -> (19) -> (20) -> (21) -> (22) -> (23) -> (24) -> (25) -> (26) -> output]
  (1): nn.SpatialConvolutionMM(3 -> 64, 7x7, 2,2, 3,3)
  (2): nn.SpatialBatchNormalization (4D) (64)
  (3): nn.ReLU
  (4): nn.SpatialMaxPooling(3x3, 2,2, 1,1)
  (5): nn.SpatialCrossMapLRN
...
```
Then I manually coded layers in PyTorch (see loadOpenFace.py) with some tentative layers code which may be supported by PyTorch officially laters (SpatialCrossMapLRN_temp.py, adopted from <a href="https://github.com/pytorch/pytorch/blob/master/torch/legacy/nn/SpatialCrossMapLRN.py">PyTorch's nn.legacy</a>).
The final model is 'openface.pth' (which may need to be renamed to 'openface_nn4_small2_v1.pth'), which can be loaded by codes in loadOpenFace.py.

Please see main section of loadOpenFace.py for how-to-use.
Simply,
```
net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()
feature = net(input_tensor)    # input_tensor should be (batch_size, 3, 96, 96)
```


* License <BR>
This is released under Apache 2.0 license.
