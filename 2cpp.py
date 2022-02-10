# rgb input
import cv2
import sys
import numpy as np
import torch
def preprocess(x):
    input_range = [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std
    return x
def preprocess_x(x):
    img = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    img = preprocess(img)
    img = np.rollaxis(img, 2, 0)
    img = img.astype(np.float32)
    x_tensor = torch.from_numpy(img).to('cuda').unsqueeze(0)

    return x_tensor
def main(argv):
    w,h= 768,512
    model = torch.load('init.pt','cuda')
    model.encoder.set_swish(memory_efficient=False)

    model.eval()

    origin = cv2.imread('deeplab1.png')
    origin = cv2.resize(origin, (w, h))

    example=preprocess_x(origin)
    traced_script_module = torch.jit.trace(model.to('cuda'), example,strict=False)

    traced_script_module.save("init.smart")
    return 0
if __name__ == "__main__":
    main(sys.argv)