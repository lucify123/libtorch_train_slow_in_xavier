import time
import torch
import cv2
import numpy as np

DEVICE = "cuda"
w, h = 768, 512


def preprocess(img):
    global w, h
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (w, h))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = x / 255.0
    mean = np.array(mean)
    x = x - mean
    std = np.array(std)
    x = x / std

    x = np.rollaxis(x, 2, 0)
    x = x.astype(np.float32)
    x_tensor = torch.from_numpy(x).to(DEVICE).unsqueeze(0)

    return x_tensor


x = preprocess(cv2.imread("4x.jpg"))
# forward


def infer_origin():
    model.eval()
    r = model(x)
    r = r.to('cpu').detach()


def train_origin():
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.1**3),
    ])

    optimizer.zero_grad()
    r = model(x)
    y = r - 1e-2
    loss = torch.nn.BCEWithLogitsLoss()(y, r)
    loss.backward()
    optimizer.step()


def infer_jit():
    model.eval()
    r = model(x)
    r = r.to('cpu').detach()


def train_jit():
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.1**3),
    ])
    optimizer.zero_grad()
    r = model(x)
    y = r - 1e-2
    loss = torch.nn.BCEWithLogitsLoss()(y, r)
    loss.backward()
    optimizer.step()


if __name__ == "__main__":
    if 1:
        model = torch.load('init.pt', 'cuda')
        model.eval()
        print('warming up,pls wait...')

        for _ in range(10):
            infer_origin()
            train_origin()

        start = time.time()
        for _ in range(50):
            infer_origin()
        it = time.time()
        print('infer cost', it - start)
        for _ in range(50):
            train_origin()
        end = time.time()
        print('train cost', end-it)
    else:
        model = torch.jit.load('init.smart', 'cuda')
        model.eval()

        print('warming up,pls wait...')
        for _ in range(10):
            infer_jit()
            train_jit()

        start = time.time()
        for _ in range(50):
            infer_jit()
        it = time.time()
        print('infer cost', it - start)
        for _ in range(50):
            train_jit()
        end = time.time()
        print('train cost', end-it)
