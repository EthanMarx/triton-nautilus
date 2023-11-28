from .resnet import Resnet
import torch

def main():
    nn = ResNet(num_ifos)
    torch.jit.save("resnet.pt")

if __name__ == "__main__":
    main()

