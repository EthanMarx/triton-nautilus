from triton_nautilus.export.resnet import ResNet
import torch
import s3fs


# load dummy model, 
# convert to torch script, 
# and send to s3 bucket on nautilus
def main(
    sample_rate = 2048,
    kernel_length = 16,
    batch_size = 512,
    num_ifos = 2,
    bucket = "triton-nautilus"
):

    kernel_size = sample_rate * kernel_length
    sample_input = torch.rand(batch_size, num_ifos, kernel_size)
    nn = ResNet(num_ifos=2, layers=[3, 4, 6, 3], norm_groups = 16)
    m = torch.jit.trace(nn, sample_input)
    
    s3 = s3fs.S3FileSystem(endpoint_url="https://s3-west.nrp-nautilus.io")
    
    s3.makedirs(f"s3://{bucket}/", exist_ok=True)
    with s3.open(f"s3://{bucket}/model.pt", "wb") as f:
        torch.jit.save(m , f)

if __name__ == "__main__":
    main()

