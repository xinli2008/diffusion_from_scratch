import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from config import *

# PIL图像转tensor
pil_to_tensor = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    ### transformer.totensor():
    # 1、将图像转化为tensor, 将数据从0-255缩放到[0-1]之间。
    # 2、将PIL或者numpy图像从(h,w,c)变化为(c,h,w)。
    transforms.ToTensor()
])

# tensor转化为PIL
tensor_to_pil = transforms.Compose([
    transforms.Lambda(lambda t: t*255),
    transforms.Lambda(lambda t: t.type(torch.uint8)),
    transforms.ToPILImage(),
])

train_dataset = torchvision.datasets.MNIST(root = "./", train = True, download = True, transform = pil_to_tensor)

if __name__ == "__main__":
    image_tensor, label = train_dataet[0]

    plt.figure(figsize=(5,5))
    pil_image = tensor_to_pil(image_tensor)
    plt.imshow(pil_image)
    plt.show()