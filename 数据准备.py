# download_cifar10.py
import urllib.request
import tarfile
import os

def download_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    
    if not os.path.exists(filename):
        print("下载CIFAR-10数据集...")
        urllib.request.urlretrieve(url, filename)
    
    # 解压文件
    print("解压文件...")
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    
    print("数据准备完成!")

if __name__ == "__main__":
    download_cifar10()