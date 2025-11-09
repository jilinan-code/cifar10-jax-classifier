import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
jax.config.update('jax_platform_name', 'cpu')
class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        # 直接从原始CIFAR-10二进制文件加载，无需torchvision
        self.data, self.labels = self._load_cifar10(train)    
    def _load_cifar10(self, train):
        """直接从CIFAR-10二进制文件加载数据"""
        data = []
        labels = []        
        if train:
            files = [f'data_batch_{i}' for i in range(1, 6)]
        else:
            files = ['test_batch']            
        for file in files:
            with open(f'cifar-10-batches-py/{file}', 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                data.append(batch['data'])
                labels.extend(batch['labels'])   
        # 合并并重塑数据
        data = np.vstack(data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # NHWC格式
        data = data.astype(np.float32) / 255.0
        data = (data - 0.5) / 0.5  # 标准化到[-1,1]        
        return data, np.array(labels)    
    def __len__(self):
        return len(self.data)    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # 第一个卷积块
        x = nn.Conv(64, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))        
        # 第二个卷积块
        x = nn.Conv(128, (3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2), (2, 2))        
        # 全连接层
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x
def create_train_state(rng, lr=0.001):
    model = CNN()
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))['params']
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(lr)
    )
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        return optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss
@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    return jnp.mean(jnp.argmax(logits, -1) == batch['label'])
def train_model(epochs=30):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, lr=0.001)
    
    train_loader = DataLoader(CIFAR10Dataset(train=True), batch_size=128, shuffle=True)
    test_loader = DataLoader(CIFAR10Dataset(train=False), batch_size=128)
    
    for epoch in range(epochs):
        train_losses = []
        for images, labels in train_loader:
            batch = {'image': jnp.array(images.numpy()), 'label': jnp.array(labels.numpy())}
            state, loss = train_step(state, batch)
            train_losses.append(loss)      
        accuracies = []
        for images, labels in test_loader:
            batch = {'image': jnp.array(images.numpy()), 'label': jnp.array(labels.numpy())}
            accuracies.append(eval_step(state, batch))       
        accuracy = np.mean(accuracies)
        avg_loss = np.mean(train_losses)
        print(f"Epoch {epoch+1}: 损失 {avg_loss:.4f}, 准确率 {accuracy*100:.2f}%")
        if accuracy > 0.70:
            print(f"达到目标! 准确率: {accuracy*100:.2f}%")
            break    
    return state, accuracy
if __name__ == "__main__":
    state, acc = train_model()
    print(f"最终准确率: {acc*100:.2f}%")