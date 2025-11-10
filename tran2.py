import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from flax import linen as nn
import pickle
jax.config.update('jax_platform_name', 'cpu')
class CIFAR10Dataset(Dataset):
    def __init__(self, train=True):
        files = [f'data_batch_{i}' for i in range(1,6)] if train else ['test_batch']
        data, labels = [], []
        for f in files:
            with open(f'cifar-10-batches-py/{f}', 'rb') as f:
                batch = pickle.load(f, encoding='latin1')
                data.append(batch['data']), labels.extend(batch['labels'])
        self.data = np.vstack(data).reshape(-1,3,32,32).transpose(0,2,3,1).astype(np.float32)/255.0
        self.data = (self.data - 0.5)/0.5  # 标准化到[-1,1]
        self.labels = np.array(labels)
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i], self.labels[i]

class PatchEmbedding(nn.Module):
    patch_size: int = 4       
    hidden_dim: int = 128    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size
        )(x)
        x = x.reshape(x.shape[0], -1, self.hidden_dim)  
        cls_token = self.param('cls_token', nn.initializers.zeros, (1, 1, self.hidden_dim))
        x = jnp.concatenate([jnp.tile(cls_token, (x.shape[0], 1, 1)), x], axis=1)
        pos_embed = self.param('pos_embed', nn.initializers.normal(0.02), 
                              (1, x.shape[1], self.hidden_dim))
        return x + pos_embed

class TransformerEncoder(nn.Module):
    num_heads: int = 8       
    mlp_dim: int = 256        
    dropout_rate: float = 0.1 

    @nn.compact
    def __call__(self, x, train=False):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(x)
        x = residual + x
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(residual.shape[-1])(x)
        return residual + x

class VisionTransformer(nn.Module):
    num_classes: int = 10     
    patch_size: int = 4       
    hidden_dim: int = 128     
    num_layers: int = 6       
    num_heads: int = 8       
    mlp_dim: int = 256        
    @nn.compact
    def __call__(self, x, train=False):
        x = PatchEmbedding(
            patch_size=self.patch_size,
            hidden_dim=self.hidden_dim
        )(x)
        for _ in range(self.num_layers):
            x = TransformerEncoder(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim
            )(x, train)
        x = nn.LayerNorm()(x)
        x = x[:, 0]  # 取CLS Token（序列第0个位置）
        return nn.Dense(self.num_classes)(x)

def train_vit(epochs=30):
    rng = jax.random.PRNGKey(0)
    model = VisionTransformer(num_classes=10, patch_size=4, hidden_dim=128, num_layers=6)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng, jnp.ones([1,32,32,3]))['params'],
        tx=optax.adamw(0.001, weight_decay=0.0001)
    )    
    train_loader = DataLoader(CIFAR10Dataset(train=True), batch_size=128, shuffle=True)
    test_loader = DataLoader(CIFAR10Dataset(train=False), batch_size=128)    
    @jax.jit
    def train_step(state, batch):
        def loss_fn(p):
            logits = model.apply({'params':p}, batch['image'])
            return optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss   
    @jax.jit
    def eval_step(state, batch):
        logits = model.apply({'params':state.params}, batch['image'])
        return jnp.mean(jnp.argmax(logits,-1) == batch['label'])    
    for epoch in range(epochs):
        losses = []
        for images, labels in train_loader:
            batch = {'image': jnp.array(images.numpy()), 'label': jnp.array(labels.numpy())}
            state, loss = train_step(state, batch)
            losses.append(loss)
        accuracies = [eval_step(state, {'image': jnp.array(images.numpy()), 
                                      'label': jnp.array(labels.numpy())}) 
                     for images, labels in test_loader]
        
        accuracy = np.mean(accuracies)
        print(f"Epoch {epoch+1}: 损失 {np.mean(losses):.4f}, 准确率 {accuracy*100:.2f}%")        
        if accuracy > 0.70: 
            print(f"达到目标! 准确率: {accuracy*100:.2f}%")
            break
if __name__ == "__main__":
    train_vit()