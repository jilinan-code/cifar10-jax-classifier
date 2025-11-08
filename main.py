import os
# 完全禁用 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
import tensorflow as tf
# 确保 TensorFlow 使用 CPU
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import numpy as np
# 强制 JAX 使用 CPU
jax.config.update('jax_platform_name', 'cpu')
print("使用的设备:", jax.default_backend())
def load_cifar10(batch_size=128):
    """加载CIFAR-10数据集并进行预处理"""
    # 下载数据集
    download_dir = '/tmp/tensorflow_datasets'
    os.makedirs(download_dir, exist_ok=True)
    ds_builder = tfds.builder('cifar10', data_dir=download_dir)
    ds_builder.download_and_prepare()
    
    def preprocess(data):
        image = tf.cast(data['image'], tf.float32) / 255.0
        image = (image - 0.5) / 0.5  # 标准化到[-1,1]
        label = data['label']
        return {'image': image, 'label': label}
    # 加载数据集
    train_ds = ds_builder.as_dataset(split='train', shuffle_files=True)
    test_ds = ds_builder.as_dataset(split='test', shuffle_files=False)
    # 应用预处理
    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)
    # 批处理
    train_ds = train_ds.batch(batch_size)
    test_ds = test_ds.batch(batch_size)
    return train_ds, test_ds
class SimpleCNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        
        x = x.reshape((x.shape[0], -1))  # 展平
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)       
        return x
def create_train_state(rng, learning_rate=0.001):
    model = SimpleCNN()
    dummy_input = jnp.ones((1, 32, 32, 3))
    params = model.init(rng, dummy_input)['params']
    
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
# 训练步骤
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        one_hot_labels = jax.nn.one_hot(batch['label'], 10)
        loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
# 评估步骤
@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    predicted_class = jnp.argmax(logits, axis=1)
    accuracy = jnp.mean(predicted_class == batch['label'])
    
    one_hot_labels = jax.nn.one_hot(batch['label'], 10)
    loss = optax.softmax_cross_entropy(logits, one_hot_labels).mean()
    
    return loss, accuracy
# 训练循环
def train_model(epochs=10):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng)
    print("加载数据集...")
    train_ds, test_ds = load_cifar10(batch_size=64) 
    print("开始训练...")
    for epoch in range(epochs):
        train_losses = []
        for batch in train_ds:
            # 转换为 JAX 数组
            batch = {k: jnp.array(v) for k, v in batch.items()}
            state, loss = train_step(state, batch)
            train_losses.append(loss)
        # 评估
        test_losses = []
        test_accuracies = []
        for batch in test_ds:
            batch = {k: jnp.array(v) for k, v in batch.items()}
            loss, accuracy = eval_step(state, batch)
            test_losses.append(loss)
            test_accuracies.append(accuracy)
        
        avg_train_loss = np.mean(train_losses)
        avg_test_loss = np.mean(test_losses)
        avg_test_accuracy = np.mean(test_accuracies)
        
        print(f"Epoch {epoch + 1}/{epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}, 测试损失: {avg_test_loss:.4f}")
        print(f"  测试准确率: {avg_test_accuracy:.4f} ({avg_test_accuracy*100:.2f}%)")
        
        if avg_test_accuracy > 0.50:
            print(f"达到目标准确率！当前准确率: {avg_test_accuracy*100:.2f}%")
            break
    
    return state
if __name__ == "__main__":
    print("平台信息:", jax.default_backend())
    print("开始训练CIFAR-10分类器...")
    
    trained_state = train_model(epochs=10)
        # 最终测试
    _, test_ds = load_cifar10(batch_size=64)
    final_accuracies = []
    for batch in test_ds:
        batch = {k: jnp.array(v) for k, v in batch.items()}
        _, accuracy = eval_step(trained_state, batch)
        final_accuracies.append(accuracy)
        final_accuracy = np.mean(final_accuracies)
        print(f"\n最终测试准确率: {final_accuracy*100:.2f}%")  
    if final_accuracy > 0.50:
        print("成功达到目标准确率！")