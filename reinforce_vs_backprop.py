import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds

from flax import linen as nn
from flax.training import train_state
import functools
import argparse

import matplotlib.pyplot as plt


class MLP(nn.Module):
    features: list

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features[:-1]]
        self.out = nn.Dense(self.features[-1])

    def apply_f(self, x):
        for layer in self.layers:
            x = nn.relu(layer(x))
        return self.out(x)

    def __call__(self, x):
        return self.apply_f(x)


class CNN(nn.Module):
    num_classes: int

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
        self.dense = nn.Dense(128)
        self.out = nn.Dense(self.num_classes)

    def apply_f(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.relu(self.conv2(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(self.dense(x))
        return x

    def __call__(self, x):
        features = self.apply_f(x)
        logits = self.out(features)
        return logits


# How does this work?
def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    loss = optax.softmax_cross_entropy(logits, one_hot)
    return loss.mean()


def compute_accuracy(logits, labels):
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)


# Backprop
@jax.jit
def train_step_backprop(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = cross_entropy_loss(logits, batch["label"])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = compute_accuracy(logits, batch["label"])
    return state, loss, acc


# Hybrid
@jax.jit
def train_step_hybrid(state, batch, rng):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        return cross_entropy_loss(logits, batch["label"])

    grads_full = jax.grad(loss_fn)(state.params)

    def reinforce_loss(params_out, features, labels, rng):
        logits_l = state.apply_fn(
            {"params": {"out": params_out}}, features, method=lambda mdl, x: mdl.out(x)
        )
        log_probs = jax.nn.log_softmax(logits_l)
        sampled_actions = jax.random.categorical(rng, logits_l)
        log_prob_sampled = jnp.take_along_axis(
            log_probs, sampled_actions[:, None], axis=1
        ).squeeze()
        rewards = (sampled_actions == labels).astype(jnp.float32)
        baseline = jnp.mean(rewards)
        return -jnp.mean((rewards - baseline) * log_prob_sampled)

    features = state.apply_fn(
        {"params": state.params}, batch["image"], method=lambda mdl, x: mdl.apply_f(x)
    )
    grad_out = jax.grad(reinforce_loss)(
        state.params["out"], features, batch["label"], rng
    )

    new_grads = {**grads_full, "out": grad_out}
    state = state.apply_gradients(grads=new_grads)

    # for reporting
    logits = state.apply_fn({"params": state.params}, batch["image"])
    loss = cross_entropy_loss(logits, batch["label"])
    acc = compute_accuracy(logits, batch["label"])
    return state, loss, acc


# Dataset pipeline
def prepare(batch_size):
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))

    train_images_mlp = np.reshape(train_ds["image"], (-1, 28 * 28)) / 255.0
    test_images_mlp = np.reshape(test_ds["image"], (-1, 28 * 28)) / 255.0

    train_images_cnn = np.reshape(train_ds["image"], (-1, 28, 28, 1)) / 255.0
    test_images_cnn = np.reshape(test_ds["image"], (-1, 28, 28, 1)) / 255.0

    train_labels = train_ds["label"]
    test_labels = test_ds["label"]

    return {
        "mlp": (train_images_mlp, train_labels, test_images_mlp, test_labels),
        "cnn": (train_images_cnn, train_labels, test_images_cnn, test_labels),
    }


def get_batches(images, labels, batch_size):
    num = images.shape[0]
    indices = np.arange(num)
    np.random.shuffle(indices)
    for start in range(0, num, batch_size):
        excerpt = indices[start : start + batch_size]
        # FIX: Use labels instead of images for 'label'
        yield {"image": images[excerpt], "label": labels[excerpt]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["mlp", "cnn"], default="mlp", help="Model type to use."
    )
    parser.add_argument(
        "--method", choices=["full", "hybrid"], default="full", help="Training method."
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs to train."
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate."
    )
    args = parser.parse_args()

    datasets = prepare(args.batch_size)
    if args.model == "mlp":
        train_images, train_labels, test_images, test_labels = datasets["mlp"]
        input_shape = (1, 28 * 28)
        model = MLP(features=[128, 64, 10])
    else:
        train_images, train_labels, test_images, test_labels = datasets["cnn"]
        input_shape = (1, 28, 28, 1)
        model = CNN(num_classes=10)

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, args.learning_rate, input_shape)

    num_train = train_images.shape[0]
    steps_per_epoch = num_train // args.batch_size
    logs = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        for batch in get_batches(train_images, train_labels, args.batch_size):
            if args.method == "full":
                state, loss, acc = train_step_backprop(state, batch)
            else:
                rng, step_rng = jax.random.split(rng)
                state, loss, acc = train_step_hybrid(state, batch, step_rng)
        epoch_time = time.time() - start_time

        test_batch = {"image": test_images, "label": test_labels}
        logits = state.apply_fn({"params": state.params}, test_batch["image"])
        test_acc = compute_accuracy(logits, test_batch["label"])
        print(
            f"Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {acc:.4f}, Test Acc: {test_acc:.4f}, Time: {epoch_time:.2f}s"
        )
        logs.append((epoch, loss, acc, test_acc))

    logs_np = np.array(logs)
    plt.figure()
    # plt.plot(logs_np[:, 0], logs_np[:, 1], label='Loss')
    plt.plot(logs_np[:, 0], logs_np[:, 2], label="Train Acc")
    plt.plot(logs_np[:, 0], logs_np[:, 3], label="Test Acc")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(f"./{args.method}_{args.model}.png")
    plt.show()


if __name__ == "__main__":
    main()
