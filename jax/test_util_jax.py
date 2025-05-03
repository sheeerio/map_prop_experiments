from util import *
from util_jax import *
import numpy as np
from numpy.testing import assert_almost_equal


def before_each():
    global x
    x = np.random.rand(3, 3)


def test_relu():
    before_each()
    assert_almost_equal(jax_relu(x), relu(x))


def test_relu_grad():
    before_each()
    assert_almost_equal(jax_relu_grad(x), relu_d(x))


def test_sigmoid():
    before_each()
    assert_almost_equal(jax_sigmoid(x), sigmoid(x))


def test_sigmoid_grad():
    before_each()
    assert_almost_equal(jax_sigmoid_grad(x), sigmoid_d(x))


def test_softplus():
    before_each()
    assert_almost_equal(jax_softplus(x), softplus(x))


def test_softmax():
    before_each()
    assert_almost_equal(jax_softmax(x), softmax(x))


def test_equal_zero():
    before_each()
    assert_almost_equal(jax_equal_zero(x), equal_zero(x))


def test_multinomial_rvs():
    key = jrandom.PRNGKey(2025)
    n = 10
    p = np.array([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    sample_jax = jax_multinomial_rvs(key, n, p)
    sample_np = multinomial_rvs(n, p)
    # print(f"sample_jax : {sample_jax}, \nsample_np : {sample_np}")


def test_one_hot():
    a = np.array([3, 6])
    size = 10
    jax_oh = jax_to_one_hot(a, size)
    np_oh = to_one_hot(a, size)
    assert_almost_equal(jax_oh, np_oh)
    assert_almost_equal(jax_from_one_hot(jax_oh), from_one_hot(np_oh))


def test_equal_zero():
    a = np.array([1, 3, 4, 5])
    assert_almost_equal(jax_equal_zero(a), equal_zero(a))
    a = np.zeros(5)
    assert_almost_equal(jax_equal_zero(a), equal_zero(a))


def test_adam_optimizer():
    jax_adam = jax_adam_optimizer(learning_rate=1e-5)
    np_adam = adam_optimizer(learning_rate=1e-5)
    grads = np.array([-1, 0.1, 2, 0.5])
    jax_grad1 = np.array(jax_adam.delta(grads=grads))
    np_grad1 = np.array(np_adam.delta(grads=grads))
    assert_almost_equal(jax_grad1, np_grad1)
    assert_almost_equal(jax_adam.delta(grads=jax_grad1), np_adam.delta(grads=np_grad1))


if __name__ == "__main__":
    test_sigmoid()
    test_softmax()
    test_multinomial_rvs()
    test_one_hot()
    test_adam_optimizer()
    # For example, use the CartPole environment.
    batch_size = 1
    env = jax_reg_MDP()
    key = jax.random.PRNGKey(2019)
    state = env.reset(key, batch_size)
    print("Initial observations:\n", state)

    actions = jax.random.randint(env.key, shape=(batch_size,), minval=0, maxval=2)

    rewards = env.act(actions)
    print("Rewards :\n", rewards)

    np_env = reg_MDP()
    state = np_env.reset(batch_size)
    print("Initial observations:\n", state)
    actions = np.random.randint(size=(batch_size,), low=0, high=2)

    rewards = np_env.act(actions)
    print("Rewards :\n", rewards)

    batch_size = 8
    rest_n = 100
    warm_n = 100
    env, env_params = gymnax.make("Pendulum-v1")

    # Create a PRNG key.
    key = jrandom.PRNGKey(0)

    # Reset the environment batch.
    state = batch_reset(key, env, env_params, batch_size)

    # Define a rollout step function.
    def rollout_step(carry, _):
        key, state = carry
        # Split key for this step.
        key, subkey = jrandom.split(key)
        # For simplicity, we choose dummy actions (zeros) for each env in the batch.
        # (Make sure that the shape and dtype match what env.step expects.)
        actions = jnp.zeros((batch_size,), dtype=jnp.float32)
        # Step the environment wrapper.
        new_state, (obs, reward, done, info) = env_wrapper_step(
            subkey, state, actions, env, env_params, rest_n, warm_n
        )
        return (key, new_state), (obs, reward, done, info)

    # Simulate 100 steps using lax.scan.
    num_steps = 200
    (final_key, final_state), rollout_outputs = jax.lax.scan(
        rollout_step,
        (key, state),
        None,  # No additional sequence input.
        length=num_steps,
    )

    # rollout_outputs is a tuple: (obs_seq, reward_seq, done_seq, info_seq)
    # Each element in the tuple has shape (num_steps, batch_size, ...)

    # For example, print the shape of observations:
    obs_seq, reward_seq, done_seq, info_seq = rollout_outputs
    print("Observations shape:", obs_seq.shape)
    print("Rewards shape:", reward_seq.shape, reward_seq)
    print("Done flags shape:", done_seq.shape, done_seq)
