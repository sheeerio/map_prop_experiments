from abc import ABC, abstractmethod
import numpy as np 
import gym
import torch
import matplotlib
import matplotlib.pyplot as plt

def relu(x):
  y = np.copy(x)
  y[y<0] = 0
  return y

def relu_d(x):  
  y = np.ones_like(x)
  y[x<0] = 0
  return y


def sigmoid(x):
  lim = 20
  l = np.zeros_like(x)
  l[np.abs(x) < lim] = 1/(1+np.exp(-x[np.abs(x) < lim]))
  l[x <= -lim] = 0
  l[x >= lim] = 1
  return l

def softplus(x):
  r = np.zeros_like(x)
  r[x>30] = x[x>30]
  r[np.abs(x)<=30] = np.log1p(np.exp(x[np.abs(x)<=30]))
  return r
  #return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)
  #return np.log(1+np.exp(x))

def softplus_i(x):
  r = np.zeros_like(x)
  r[x>30] = x[x>30]
  r[x<0.01] = np.log(-1+np.exp(0.01))
  b_flag = (x>=0.01) & (x<=30)
  r[b_flag] = np.log(-1+np.exp(x[b_flag]))
  return r

def softmax(X, theta=1.0, axis=None):
    y = np.atleast_2d(X)
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)
    y = y * float(theta)
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    y[y<-30] = -30
    y = np.exp(y)    
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    p = y / ax_sum
    if len(X.shape) == 1: p = p.flatten()
    return p

def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out

BINARY_LIN_P = 0.0001
BINARY_LIN_M = 1.000001

def binary_lin_z(x):
    p = BINARY_LIN_P
    c_1 = -np.log(p/((1-p)))
    c_2 = BINARY_LIN_M*c_1
    y = np.zeros_like(x)
    y[x<-c_2] = 0.
    y[(-c_2<=x) & (x<-c_1)] = ((p*x+c_2*p)/(c_2-c_1))[(-c_2<=x) & (x<-c_1)]
    y[(-c_1<=x) & (x<c_1)] = ((0.5-p)/c_1*x + 0.5)[(-c_1<=x) & (x<c_1)]
    y[(c_1<=x) & (x <c_2)] = ((p*x+c_2-c_1-p*c_2)/(c_2-c_1))[(c_1<=x) & (x <c_2)] 
    y[x >= c_2] = 1.
    return y

def binary_lin_z_d(x):
    p = BINARY_LIN_P
    c_1 = -np.log(p/((1-p)))
    c_2 = BINARY_LIN_M*c_1
    y = np.zeros_like(x)
    y[(-c_2<=x) & (x<-c_1)] = (p/(c_2-c_1))
    y[(-c_1<=x) & (x<c_1)] = ((0.5-p)/c_1)
    y[(c_1<=x) & (x <c_2)] = (p/(c_2-c_1))
    return y

def from_one_hot(y):      
  return np.argmax(y, axis=-1)

def to_one_hot(a, size):
  oh = np.zeros((a.shape[0], size), np.int)
  oh[np.arange(a.shape[0]), a.astype(int)] = 1
  return oh

def getl(x, n):
  return x[n] if type(x) == list else x        
        
def equal_zero(x):
  return np.logical_and(x > -1e-8, x < 1e-8).astype(np.float32)

def mask_neg(x):
  return (x < 0).astype(np.float32) 
               
def sign(x):
  return (x > 1e-8).astype(np.float32) - (x < -1e-8).astype(np.float32)

def zero_to_neg(x):
  return (x > 1e-8).astype(np.float32) - (x <= 1e-8).astype(np.float32)

def neg_to_zero(x):
  return (x > 1e-8).astype(np.float32)  

def sum_list_map(x, y, f=None):
  if f is None:
    return [x[n]+y[n] for n in range(len(x))]
  else:
    return [x[n]+f(y[n]) for n in range(len(x))]
    
def apply_mask(x, mask):        
  return (x.T * mask).T

def linear_interpolat(start, end, end_t, cur_t):  
  if type(start) == list:
    if type(end_t) == list:
      return [(e - s) * min(cur_t, d) / d + s for (s, e, d) in zip(start, end, end_t)]
    else:    
      return [(e - s) * min(cur_t, end_t)  / end_t + s for (s, e) in zip(start, end)]
  else:
    if type(end_t) == list:
      return [(end - start) * min(cur_t, d) / d + s for d in end_t]
    else:          
      return (end - start) * min(cur_t, end_t) / end_t + start     
  
class MDP(ABC): 
  def __init__(self):
    super().__init__()        

  @abstractmethod
  def reset(self, batch_size):
    # return states with type np.array of size (batch_size, x_size)
    pass

  @abstractmethod
  def act(self, actions):
    # return rewards with type np.array of size (batch_size)
    # actions: action with type np.array of size (batch_size, action_size)
    pass
    
class xor_MDP(MDP):
  def __init__(self):
    self.x_size=2
    self.action_size=1
    self.x=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    self.y=np.array([1, 0, 0, 1])
    super().__init__()

  def reset(self, batch_size):
    self.s = np.random.choice(4, batch_size)
    return self.x[self.s]

  def act(self, actions):
    corr = (self.y[self.s][:, np.newaxis] == actions.astype(np.int32)).astype(np.float32)
    return np.random.binomial(1, 0.8*corr+0.1).astype(np.float32)

class complex_multiplexer_MDP(MDP):
  def __init__(self, addr_size=2, action_size=2, zero=True, reward_zero=True):
    self.addr_size=addr_size
    self.action_size=action_size      
    self.x_size= addr_size*action_size + 2 ** addr_size
    self.zero = zero
    self.reward_zero = reward_zero
    super().__init__()

  def reset(self, batch_size):
    addr_size = self.addr_size
    action_size = self.action_size
    x_size = self.x_size        
    self.x = np.random.binomial(1, 0.5, size=(batch_size, x_size))
    addr = np.sum(self.x[:, :addr_size*action_size].reshape([-1, action_size, addr_size]) * 2**(addr_size-1-np.arange(addr_size)), axis=2)        
    self.y = self.x[np.arange(self.x.shape[0])[:, np.newaxis], addr_size*action_size + addr]
    if not self.zero: self.y = zero_to_neg(self.y)
    return self.x if self.zero else zero_to_neg(self.x)

  def act(self, actions):
    corr = (self.y == actions.astype(np.int32)).astype(np.float32)        
    return corr if self.reward_zero else zero_to_neg(corr)

  def expected_reward(self, p):
    # for action_size=2 only
    reward_f = np.full((self.x.shape[0], 2), 0 if self.reward_zero else -1)
    y_zero = self.y if self.zero else neg_to_zero(self.y)
    reward_f[np.arange(self.x.shape[0]), y_zero[:,0].astype(np.int)] = 1
    return np.sum(reward_f * p, axis=-1)
      
class k_arm_binary_bandit(MDP):
  def  __init__(self, action_size=2, p=np.array([[0.9, 0.9], [0.1, 0.1]]), zero=False):
    self.action_size = action_size
    self.p=p
    self.zero=zero
    super().__init__()

  def reset(self, batch_size):
    return None
  
  def act(self, actions):
    index = actions.T
    if not self.zero: index = neg_to_zero(index)
    index = index.astype(int)
    p = self.p[tuple(index)]
    r = np.random.binomial(1, p)
    return r if self.zero else zero_to_neg(r)

class k_arm_binary_bandit_2(MDP):
  def  __init__(self, action_size=100, c=30, zero=False):
    self.c = c
    self.action_size=action_size
    self.zero=zero
    super().__init__()

  def reset(self, batch_size):
    return None
  
  def act(self, actions):
    action_sum = np.sum(actions==1, axis=-1)
    r = (action_sum < self.c) * (action_sum/self.c) + (action_sum >= self.c) * (1 - (action_sum - self.c)/self.c)
    r = np.clip(r, 0, 1)
    if not self.zero: r = r * 2 - 1
    return r

class simple_grad_optimizer():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def delta(self, name, grads, learning_rate=None):                  
        learning_rate = self.learning_rate if learning_rate is None else learning_rate                
        return [learning_rate*i for i in grads]
    
class adam_optimizer():
    def __init__(self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-09):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self._cache = {}        
    
    def delta(self, grads, name="w", learning_rate=None, gate=None):
        if name not in self._cache:
            self._cache[name] = [[np.zeros_like(i) for i in grads],
                                 [np.zeros_like(i) for i in grads],
                                 0]
        self._cache[name][2] += 1 
        t = self._cache[name][2]
        deltas = []
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        learning_rate = self.learning_rate if learning_rate is None else learning_rate        
        for n, g in enumerate(grads):                
            m = self._cache[name][0][n]
            v = self._cache[name][1][n]
            m = beta_1 * m + (1 - beta_1) * g
            v = beta_2 * v + (1 - beta_2) * np.power(g, 2)
            if gate is not None:
              self._cache[name][0][n] = m
              self._cache[name][1][n] = v            
            else:
              self._cache[name][0][n][gate] = m[gate]
              self._cache[name][1][n][gate] = v[gate]

            m_hat = m / (1 - (np.power(beta_1, t) if t < 1000 else 0))
            v_hat = v / (1 - (np.power(beta_2, t) if t < 1000 else 0))
            deltas.append(learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon))
        
        return deltas 


def plot(curves, names=None, mv_n=100, xlabel="Episodes", ylabel="Running Average Return", ylim=None, loc=4):  
  if names is None:
    names = {i:i for i in curves.keys()}
  plt.figure(figsize=(5*2, 2.5*2), dpi=80)   
  for i, m in enumerate(names.keys()):    
    v = np.array([mv(i, mv_n) for i in curves[m]])
    v = np.mean(v, axis=0)
    r_std = np.std(v, axis=0)/np.sqrt(len(curves[m]))
    v = np.concatenate([np.full([mv_n-1,],np.nan), v])
    k = names[m]
    ax = plt.gca()         
    ax.plot(np.arange(len(v)), v, label=k)
    ax.fill_between(np.arange(len(v)), v-r_std, v+r_std, label=None, alpha=0.2)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)      
  if ylim is not None: plt.ylim(ylim)
  plt.gca().legend(loc=loc)
  plt.set_cmap('jet')
  plt.show()
    
def errorfill(x, y, yerr, alpha_fill=0.0, ax=None, label=None):
  ax = ax if ax is not None else plt.gca()
  if np.isscalar(yerr) or len(yerr) == len(y):
    ymin = y - yerr
    ymax = y + yerr
  elif len(yerr) == 2:
    ymin, ymax = yerr
  ax.plot(x, y, label=label)    
  ax.fill_between(x, ymax, ymin, alpha=alpha_fill)    

def mv(a, n=1000) :
  ret = np.cumsum(a, dtype=float)
  ret[n:] = ret[n:] - ret[:-n]
  return ret[n - 1:] / n      


# Following are util funcitons for learning value function

def real_to_bin(real, bin_min=0, bin_max=1, bin_num=10):  
  bin_index = np.array((real - bin_min)/(bin_max-bin_min) * bin_num).astype(np.int)
  bin_index -= 1
  bin_index[bin_index>=bin_num] = bin_num-1
  bin_index[bin_index<0] = 0
  return bin_index

class batch_envs():
  
  def __init__(self, name, batch_size=1, rest_n=100, warm_n=100):
    self._batch_size = int(batch_size)
    self._action = None
    self._reward = np.zeros(batch_size)
    self._isEnd = np.ones(batch_size, bool)
    self._truncatedEnd = np.zeros(batch_size, bool)
    self._rest = np.zeros(batch_size)
    self._warm = np.zeros(batch_size)
    self._state = np.zeros((batch_size,)+gym.make(name).reset().shape)
    self._stateCode = np.zeros(batch_size, np.int)
    self._rest_n = rest_n
    self._warm_n = warm_n     
    self._env = [gym.make(name) for _ in range(batch_size)]    

  @property
  def name(self):
    return self._name

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def reward(self):
    return self._reward

  @property
  def action(self):
    return self._action

  @property
  def action_space(self):
    return self._env[0].action_space

  @property
  def isEnd(self):
    return self._isEnd

  @property
  def stateCode(self):
    return self._stateCode

  @property
  def state(self):
    return self._state    

  @property
  def info(self):
    return {"stateCode": self._stateCode, "truncatedEnd": self._truncatedEnd}
  
  def step(self, action):
    self._rest[self._isEnd] += 1
    self._warm += 1
    self.reset(self._rest>self._rest_n)      
    isLive = np.logical_and(self._warm>self._warm_n, ~self._isEnd)
    self._reward[~isLive] = 0
    for i in isLive.nonzero()[0]:   
      self._state[i], self._reward[i], self._isEnd[i], info = self._env[i].step(action[i])  
      self._truncatedEnd[i] = info['TimeLimit.truncated'] if 'TimeLimit.truncated' in info else False
      if self._truncatedEnd[i]: self._rest[i] = self._rest_n

    # Live  
    self._stateCode[isLive] = 0            
    # Rest
    self._stateCode[self._rest>=1] = 1
    # Warm up
    self._stateCode[self._warm<= self._warm_n] = 3    
    # Reset
    self._stateCode[self._warm==0] = 2

    return self.state, self.reward, self._isEnd, self.info

  def reset(self, index=None):    
    for i in range(self._batch_size) if index is None else index.nonzero()[0]:
      self._state[i] = self._env[i].reset()
      self._reward[i] = 0
    if index is None: index = slice(None)  
    self._rest[index] = 0
    self._warm[index] = 0
    self._truncatedEnd[index] = False
    self._isEnd[index] = False
    return self._state

class uniform_policy():
  def __init__(self, batch_size, actions):
    self.actions = np.array(actions, np.int)
    self.batch_size = batch_size
  
  def __call__(self, state):
    return np.random.choice(self.actions, size=self.batch_size)    

def est_V(env, policy, T, bin_min, bin_max, bin_num, gamma):
  # Estimate value of V_\pi by Monte Carlo
  batch_size = env._batch_size  
  state_size = env._state.shape[1]
  state_hist = np.zeros((batch_size, state_size, T))  
  reward_hist = np.zeros((batch_size, T))
  isEnd_hist = np.zeros((batch_size, T), bool)
  stateCode_hist = np.zeros((batch_size, T))
  
  state = env.reset()    
  for t in range(T):    
    state_hist[:, :, t] = state
    action = policy(state)
    state, reward, isEnd, info = env.step(action)    
    reward_hist[:, t] = reward
    isEnd_hist[:, t] = isEnd    
    stateCode_hist[:, t] = info['stateCode']

  ret_sum = np.zeros((bin_num,)*state_size)
  ret_num = np.zeros((bin_num,)*state_size)
  value = np.zeros((bin_num,)*state_size)
  
  complete = np.zeros(batch_size, bool)
  ret = np.zeros(batch_size)
  
  for t in range(T-1, -1, -1):
    state = state_hist[:, :, t]
    reward = reward_hist[:, t]
    isEnd = isEnd_hist[:, t]
    stateCode = stateCode_hist[:, t]
    ret[isEnd] = 0.
    complete[isEnd] = True    
    valid = np.logical_and(complete, stateCode == 0)
    bin_state = tuple(real_to_bin(real=state[valid], bin_min=bin_min, bin_max=bin_max, bin_num=bin_num).T)
    ret = gamma*ret + reward    
    if np.any(valid):      
      ret_sum[bin_state] += ret[valid]*np.sqrt(gamma)
      ret_num[bin_state] += 1
    
  value[ret_num!=0]=ret_sum[ret_num!=0]/ret_num[ret_num!=0]
  return value, ret_num, ret_sum  

def batch_autograd(v, inputs, batch_size, **args):
  grads = []
  for i in inputs:
    grads.append(torch.zeros((batch_size,)+tuple(i.size())))
  for b in range(batch_size):
    g = torch.autograd.grad(v[b], inputs=inputs, **args)
    for k in range(len(inputs)):
      grads[k][b] = g[k] if g[k] is not None else 0.
  return grads

def ls_sum(a, b):
  for i in range(len(a)):
    a[i] =  (a[i] if a[i] is not None else 0) + (b[i] if b[i] is not None else 0)     

class State_to_spike_Fourier():  
  def __init__(self, time, bin_min, bin_max, bin_num, k=2, n=4, basis=False, soft=False, cross_term=True, double=False, rep=1):
    # create vector of coefficient
    if type(k) != np.ndarray:
      if type(k) == list:
        k = np.array(k)
      else:
        k = np.array([k]*n)
    if cross_term:
      self._c = np.zeros((n, np.prod(k+1)))
      for i in range(np.prod(k+1)):
          l = i
          for j in range(n):    
              self._c[j, i] = l % (k[j]+1)
              l //= (k[j]+1)
              if l == 0: break
    else:
      self._c = np.zeros((n, np.sum(k)))
      j, l = 0, 0
      for i in range(np.sum(k)):
        l += 1
        self._c[j, i] = l  
        if l >= k[j]:
          l = 0
          j += 1
    self._k = k
    self._n = n    
    self.mean_re = (bin_max + bin_min) / 2
    self.range_re = bin_max - bin_min
    self.time = time    
    self.out_shape = (((self._c.shape[-1])*(2 if double else 1)+(1 if basis else 0))*rep,1)
    self.basis = basis
    self.soft = soft
    self.double = double
    self.rep = rep
      
  def __call__(self, state, augment=None):    
    norm_state = (state - self.mean_re) / self.range_re + 0.5  
    norm_state = np.clip(norm_state, 0, 1)  
    if self.double:
      f_basis_p = relu(np.cos(np.pi * np.dot(norm_state, self._c)))
      f_basis_n = relu(-np.cos(np.pi * np.dot(norm_state, self._c)))
      f_basis = np.concatenate([f_basis_p, f_basis_n], -1)      
    else:
      f_basis = (np.cos(np.pi * np.dot(norm_state, self._c))+1)/2 
    if self.basis:
      f_basis = np.concatenate([f_basis, np.ones((state.shape[0], 1))], axis=-1)
    if augment is not None:
      f_basis = np.concatenate([f_basis, augment], axis=-1)        
      
    if self.soft: 
      bin_output = np.broadcast_to(f_basis[np.newaxis, :, :, np.newaxis,], (self.time, state.shape[0], self.out_shape[0]//self.rep, self.rep))
      bin_output = bin_output.reshape(self.time, state.shape[0], self.out_shape[0], 1)
    else:
      bin_output = np.random.binomial(n=1,p=f_basis, size=(self.time, self.rep,)+f_basis.shape)      
      bin_output = np.swapaxes(bin_output, 1, 2).reshape(self.time, state.shape[0], -1, 1)       
    return torch.from_numpy(bin_output[0,:,:,0]).float()         