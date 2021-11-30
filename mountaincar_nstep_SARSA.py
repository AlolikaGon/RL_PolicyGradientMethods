import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
import random
import sys

x_range = [-1.2, 0.5]
v_range = [-0.07, 0.07]
x_start = [-0.6, -0.4]
softmax_sigma = 0.1
epsilon = 0.1
M =3
action_set = np.array([-1, 1])

def transition_function(x_, v_, action): #x, v are real numbers
    # x_, v_ = get_realvalues(curr_state)
    v = v_ + 0.001*action - 0.0025*np.cos(3*x_)
    x = x_ + v
    #clip x and v
    if x > x_range[1]:
        x = x_range[1]
        v = 0
    elif x < x_range[0]:
        x = x_range[0]
        v = 0
    if v > v_range[1]:
        v = v_range[1]
    elif v < v_range[0]:
        v = v_range[0]    
    return x, v

def reward(x, v):
    if x == x_range[1]:
        return 0
    else:
        return -1

def normalize(x, v, cosineflag=True):
    if cosineflag:
        x = (x-x_range[0])/(x_range[1]-x_range[0])
        v = (v - v_range[0])/(v_range[1] - v_range[0])        
    else:
        x = 2*(x-x_range[0])/(x_range[1]-x_range[0]) -1
        v = 2*(v - v_range[0])/(v_range[1] - v_range[0]) -1

    return  x, v

def fourier(x, v, cosineflag=False): #4M+1 features
    #normalize
    x, v = normalize(x, v, cosineflag)
    phi = [1]
    if cosineflag:
        for i in range(1, M+1):
            phi.append(np.cos(i*np.pi*x))
        for i in range(1, M+1):
            phi.append(np.cos(i*np.pi*v))
    else:
        for i in range(1, M+1):
            phi.append(np.sin(i*np.pi*x))
        for i in range(1, M+1):
            phi.append(np.sin(i*np.pi*v))
    return np.array(phi)

def softmax_action(policy_params, x, v):    
    phi_s = fourier(x, v) # (2M+1, )
    policy_val = np.zeros(action_set.shape)
    for i in range(len(action_set)):
        policy_val[i] = np.dot(phi_s, policy_params[i*len(phi_s): (i+1)*len(phi_s)])
    policy_exp = np.exp(softmax_sigma*policy_val)
    policy_exp /= np.sum(policy_exp)
    # print(policy_exp, x, x_dot, theta, theta_dot)
    return policy_exp #(2, )

def epsilon_greedy(policy_params, x, v):
    phi_s = fourier(x, v) # (2M+1, )
    policy_val = np.zeros(action_set.shape)
    for i in range(len(action_set)):
        policy_val[i] = np.dot(phi_s, policy_params[i*len(phi_s): (i+1)*len(phi_s)])
    max_idx = np.argmax(policy_val)
    prob = np.ones(policy_val.shape)*epsilon
    prob[max_idx] = 1-epsilon
    return prob

def get_q_sa(policy_params, x, v, action):
    phi_s = fourier(x, v)
    if action == -1:
        return np.dot(phi_s, policy_params[0:len(phi_s)])
    else:
        return np.dot(phi_s, policy_params[len(phi_s):])

def nstep_SARSA(alpha, n, gamma=1.0):
    policy_params = np.zeros((len(action_set)*(2*M+1)))
    episode_length = []
    for iter in range(500):
        policy_params_temp = policy_params.copy()
        #run episode
        state_action_list, reward_list = [], []
        x = np.random.uniform(x_start[0], x_start[1])
        v =0
        curr_action = random.choices(action_set, epsilon_greedy(policy_params, x, v))[0]
        T = 1000000
        t = 0
        step = 0
        while True:
            if t<T:
                state_action_list.append([x,v,curr_action])
                next_x, next_v = transition_function(x, v, curr_action)
                curr_reward = reward(next_x, next_v)
                reward_list.append(curr_reward)
                if next_x == x_range[1] or step >1000: #epsiode is terminating
                    T = t+1
                else:
                    next_action = random.choices(action_set, epsilon_greedy(policy_params, next_x, next_v))[0]
            tau = t -n -1
            if tau>=0:
                G = 0
                for i in range(tau+1, min(tau+n, T)):
                    G += reward_list[i-1]*gamma**(i-tau-1)
                if tau+n<T:
                    # phi_s_tau_plus_n = fourier(state_action_list[tau+n][0], state_action_list[tau+n][1])
                    # if state_action_list[tau+n] == -1:
                    #     G += np.dot(phi_s_tau_plus_n, policy_params[0:len(phi_s_tau_plus_n)])*gamma**n
                    # else:
                    #     G += np.dot(phi_s_tau_plus_n, policy_params[len(phi_s_tau_plus_n):])*gamma**n

                    G += get_q_sa(policy_params, state_action_list[tau+n][0], state_action_list[tau+n][1], state_action_list[tau+n][2])
                phi_stau = fourier(state_action_list[tau][0], state_action_list[tau][1])
                q_stau_atau = get_q_sa(policy_params, state_action_list[tau][0], state_action_list[tau][1], state_action_list[tau][2])
                if state_action_list[tau][-1] == -1:    
                    policy_params[0:(2*M+1)] += alpha*(G - q_stau_atau)*phi_stau
                else:
                    policy_params[(2*M+1):] += alpha*(G - q_stau_atau)*phi_stau
            x, v, curr_action = next_x, next_v, next_action
            step += 1
            t += 1
            if tau == T-1:
                break
        episode_length.append(step)
        max_diff = np.max(np.abs(policy_params - policy_params_temp))
        print(" iteration ", iter, "EPISODE length ", step, " max_diff ", max_diff)
        if max_diff < 1e-3:
            break
    plt.figure()
    plt.plot(range(len(episode_length)), episode_length)
    plt.xlabel('Iterations')
    plt.ylabel('Epsiode length')
    plt.savefig('../RL_project_graphs/mountaincar_nstep_SARSA')

nstep_SARSA(alpha = 3e-4, n=8, gamma=1.0)                
    