import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
import random
import sys

X_range = [-4.8, 4.8]
v_range = [-10, 10]#[-100000, 100000] #[float('-inf'), float('inf')]
theta_range = [-24, 24]
anglev_range = [-10, 10] #[-100000, 100000]#[float('-inf'), float('inf')]
start_range = [-0.05, 0.05]

terminating_cond =[2.4, 12, 200]
action_set = [0,1] #left, right

M = 3
epsilon = 0.1

def in_radian(ang):
    return ang*np.pi/180

def transition(action, x, x_dot, theta, theta_dot):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5  # actually half the pole's length
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02

    force = force_mag if action == 1 else -force_mag
    costheta = np.cos(theta) # theta in radians
    sintheta = np.sin(theta)

    # from gym https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    
    #euler
    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
    
    #semi euler
    # x_dot = x_dot + tau * xacc
    # x = x + tau * x_dot
    # theta_dot = theta_dot + tau * thetaacc
    # theta = theta + tau * theta_dot

    return x, x_dot, theta, theta_dot
    
def is_terminating(x, x_dot, theta, theta_dot, step):
    if x <= -terminating_cond[0] or x >= terminating_cond[0] or theta <= -in_radian(terminating_cond[1]) or theta >= in_radian(terminating_cond[1]) or step>=terminating_cond[2]:
        return True
    return False

def reward(x, x_dot, theta, theta_dot, step):
    if is_terminating(x, x_dot, theta, theta_dot, step):
        return 0
    return 1

def normalize(x, x_dot, theta, theta_dot, cosineflag=True):
    if cosineflag:
        x = (x-X_range[0])/(X_range[1]-X_range[0])
        theta = (theta-theta_range[0])/(theta_range[1]-theta_range[0])
        x_dot = (x_dot - v_range[0])/(v_range[1] - v_range[0])
        theta_dot = (theta_dot - anglev_range[0])/(anglev_range[1] - anglev_range[0])
        
    else:
        x = 2*(x-X_range[0])/(X_range[1]-X_range[0]) -1
        theta = 2*(theta-theta_range[0])/(theta_range[1]-theta_range[0]) -1
        x_dot = 2*(x_dot - v_range[0])/(v_range[1] - v_range[0]) -1
        theta_dot = 2*(theta_dot - anglev_range[0])/(anglev_range[1] - anglev_range[0]) -1

    return  x, x_dot, theta, theta_dot

def fourier(x, x_dot, theta, theta_dot, cosineflag=False): #4M+1 features
    #normalize
    x, x_dot, theta, theta_dot = normalize(x, x_dot, theta, theta_dot, cosineflag)
    phi = [1]
    if cosineflag:
        for i in range(1, M+1):
            phi.append(np.cos(i*np.pi*x))
        for i in range(1, M+1):
            phi.append(np.cos(i*np.pi*x_dot))
        for i in range(1, M+1):
            phi.append(np.cos(i*np.pi*theta))
        for i in range(1, M+1):
            phi.append(np.cos(i*np.pi*theta_dot))
    else:
        for i in range(1, M+1):
            phi.append(np.sin(i*np.pi*x))
        for i in range(1, M+1):
            phi.append(np.sin(i*np.pi*x_dot))
        for i in range(1, M+1):
            phi.append(np.sin(i*np.pi*theta))
        for i in range(1, M+1):
            phi.append(np.sin(i*np.pi*theta_dot))
    return np.array(phi)

def epsilon_greedy(policy_params, x, x_dot, theta, theta_dot):
    phi_s = fourier(x, x_dot, theta, theta_dot) # (4M+1, )
    policy_val = np.zeros((len(action_set)))
    for i in range(len(action_set)):
        policy_val[i] = np.dot(phi_s, policy_params[i*len(phi_s): (i+1)*len(phi_s)])
    max_idx = np.argmax(policy_val)
    prob = np.ones(policy_val.shape)*epsilon
    prob[max_idx] = 1-epsilon
    return prob

def get_q_sa(policy_params, x, x_dot, theta, theta_dot, action):
    phi_s = fourier(x, x_dot, theta, theta_dot)
    if action == -1:
        return np.dot(phi_s, policy_params[0:len(phi_s)])
    else:
        return np.dot(phi_s, policy_params[len(phi_s):])

def nstep_SARSA(alpha, n, gamma=1.0):
    policy_params = np.zeros((len(action_set)*(4*M+1)))
    episode_length = []
    for iter in range(500):
        policy_params_temp = policy_params.copy()
        #run episode
        state_action_list, reward_list = [], []
        x = np.random.uniform(start_range[0], start_range[1])
        theta = np.random.uniform(start_range[0], start_range[1])
        x_dot  = np.random.uniform(start_range[0], start_range[1])
        theta_dot = np.random.uniform(start_range[0], start_range[1])

        curr_action = random.choices(action_set, epsilon_greedy(policy_params, x, x_dot, theta, theta_dot))[0]
        T = 1000000
        t = 0
        step = 0
        while True:
            if t<T:
                state_action_list.append([x, x_dot, theta, theta_dot, curr_action])
                next_x, next_x_dot, next_theta, next_theta_dot = transition(x, x_dot, theta, theta_dot, curr_action)
                curr_reward = reward(next_x, next_x_dot, next_theta, next_theta_dot, step)
                reward_list.append(curr_reward)
                print(x, x_dot, theta, theta_dot, curr_action)
                if is_terminating(next_x, next_x_dot, next_theta, next_theta_dot, step): #epsiode is terminating
                    T = t+1
                else:
                    next_action = random.choices(action_set, epsilon_greedy(policy_params, next_x, next_x_dot, next_theta, next_theta_dot))[0]
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

                    G += get_q_sa(policy_params, state_action_list[tau+n][0], state_action_list[tau+n][1], state_action_list[tau+n][2], state_action_list[tau+n][3], state_action_list[tau+n][4])
                phi_stau = fourier(state_action_list[tau][0], state_action_list[tau][1], state_action_list[tau][2], state_action_list[tau][3])
                q_stau_atau = get_q_sa(policy_params, state_action_list[tau][0], state_action_list[tau][1], state_action_list[tau][2], state_action_list[tau][3], state_action_list[tau][4])
                if state_action_list[tau][-1] == -1:    
                    policy_params[0:(4*M+1)] += alpha*(G - q_stau_atau)*phi_stau
                else:
                    policy_params[(4*M+1):] += alpha*(G - q_stau_atau)*phi_stau
            x, x_dot, theta, theta_dot, curr_action = next_x, next_x_dot, next_theta, next_theta_dot, next_action
            step += 1
            t += 1
            if tau == T-1:
                break
        episode_length.append(step)
        max_diff = np.max(np.abs(policy_params - policy_params_temp))
        print(" iteration ", iter, "EPISODE length ", t, " max_diff ", max_diff, '\n')
        if max_diff < 1e-3:
            break
    
    plt.figure()
    plt.plot(range(len(episode_length)), episode_length)
    plt.xlabel('Iterations')
    plt.ylabel('Epsiode length')
    plt.savefig('../RL_project_graphs/cartpole_nstep_SARSA')

nstep_SARSA(alpha = 3e-2, n=1, gamma=1.0)