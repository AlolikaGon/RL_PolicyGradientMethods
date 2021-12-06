import numpy as np
np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
import random
import sys
np.random.seed(1)
max_r, max_c = 4, 4
# state_value = np.zeros((5,5))
action_set = {'AU': 0, 'AR': 1, 'AD': 2, 'AL': 3}
action_coord = [[-1,0], [0,1], [1,0], [0,-1]]
terminal_states = [[4,4]]
obstacles = [[2,2], [3,2]]
obstacle_index = [(max_r+1)*r+c for r, c in obstacles]
water = [[4,2]]
initial_states = [[0,0]]# [[i,j] for j in range(max_c+1) for i in range(max_r+1) if [i,j] not in obstacles+terminal_states]
softmax_sigma=1.0

print_arrow = {'AU': u'\u2191', 'AR': u'\u2192', 'AL': u'\u2190', 'AD': u'\u2193', 'G': 'G'}

def transition_function(state_r, state_c, action, prob = [0.8, 0.05, 0.05, 0.1]):
# def transition_function(state_r, state_c, action, prob = [1.0, 0.0, 0.0, 0.0]): #deterministic
    if action not in action_set:
        print("wrong action")
        sys.exit(0)
    if state_r == max_r and state_c == max_c:
        return []
    action_idx = action_set[action]
    action_idx_r = (action_idx+1)%4
    action_idx_l = (action_idx-1)%4
    # del_r, del_c = random.choices([action_coord[action_idx], action_coord[action_idx_r], action_coord[action_idx_l], [0,0]], weights=[0.8, 0.05, 0.05, 0.1])
    future_states = []
    temp = [action_coord[action_idx], action_coord[action_idx_r], action_coord[action_idx_l], [0,0]]
    for i in range(4):
        del_r, del_c = temp[i]
        new_state_r, new_state_c = state_r, state_c
        if 0<=state_r + del_r <= max_r:
            new_state_r += del_r
        if 0<=state_c + del_c <= max_c:
            new_state_c += del_c
        if [new_state_r, new_state_c] in obstacles:
            new_state_r, new_state_c = state_r, state_c
        future_states.append([new_state_r, new_state_c, prob[i]]) 
    return future_states

def reward(next_state, gold_state_reward=5):
    if next_state in terminal_states:
        return 10
    if next_state in water:
        return -10    
    return 0

def softmax_select(theta, phi, state_idx):
    phi_s = phi[state_idx].T
    # print(phi_s.shape, theta)
    action_values = softmax_sigma*np.dot(phi_s, theta)
    policy = np.exp(action_values)
    policy /= sum(policy)
    return policy
    

def runEpisode(theta, phi):
    curr_state = random.choices(initial_states)[0] #[row, col]
    state_list, action_list, reward_list = [], [], []
    timestep = 0
    while curr_state not in terminal_states:# and timestep < 10000:
        state_idx = (max_r+1)*curr_state[0] + curr_state[1]
        action_idx = np.argmax(softmax_select(theta, phi, state_idx))
        curr_action = list(action_set.keys())[action_idx] #\in {Au, AL, AR, AD}
        
        possible_next_states = transition_function(curr_state[0], curr_state[1], curr_action) #next states with prob
        next_state = random.choices([[s[0], s[1]] for s in possible_next_states], [s[2] for s in possible_next_states])[0] #sample according to prob
        curr_reward = reward(next_state)
        # print(curr_state, curr_action, curr_reward, next_state)
        state_list.append(curr_state)
        action_list.append(curr_action)
        #append discounted reward
        reward_list.append(curr_reward)
        curr_state = next_state
        timestep += 1

    return state_list, action_list, reward_list

# for r in range(max_r+1):
#     for c in range(max_c+1):
#         index = (max_r+1)*r+c

def REINFORCE(gamma, alpha_w, alpha_theta, algo_type = 'without_baseline'):
    total_states = (max_r+1)*(max_c+1)# - len(obstacles)
    theta = np.random.normal(0, 0.1, (total_states, len(action_set))) #np.ones((total_states, len(action_set)))*0.01 #
    # momentum, vel = np.zeros(theta.shape), np.zeros(theta.shape)
    # beta1, beta2 = 0.9, 0.999
    val_w = np.zeros((total_states))
    phi = np.zeros((total_states, total_states))
    np.fill_diagonal(phi, 1.0)
    epsiodes_graph = []
    threshold = 0.1
    
    for iter in range(2000):


        state_list, action_list, reward_list = runEpisode(theta, phi)
        T = len(reward_list)
        print("\n EPISODE LENGTH: ",len(reward_list), "CURR ITER: ", iter)

        #create list of n-steps returns
        return_list = np.zeros(T)
        return_list[-1] = reward_list[-1]
        for t in range(T-2, -1, -1):
            return_list[t] = reward_list[t] + gamma*return_list[t+1]
        epsiodes_graph.append(T)
        T_range = np.arange(T)
        
        theta_temp = theta.copy()

        #loop through epsiode for parameter update
        for t in T_range:
            
            curr_state, curr_action = state_list[t], action_list[t]
            state_idx = (max_r+1)*curr_state[0] + curr_state[1] #row_idx*5+col_idx
            action_idx = action_set[curr_action]
            phi_s = phi[state_idx]
            if algo_type == 'without_baseline':
                delta = return_list[t]
            elif algo_type == 'with_baseline':
                delta = return_list[t] - np.dot(phi_s, val_w)
            #w update
            #theta update
            
            policy = softmax_select(theta, phi, state_idx) #returns an array of size 4
            for i in range(len(action_set)):
                if action_idx == i:
                    theta[:, i] += alpha_theta*delta*(1-policy[action_idx])*phi_s
                else:
                    theta[:, i] += alpha_theta*delta*(-1*policy[action_idx])*phi_s
            print(action_idx, delta, policy)
            
            if algo_type == 'with_baseline':
                val_w += alpha_w*delta*phi_s
            #adam update
            # momentum = beta1*momentum + (1-beta1)*delta*delta_ln_pi
            # vel = beta2*vel + (1-beta2)*(delta*delta_ln_pi)**2
            # delta_theta = alpha_theta*momentum/(1-beta1)/(np.sqrt(vel/(1 - beta2)) + 10**(-8))

        #relative change in theta
        max_diff = np.max(np.abs(theta_temp - theta))
        print("max_diff ", max_diff)
        if  max_diff/alpha_theta < threshold:
            break


            
    #final policy
    greedy_policy = [[None for i in range(max_c+1)] for j in range(max_r+1)]
    for i in range(max_r+1):
        for j in range(max_c+1):
            if [i,j] in terminal_states:
                greedy_policy[i][j] = 'G'
                continue
            if [i,j] in obstacles:
                continue
            state_idx = (max_r+1)*i + j
            policy = softmax_select(theta, phi, state_idx)
            action_idx = np.argmax(policy)
            greedy_policy[i][j] = list(action_set.keys())[action_idx]

    # print(greedy_policy)
    print("final policy ")
    for r in greedy_policy:
        s = ''
        for dir in r:
            if dir in print_arrow:
                s += print_arrow[dir] +'  '
            else:
                s += '   '
        print(s)

    plt.figure()
    plt.plot(np.arange(len(epsiodes_graph)), epsiodes_graph)
    plt.xlabel('Iterations')
    plt.ylabel('#Episodes')
    # plt.ylim([-100, 1000])
    plt.savefig('graph_gridworld_reinforce_'+str(algo_type))


alpha_w, alpha_theta = 0.0001, 1e-8
REINFORCE(0.9, alpha_w, alpha_theta, 'without_baseline')

