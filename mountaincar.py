import gym

env = gym.make('CartPole-v1') 
observation = env.reset()

# action-space & observation-space describes what is the valid format of action & state parameters for that particular env to work on with
print("Action space ", env.action_space) #Discerete(3) -> 0,1,2 
#disceret class; #var in class n, start; #functions sample(), contains(x)
print("State space ",env.observation_space) #[Output: ] Box(2,)
#box class; low, high, shape of state
for t in range(50):
        # env.render()        
        action = env.action_space.sample()
        print(observation, action)
        observation, reward, done, info = env.step(action) 
        #observation represents environments next state
        #reward a float of reward in previous action
        #done when it’s time to reset the environment or goal achieved
        #info a dict for debugging, it can be used for learning if it contains raw probabilities of environment’s last state
        print(observation, reward, done, info)
        if done:
            print("Finished after {} timesteps".format(t+1))
            break
