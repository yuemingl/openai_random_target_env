import gym
import math
import argparse

from stable_baselines.common.policies import MlpPolicy
from RandomTargetVecEnv import RandomTargetVecEnv
from stable_baselines import PPO2

# configuration
parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train')
parser.add_argument('-w', '--weight', help='trained weight path', type=str, default='')
args = parser.parse_args()
mode = args.mode

if mode == 'train':
	# train the policy using 10 environment
	env = RandomTargetVecEnv(10)

	model = PPO2(
		policy=MlpPolicy,
		policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
		env=env,
	    n_steps=200,
		verbose=1
	)
	model.learn(total_timesteps=300000)
	model.save('model.save')
	print('saved model to model.save')
	print('run using --mode test --weight <model-save-path> to test the model')
else:
	weight_path = args.weight
	if weight_path == "":
		print("Can't find trained weight, please provide a trained weight with --weight switch\n")
	else:
		model = PPO2.load(weight_path)
		print("Loaded weight from {}\n".format(weight_path))

	# test using 1 environment
	env = RandomTargetVecEnv(1)
	obs = env.reset()
	for i in range(200):
	    action, states = model.predict(obs)
	    obs, rewards, dones, info = env.step(action)
	    print('step=',i,' current_pos=(',obs[0][0],',',obs[0][1],') target=(',obs[0][2],',',obs[0][3],')')
