import numpy as np
import gym
import cv2

from vae import *

img_shape = (64,64,1)
batch_size = 64
latent_dim = 16

def log_run(num_it=1000):
	env = gym.make('Pong-v0')
	frames = []
	obs = env.reset()
	for t in range(num_it):
		env.render()
		act = env.action_space.sample()
		obs = cv2.cvtColor(obs,cv2.COLOR_BGR2GRAY)
		obs = cv2.resize(obs,img_shape[:-1]).reshape(img_shape)
		frames.append(obs/255.0) # [obs, act]
		obs, rwd, done, _ = env.step(act)
		if done:
			break
	env.close()
	return np.array(frames)

np.random.seed(237)
frames = log_run()
num_sample = len(frames)

model = VariantionalAutoencoder([None,64,64,1], 1e-3, batch_size, latent_dim)
for epoch in range(11):
    for iter in range(num_sample // batch_size):
        # Obtina a batch
        inds = np.random.choice(range(frames.shape[0]), batch_size, False)
        batch = np.array([frames[i] for i in inds])
        # Execute the forward and the backward pass and report computed losses
        loss, recon_loss, latent_loss = model.run_single_step(batch)
    if epoch % 2 == 0:
        print('[Epoch {}] Loss: {}, Recon loss: {}, Latent loss: {}'.format(
            epoch, loss, recon_loss, latent_loss))
    if epoch % 10 == 9:
        cv2.imshow('Image window',model.reconstructor([batch[0]])[0])
        cv2.waitKey()
print('Done!')

