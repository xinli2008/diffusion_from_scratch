import torch
import numpy as np

class DDPMScheduler:
    def __init__(self, beta_schedule, noise_steps, beta_start = 0, beta_end = 0.02, cosine_s = 8e-3, noise_offset = 0, device = None):
        r"""
        Classic DDPM with Gaussian diffusion, in image space
        """
        self.beta_schedule = beta_schedule
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.noise_steps = noise_steps
        self.noise_offset = noise_offset
        self.device = device
        self.cosine_s = cosine_s

        if self.beta_schedule == "linear":
            betas = torch.linspace(start = beta_start, end = beta_end, steps = noise_steps)
        elif self.beta_schedule == "cosine":
            timesteps = (torch.arange(noise_steps + 1, dtype=torch.float64) / noise_steps + cosine_s)
            alphas = timesteps / (1 + cosine_s) * np.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = np.clip(betas, a_min=0, a_max=0.999)
        elif self.beta_schedule == "sqrt_linear":
            betas = torch.linspace(beta_start, beta_end, noise_steps, dtype=torch.float64) ** 0.5
        else:
            raise ValueError(f"unexcepted beta schedule for {beta_schedule}")

        self.alpha = 1 - betas
        self.alpha_cum_product = torch.cumprod(self.alpha, dim = 0)
        self.sqrt_alpha_cum_product = torch.sqrt(self.alpha_cum_product)
        self.sqrt_one_minus_alpha_cum_product = torch.sqrt(1 - self.alpha_cum_product)
        self.alpha_cum_product_prev = torch.cat((torch.tensor([1.0]), self.alpha_cum_product[:-1]), dim = -1)
        self.variance = (1 - self.alpha) * (1 - self.alpha_cum_product_prev) / (1 - self.alpha_cum_product)
        
    
    def q_sample(self, batch_x, batch_t):
        r"""
        Samples a noisy version of the input data at a given timestep.

        Args:
            batch_x (torch.Tensor): Input data tensor of shape (batch_size, channels, height, width).
            batch_t (torch.Tensor): Timestep indices for each sample in the batch, of shape (batch_size,).

        Returns:
            tuple: A tuple containing:
                - x_noisy (torch.Tensor): Noisy version of the input data.
                - noise (torch.Tensor): The random noise added to the input data.
        """
        device = batch_x.device
        sqrt_alpha_cum_product = self.sqrt_alpha_cum_product.to(device)[batch_t].view(batch_x.shape[0], 1, 1, 1)
        sqrt_one_minus_alpha_cum_product = self.sqrt_one_minus_alpha_cum_product.to(device)[batch_t].view(batch_x.shape[0], 1, 1, 1)
        # Sample noise that we'll add to the image
        noise = torch.randn_like(batch_x)
        x_noisy = sqrt_alpha_cum_product * batch_x + sqrt_one_minus_alpha_cum_product * noise
        return x_noisy, noise
    
    def denoise_sample(self, denoising_model, batch_xt, batch_cls):
        r"""
        Denoise a batch of samples using the provided denoising model.
        """
        NUM_TIMESTEPS = 1000
        denoised_images = [batch_xt]

        with torch.no_grad():
            for t in range(NUM_TIMESTEPS - 1, -1, -1):
                current_timestep = torch.full((batch_xt.size(0)), t)
                shape = (batch_xt.size(0), 1, 1, 1)

                # predict noise for the current timestep
                predicted_noise = denoising_model(batch_xt, current_timestep, batch_cls)

                # Generate random noise for adding variability
                random_noise = torch.rand_like(batch_xt)
                
                batch_mean_t = 1/ self.sqrt_alpha_cum_product[current_timestep].view(*shape) * \
                                [
                                    batch_xt - ((1 - self.alpha[current_timestep].view(*shape)) / self.sqrt_one_minus_alpha_cum_product[current_timestep].view(*shape) * predicted_noise)
                                ] 
                if t != 0:
                    batch_xt = batch_mean_t + random_noise * torch.sqrt(self.variance[current_timestep].view(*shape))
                denoised_images.append(batch_xt)
            
            return denoised_images