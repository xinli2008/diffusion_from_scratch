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
        
        # NOTE: Sample noise that we'll add to the image
        noise = torch.randn_like(batch_x)

        batch_alpha_cum_product = self.alpha_cum_product.to(device)[batch_t].view(batch_x.shape[0], 1, 1, 1)
        x_noisy = torch.sqrt(batch_alpha_cum_product) * batch_x + torch.sqrt(1 - batch_alpha_cum_product) * noise
        
        return x_noisy, noise
    
    def denoise_sample(self, denoising_model, batch_x_t, batch_cls):
        r"""
        Denoise a batch of samples using the provided denoising model.
        """
        NUM_TIMESTEPS = 1000
        denoised_images = [batch_x_t,]
        with torch.no_grad():
            for t in range(NUM_TIMESTEPS - 1, -1, -1):
                current_timestep = torch.full((batch_x_t.size(0),), t).to(batch_x_t.device)
                shape = (batch_x_t.size(0), 1, 1, 1)

                # predict noise for the current timestep
                predicted_noise = denoising_model(batch_x_t, current_timestep, batch_cls)

                # Generate random noise for adding variability
                random_noise = torch.randn_like(batch_x_t)
                
                batch_mean_t = (1/torch.sqrt(self.alpha.to(batch_x_t.device)[current_timestep].view(*shape))) * \
                                (
                                    batch_x_t - ((1 - self.alpha.to(batch_x_t.device)[current_timestep].view(*shape)) / torch.sqrt(1 - self.alpha_cum_product.to(batch_x_t.device)[current_timestep].view(*shape)) * predicted_noise)
                                ) 
                if t != 0:
                    batch_x_t = batch_mean_t + random_noise * torch.sqrt(self.variance.to(batch_x_t.device)[current_timestep].view(*shape))
                else:
                    batch_x_t = batch_mean_t

                denoised_images.append(batch_x_t)

        return denoised_images