import torch
from tqdm import tqdm

from opensora.registry import SCHEDULERS

from .rectified_flow import RFlowScheduler, timestep_transform
import os 
import pandas as pd
import numpy as np
@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=True,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform
        self.prev_vel = None
        self.best_cos = 0
        self.cos_tolerance = 1e-3
        
        self.tolerance_n = 3
        self.max_tolerance_n = 3

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )
        self.minimum_steps = 14

    def sample(
        self,
        model,
        text_encoder, # encoded_text
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
        lpl_setting=None
    ):
        
        # save z
        import pickle
        save_path = "{SAVE_PATH}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # pickle.dump
        with open(f"{save_path}/z.pkl", 'wb') as f:
            pickle.dump(z.detach().to('cpu'), f)
        
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        # save y
        with open(f'{save_path}/y.pkl', 'wb') as f:
            pickle.dump(model_args["y"].detach().to('cpu'), f)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]
    
        # save timesteps into pandas csv file
        csv_timesteps = pd.DataFrame([x.item() for x in timesteps])
        # prepare timesteps
        
        custom_N = lpl_setting if lpl_setting is not None else 2
        additional_steps = 0
        l_touch = 0
        
        cust_timesteps = self.get_dynamic_noise_schedule(custom_N, z, device, additional_args)
        leap_it = (len(timesteps)//(custom_N)) * (custom_N-1)
        # save cust_timesteps into pandas csv file
        csv_cust_timesteps = pd.DataFrame([x.item() for x in cust_timesteps])

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)
        
        progress_wrap = tqdm if progress else (lambda x: x)
        k = 1 if custom_N ==4 else 0
        for i, t in (enumerate(timesteps)):
            if i==leap_it+k and additional_steps == 1:
                print("Temporal Leap: ", i)
                if mask is not None:
                    mask_t = mask * self.num_timesteps
                    x0 = z.clone()
                    x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), cust_timesteps[-1])
                    mask_t_upper = mask_t >= cust_timesteps[-1].unsqueeze(1)
                    model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                    mask_add_noise = mask_t_upper & ~noise_added
                    z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                    noise_added = mask_t_upper
                z_in = torch.cat([z, z], 0)
                cust_t = torch.cat([cust_timesteps[-1], cust_timesteps[-1]], 0)
                pred_cond, pred_uncond = pred.chunk(2, dim=0)

                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                dt = timesteps[i]
                dt = dt / 1000
                z = z + v_pred * dt[:, None, None, None, None]
                if mask is not None:
                    z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
                return z
            elif i==leap_it and additional_steps > 1 and l_touch == 0:
                for j in tqdm(range(additional_steps, 0, -1),desc=f"Temporal Leap, {additional_steps} steps"):
                    if mask is not None:
                        mask_t = mask * self.num_timesteps
                        x0 = z.clone()
                        x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), cust_timesteps[-j])
                        mask_t_upper = mask_t >= cust_timesteps[-j].unsqueeze(1)
                        model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                        mask_add_noise = mask_t_upper & ~noise_added
                        z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                        noise_added = mask_t_upper
                    z_in = torch.cat([z, z], 0)
                    cust_t = torch.cat([cust_timesteps[-j], cust_timesteps[-j]], 0)
                    pred = model(z_in, cust_t, **model_args).chunk(2, dim=1)[0]
                    pred_cond, pred_uncond = pred.chunk(2, dim=0)
                    v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                    dt = cust_timesteps[-j] - cust_timesteps[-j+1] if j < additional_steps else cust_timesteps[-j]
                    dt = dt / 1000
                    z = z + v_pred * dt[:, None, None, None, None]
                    if mask is not None:
                        z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
                return z
            elif i==leap_it and additional_steps > 1 and l_touch == 1:
                if mask is not None:
                    mask_t = mask * self.num_timesteps
                    x0 = z.clone()
                    x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), cust_timesteps[-1])
                    mask_t_upper = mask_t >= cust_timesteps[-1].unsqueeze(1)
                    model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                    mask_add_noise = mask_t_upper & ~noise_added
                    z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                    noise_added = mask_t_upper
                z_in = torch.cat([z, z], 0)
                cust_t = torch.cat([cust_timesteps[-1], cust_timesteps[-1]], 0)
                pred = model(z_in, cust_t, **model_args).chunk(2, dim=1)[0]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)

                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                dt = cust_timesteps[-1]
                dt = dt / 1000
                z = z + v_pred * dt[:, None, None, None, None]
                if mask is not None:
                    z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
                

                # additional last touch
                if mask is not None:
                    mask_t = mask * self.num_timesteps
                    x0 = z.clone()
                    x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), timesteps[-2])
                    mask_t_upper = mask_t >= timesteps[-2].unsqueeze(1)
                    model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                    mask_add_noise = mask_t_upper & ~noise_added
                    z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                    noise_added = mask_t_upper
                z_in = torch.cat([z, z], 0)
                cust_t = torch.cat([timesteps[-2], timesteps[-2]], 0)
                pred = model(z_in, cust_t, **model_args).chunk(2, dim=1)[0]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)

                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                dt = timesteps[-2] - timesteps[-1]

                dt = dt / 1000
                z = z + v_pred * dt[:, None, None, None, None]
                if mask is not None:
                    z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

                # additional last touch
                if mask is not None:
                    mask_t = mask * self.num_timesteps
                    x0 = z.clone()
                    x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), timesteps[-1])
                    mask_t_upper = mask_t >= timesteps[-1].unsqueeze(1)
                    model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                    mask_add_noise = mask_t_upper & ~noise_added
                    z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                    noise_added = mask_t_upper
                z_in = torch.cat([z, z], 0)
                cust_t = torch.cat([timesteps[-1], timesteps[-1]], 0)
                pred = model(z_in, cust_t, **model_args).chunk(2, dim=1)[0]
                pred_cond, pred_uncond = pred.chunk(2, dim=0)

                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                dt = timesteps[-1]

                dt = dt / 1000
                z = z + v_pred * dt[:, None, None, None, None]
                if mask is not None:
                    z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)


                
                return z


            else:
                # mask for adding noise
                if mask is not None:
                    mask_t = mask * self.num_timesteps
                    x0 = z.clone()
                    x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                    
                    mask_t_upper = mask_t >= t.unsqueeze(1)
                    model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                    mask_add_noise = mask_t_upper & ~noise_added

                    z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                    noise_added = mask_t_upper

                # classifier-free guidance
                z_in = torch.cat([z, z], 0)
                t = torch.cat([t, t], 0)
               
                model_args["step"] = i
                pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]

                pred_cond, pred_uncond = pred.chunk(2, dim=0)
                v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                shoot_flag = self.compute_cosine_simularity(v_pred,i)
                
                # update z
                dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]

                dt = timesteps[i] if shoot_flag else dt

                dt = dt / self.num_timesteps
                z = z + v_pred * dt[:, None, None, None, None]

                if mask is not None:
                    z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)
                
                if shoot_flag:
                    return z
        return z

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)

    def get_dynamic_noise_schedule(self, n, z, device,additional_args=None):
        custom_steps = n
        cust_timesteps = [(1.0 - i / custom_steps) * self.num_timesteps for i in range(custom_steps)]
        if self.use_discrete_timesteps:
            cust_timesteps = [int(round(t)) for t in cust_timesteps]
        cust_timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in cust_timesteps]
        if self.use_timestep_transform:
            cust_timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in cust_timesteps]

        return cust_timesteps

    
    def compute_f_metrics(self, pred):
        ''' Compute median frequency'''
        # rearrange
        pred = pred.cpu().permute(0,2,1,3,4)
        # only first frame
        pred = pred[:,0,:,:,:]
        # fft

        freq_val = torch.fft.fftn(pred,dim=[0,1,2,3])
        # save freq_val into pandas csv file
        
        freq_val = freq_val.detach().cpu().numpy()
        freq_val = np.abs(freq_val)
        freq_val = freq_val.reshape(-1, freq_val.shape[-1])
        mean_freq_val = np.mean(freq_val)
        freq_val = pd.DataFrame(freq_val)
        save_path = "{SAVE_PATH}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        freq_val.to_csv(f"{save_path}/freq_val.csv", index=False, header=False, mode='a')

        return mean_freq_val
    
    def compute_cosine_simularity(self,pred_v,i):
        shoot_flag = False
        # compute cosine simularity between current and previous velocity
        if self.prev_vel is not None:
            # (0,0,0,d,e) -> (d,e)
            pred_v_val = pred_v[0,0,0,:,:]
            prev_vel = self.prev_vel[0,0,0,:,:]

            
            if i >= self.minimum_steps:
                cos = torch.nn.functional.cosine_similarity(pred_v_val, prev_vel,dim=1)
                print(f"at step {i}, cosine similarity: ", cos.mean().item(),self.tolerance_n)
                if cos.mean() > self.best_cos + self.cos_tolerance:
                    self.best_cos = cos.mean()
                    self.tolerance_n = self.max_tolerance_n
                    shoot_flag  = False
                else:
                    self.tolerance_n -= 1


            if self.tolerance_n == 0:
                self.tolerance_n = self.max_tolerance_n
                self.best_cos = 0
                shoot_flag = True
                # save i to csv
                save_path = "{SAVE_PATH}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(f"{save_path}/shoot_flag.csv", 'a') as f:
                    f.write(f"{i}\n")
                    


        self.prev_vel = pred_v
        return shoot_flag
        
        