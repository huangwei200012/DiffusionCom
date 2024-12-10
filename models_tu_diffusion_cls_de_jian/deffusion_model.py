import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
# 定义 beta 调度函数
def linear_beta_schedule(timesteps, beta_min=0.0001, beta_max=0.02):
    return np.linspace(beta_min, beta_max, timesteps)

def quadratic_beta_schedule(timesteps, beta_min=0.0001, beta_max=0.02):
    return beta_min + (np.linspace(0, 1, timesteps) ** 2) * (beta_max - beta_min)

# def cosine_beta_schedule(timesteps):
#     return 1 - np.cos(np.linspace(0, np.pi / 2, timesteps))

def cosine_beta_schedule(timesteps, scale=1.0):
    return scale * (1 - np.cos(np.linspace(0, np.pi / 2, timesteps)))
def exponential_beta_schedule(timesteps, beta_min=0.0001, beta_max=0.02):
    return beta_min * np.exp(np.linspace(0, 1, timesteps) * np.log(beta_max / beta_min))

# 设置总的扩散步骤数
timesteps = 1000

# 计算不同调度下的 beta 值
beta_linear = linear_beta_schedule(timesteps)
beta_quadratic = quadratic_beta_schedule(timesteps)
beta_cosine = cosine_beta_schedule(timesteps)
beta_exponential = exponential_beta_schedule(timesteps)
# import pdb;pdb.set_trace()
# Forward Diffusion Process: Add noise progressively
def forward_diffusion_process(fact_embed, timesteps, beta_schedule):
    """
    Add noise progressively to the fact embedding based on the beta schedule.
    :param fact_embed: Embedding of the fact (head, relation, tail)
    :param timesteps: Number of diffusion steps
    :param beta_schedule: Beta values for each timestep
    :return: Noised embedding at each timestep
    """
    # Store the noised embeddings for each time step
    noised_embeddings = []
    
    for t in range(timesteps):
        beta_t = torch.tensor(beta_schedule[t], dtype=torch.float32)
        # Add Gaussian noise to the fact embedding
        noise = torch.randn_like(fact_embed)
        # Forward diffusion equation: q(x_t | x_{t-1}) = sqrt(1 - beta) * x_{t-1} + sqrt(beta) * noise
        # fact_embed = alphas_bar_sqrt[t] * fact_embed + one_minus_alphas_bar_sqrt[t] * noise
        fact_embed = torch.sqrt(1 - beta_t) * fact_embed + torch.sqrt(beta_t) * noise
        noised_embeddings.append(fact_embed)
    
    return noised_embeddings
# Reverse Process to iteratively denoise fact embeddings
def reverse_diffusion_process(denoiser, noisy_fact, head, time_steps):
    for t in range(time_steps - 1, -1, -1):
        # Denoise at each step t
        noise_pred = denoiser(noisy_fact, head,t)
        noisy_fact = noisy_fact - noise_pred
    return noisy_fact


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        # sin and cos embeddings
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class LinearTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(1, embedding_dim)

    def forward(self, timesteps):
        # Add a new dimension for linear embedding
        return self.linear(timesteps.unsqueeze(1).float())
# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, features):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(features)

    def forward(self, x):
        return self.norm(x)

# MLP block
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ScaleWithConditionAddMul(nn.Module):
    def __init__(self, embedding_dim):
        super(ScaleWithConditionAddMul, self).__init__()
        self.scale_factor = nn.Parameter(torch.ones(embedding_dim))
    
    def forward(self, fact_embedding, condition_embedding):
        # 将条件嵌入通过加法或乘法结合到Fact Embedding
        combined_input = fact_embedding + condition_embedding  # 或者用乘法 fact_embedding * condition_embedding
        # 对combined_input应用可学习的缩放
        return combined_input * self.scale_factor
# CFDenoiser Block
class CFDenoiserBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CFDenoiserBlock, self).__init__()
        self.layer_norm1 = LayerNorm(input_dim)
        # self.pointwise_conv = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        # self.layer_norm2 = LayerNorm(input_dim)
        self.mlp = MLP(input_dim,hidden_dim)
        # self.mlp = nn.Linear(input_dim, input_dim)
        self.scale1 = ScaleWithConditionAddMul(input_dim)
        # self.scale2 = ScaleWithConditionAddMul(input_dim)
        sin_emb = SinusoidalTimeEmbedding(768)
        timesteps = torch.arange(0, 1000, dtype=torch.float32)
        # self.fc2 = nn.Linear(768, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        
        self.time_embed = sin_emb(timesteps)  # Timestep embedding
    
    def forward(self, x, condition_embedding, Timestep_embedding):
        # First residual block
        
        condition_embedding = condition_embedding + Timestep_embedding
        condition_embedding = self.fc2(condition_embedding)
        x = x + condition_embedding


        # # Second residual block with MLP
        # residual = x
        # x = self.layer_norm2(x)
        # x = self.pointwise_conv(x.unsqueeze(-1)).squeeze(-1)  # Pointwise conv as 1D conv
        # x = self.scale2(x,condition_embedding)
        # x = x + residual
        return x

# Complete Conditional Fact Denoiser
class CFDenoiser(nn.Module):
    def __init__(self, input_dim, hidden_dim, condition_dim,cfd_block_number,timesteps):
        super(CFDenoiser, self).__init__()
        # 这里可以是condition_dim或者是input_dim，主要是和condition_encoder进行一个对比
        sin_emb = SinusoidalTimeEmbedding(input_dim)
        # in_emb = SinusoidalTimeEmbedding(condition_dim)
        timesteps = torch.arange(0, timesteps, dtype=torch.float32)
        self.time_embedding = sin_emb(timesteps)
        # self.time_embedding = torch.cat((self.time_embedding,self.time_embedding[:,-2:-1]),dim=-1)
        self.condition_encoder = nn.Linear(condition_dim, input_dim)
        # self.cfd_block = CFDenoiserBlock(input_dim, hidden_dim)
        self.cfd_block_number = cfd_block_number
        self.cfd_block_layers = nn.ModuleList([CFDenoiserBlock(input_dim, hidden_dim) for _ in range(cfd_block_number)])
        self.layer_norm = LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, input_dim)  # Final linear layer to predict noise

    def forward(self, x, condition, time_step):
        # Embedding the condition and time step
        # import pdb;pdb.set_trace()
        condition_embedding = self.condition_encoder(condition)
        time_embedding = self.time_embedding[time_step]

        # Denoising process with condition and time embedding
        for idx in range(self.cfd_block_number):
            x = self.cfd_block_layers[idx](x, condition_embedding, time_embedding.to(condition_embedding.device))
        
        # Output noise prediction
        x = self.layer_norm(x)
        predicted_noise = self.linear(x)
        
        return predicted_noise

# # Conditional Fact Denoiser (CFDenoiser)
# class ConditionalFactDenoiser(nn.Module):
#     def __init__(self, embed_dim, hidden_dim):
#         super(ConditionalFactDenoiser, self).__init__()
#         self.condition_encoder = nn.Linear(embed_dim, hidden_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, embed_dim)
#         )
#         sin_emb = SinusoidalTimeEmbedding(14951)
#         timesteps = torch.arange(0, 1000, dtype=torch.float32)
        
#         self.time_embed = sin_emb(timesteps)  # Timestep embedding

#     def forward(self, fact_embed, time_step, h):
#         # Encode conditions
#         condition = self.mlp(h)  # Simple sum as mentioned in paper
#         # Time embedding
#         time_embed = self.time_embed(time_step)
#         # Combine embeddings
#         combined = condition + time_embed
#         denoised_output = self.mlp(combined)
#         return denoised_output
# sin_emb = SinusoidalTimeEmbedding(embedding_dim)
# timesteps = torch.arange(0, 1000, dtype=torch.float32)
# # step信息搞定
# sin_time_embedding = sin_emb(timesteps)

if __name__ == "__main__":
    # sin_emb = SinusoidalTimeEmbedding(14951)
    # timesteps = torch.arange(0, 1000, dtype=torch.float32)
    # timestep = 1000
    # sin_time_embedding = sin_emb(timesteps)

    # sin_emb = SinusoidalTimeEmbedding(14951)
    # timesteps = torch.arange(0, 1000, dtype=torch.float32)
    # time_embedding = sin_emb(timesteps)
    # timestep = 1000
    # noised_embeddings = forward_diffusion_process(tensor,timestep,beta_linear)
    # noised_embedding = noised_embeddings[-1]
    # denoiser = ConditionalFactDenoiser(768,2048)
    # 总的test
    tensor = torch.randn(16, 14951)
    tensor2 = torch.randn(16, 768)
    CFDenoiser = CFDenoiser(1024,2048,1536,1,10)
    
    import pdb;pdb.set_trace()

