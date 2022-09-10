from math import sqrt
from utils.drop_path import DropPath
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))

# helpers

def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

# classes

class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        
        return self.fn(self.norm(x), **kwargs)
    

class FeedForward(nn.Module):
    def __init__(self, num_tokens, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
  
        
    def forward(self, x):
        return self.net(x)    
    
class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads = 8, dim_head = 64, dropout = 0., suppress = 0., thres = 1.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.num_patches = num_patches
        
        self.thres = thres
        self.suppress_status = True if suppress != 0 else False
        if self.suppress_status:
            self.suppress = nn.Parameter(torch.tensor(suppress))

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        init_weights(self.to_qkv)        
 

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
      
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        attn = self.attend(dots)

        if self.suppress_status:
            row_max = torch.amax(attn, dim = -1).unsqueeze(-1).expand(attn.shape)
            row_mask = torch.le(attn, row_max*self.thres)
            row_mask_after = torch.mul(attn, row_mask)
            row_sum_thres = torch.sum(row_mask_after, dim = -1).unsqueeze(-1).expand(attn.shape)
            ele_coefs = self.suppress * torch.div(attn, row_sum_thres+0.000001)
            attn = ~row_mask*attn+row_mask*ele_coefs*attn

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)
    

class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout = 0., stochastic_depth=0.,
                     suppress = 0, thres = 1.):
        super().__init__()
        self.layers = nn.ModuleList([])
        num_patches = num_patches**2 + 1
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim, Attention(dim, num_patches, heads = heads, dim_head = dim_head, 
                dropout = dropout,  suppress = suppress, thres = thres)),
                PreNorm(num_patches, dim, FeedForward(num_patches, dim, mlp_dim_ratio, dropout = dropout))
            ]))            
            
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()
    def forward(self, x):
        for (attn, ff) in self.layers:       
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
            
        return x
    
# depthwise convolution, for pooling

def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
class DepthWiseConv2d(nn.Module):
    def __init__(self, img_size, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.out_size = img_size // 2
        
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_out, dim_out, kernel_size = 1, bias = bias)
        ) 
                
        
    def forward(self, x):
        return self.net(x)
    
    
# pooling layer

class Pool(nn.Module):
    def __init__(self, img_size, dim):
        super().__init__()
        self.downsample = DepthWiseConv2d(img_size, dim, dim * 2, kernel_size = 3, stride = 2, padding = 1)
        self.cls_ff = nn.Linear(dim, dim * 2)
        self.dim = dim
        

    def forward(self, x):
        cls_token, tokens = x[:, :1], x[:, 1:]

        cls_token = self.cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b c h w', h = int(sqrt(tokens.shape[1])))
        tokens = self.downsample(tokens)
        tokens = rearrange(tokens, 'b c h w -> b (h w) c')

        return torch.cat((cls_token, tokens), dim = 1)
    
# main class

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PiT(nn.Module):
    def __init__(self, *, img_size, patch_size, num_classes, dim, depth, heads, mlp_dim_ratio, dim_head = 64, dropout = 0., 
                 emb_dropout = 0., stochastic_depth=0., suppress = 0, thres = 1.):
        super(PiT, self).__init__()
        heads = cast_tuple(heads, len(depth))
        self.num_classes = num_classes

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(3, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c')
        )
            
            
        output_size = img_size // patch_size
        num_patches = output_size ** 2   
        

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([])

        for ind, (layer_depth, layer_heads) in enumerate(zip(depth, heads)):
            not_last = ind < (len(depth) - 1)
            
            self.layers.append(Transformer(dim, output_size, layer_depth, layer_heads,
                                           dim_head, dim*mlp_dim_ratio, dropout, stochastic_depth, 
                                            suppress = suppress, thres = thres))

            if not_last:
                self.layers.append(Pool(output_size, dim))
                output_size = conv_output_size(output_size, 3, 2, 1)
                
                dim *= 2
       
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.apply(init_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)     
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        for i, layer in enumerate(self.layers):  
            x = layer(x)

        return self.mlp_head(x[:, 0])
    
