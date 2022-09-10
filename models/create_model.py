from .vit import ViT
from .pit import PiT

def create_model(img_size, n_classes, args):
    if args.model == 'vit':
        patch_size = 4 if img_size == 32 else 8
        model = ViT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=192, 
                    mlp_dim_ratio=2, depth=9, heads=12, dim_head=192//12,
                    stochastic_depth=args.sd, suppress = args.suppress, thres = args.thres)
    elif args.model == 'pit':
        patch_size = 2 if img_size == 32 else 4    
        args.channel = 96
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        dim_head = args.channel // args.heads[0]
        
        model = PiT(img_size=img_size, patch_size = patch_size, num_classes=n_classes, dim=args.channel, 
                    mlp_dim_ratio=2, depth=args.depth, heads=args.heads, dim_head=dim_head, 
                    stochastic_depth=args.sd, suppress = args.suppress, thres = args.thres)

       
    return model
