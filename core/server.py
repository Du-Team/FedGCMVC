import torch
from modules.federated_loss import MultiViewSwAVLoss

class Server(torch.nn.Module):
    def __init__(self, args, device="cpu"):
        super().__init__()
        self.device = device
        self.criterion = MultiViewSwAVLoss(
            feature_dim=args.feature_dim,
            class_num=args.class_num,
            temperature=args.instance_temperature,
            epsilon=args.epsilon if hasattr(args, 'epsilon') else 0.1,
        )
        self.criterion.to(self.device)

    def step(self, z_list):
        loss = self.criterion(z_list)
        
        for z in z_list:
            if z.grad is not None: z.grad.zero_()

        loss.backward()
        
        grads_list = [z.grad for z in z_list]
        
        return loss.item(), grads_list