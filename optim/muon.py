import torch
import torch.nn as nn  


def zeropower_newton_schulz(x, steps=5, eps=1e-7):  
     original_shape = x.shape 
     
     if x.ndim > 2:  
          x = x.reshape(x.shape[0], -1)
  
     transposed = False  
     if x.shape[0] > x.shape[1]:   
          x = x.T  
          transposed = True  

     x = x / (x.norm() + eps)  

     a, b, c = 3.4445, -4.7750, 2.0315       
     
     for _ in range(steps):  
          xx_t = x @ x.T 
          x = a * x + (b * xx_t + c * xx_t @ xx_t) @  x   

     if transposed:  
          x = x.T  

     return x.reshape(original_shape)     






class Muon(torch.optim.Optimizer):   

     def __init__(self, parameters, lr=1e-3, momentum_beta=0.95, weight_decay=0.0, ns_steps=5):  
          defaults = dict(
               lr= lr, 
               momentum_beta=momentum_beta, 
               weight_decay=weight_decay,
               ns_steps=ns_steps
          ) 
          super().__init__(parameters, defaults)   

     @torch.no_grad() 
     def step(self, closure= None):          
         
         loss = None 

         if closure is not None:  
              with torch.enable_grad(): 
                   loss = closure()
         
         for group in self.param_groups:  
              lr = group["lr"] 
              beta= group["momentum_beta"] 
              weight_decay = group["weight_decay"] 
              ns_steps = group["ns_steps"]   

              for p in group["params"]:  
                   if p.grad is None:  
                        continue  
                   
                   grad = p.grad  

                   if grad.ndim < 2:  
                        continue  
                   
                   state = self.state[p]  

                   if "momentum_buffer" not in state:  
                        state["momentum_buffer"] = torch.zeros_like(grad)  

                   momentum = state["momentum_buffer"] 
                   momentum.mul_(beta).add_(grad)   
                   update = grad + beta * momentum
                   update = zeropower_newton_schulz(update, steps=ns_steps)

                   if weight_decay != 0:  
                        p.mul_(1 - lr * weight_decay)  

                   p.add_(update, alpha=-lr)  
     
         return loss  
     

