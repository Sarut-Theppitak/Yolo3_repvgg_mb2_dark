from model.yolov3 import Yolov3
import config.yolov3_config_yoloformat as cfg
import torch


# def whole_model_convert(train_model:torch.nn.Module, deploy_model:torch.nn.Module, save_path=None):
#     all_weights = {}
#     train_dict = train_model.state_dict()
#     deploy_dict = deploy_model.state_dict()
#     for name, module in train_model.named_modules():
#         if (hasattr(module, '__backbone')):
#             continue
#         if hasattr(module, 'repvgg_convert'):
#             kernel, bias = module.repvgg_convert()
#             all_weights[name + '.rbr_reparam.weight'] = kernel
#             all_weights[name + '.rbr_reparam.bias'] = bias
#             print('convert RepVGG block')
#         else: 
#             for p_name, p_tensor in module.named_parameters():    # p_name is weight and bias
#                 full_name = name + '.' + p_name
#                 if full_name not in all_weights:
#                     # all_weights[full_name] = p_tensor.detach().cpu().numpy()
#                     all_weights[full_name] = p_tensor.detach()
#             for p_name, p_tensor in module.named_buffers():
#                 full_name = name + '.' + p_name
#                 if full_name not in all_weights:
#                     # all_weights[full_name] = p_tensor.cpu().numpy()
#                     all_weights[full_name] = p_tensor

#     deploy_model.load_state_dict(all_weights, strict=False)
#     if save_path is not None:
#         torch.save(deploy_model.state_dict(), save_path)

#     return deploy_model

# train_state_dict = torch.load('./weight/repA1g4_seed1_20each/best.pt')
# train_model = Yolov3(cfg=cfg, deploy=False)
# train_model.load_state_dict(train_state_dict, strict=False)
# deploy_model = Yolov3(cfg=cfg, deploy=True)


# whole_model_convert(train_model, deploy_model)

# deploy_state_dict = deploy_model.state_dict()
# torch.save(deploy_state_dict, './weight/repA1g4_seed1_20each/best_deploy.pt')


################################# test eqivalent ##################################################

train_state_dict = torch.load('./weight/repA1g4_seed1_20each/best.pt')
train_model = Yolov3(cfg=cfg, deploy=False)
train_model.load_state_dict(train_state_dict, strict=False)
train_model.cuda()

deploy_state_dict = torch.load('./weight/repA1g4_seed1_20each/best_deploy.pt')
deploy_model = Yolov3(cfg=cfg, deploy=True)
deploy_model.load_state_dict(deploy_state_dict, strict=False)
deploy_model.cuda()

train_model.eval()
deploy_model.eval()

x = torch.randn(1, 3, 224, 224).cuda()

with torch.no_grad():
    train_p, train_p_d = train_model(x)
    deploy_p, deploy_p_d = deploy_model(x)
    print(((train_p[0] - deploy_p[0]) ** 2).sum()) 
    print(((train_p[1] - deploy_p[1]) ** 2).sum()) 
    print(((train_p[2] - deploy_p[2]) ** 2).sum()) 
    print(((train_p_d - deploy_p_d) ** 2).sum()) 