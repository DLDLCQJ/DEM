import timm
import torch
import torch.nn as nn
import torchvision.models as tvmodels
from torch.autograd import Function
import torch.nn.functional as F

from modules.attention import SelfAttention, CrossAttention,SelfAttention_cbp,CrossAttention_cbp
from modules.extractor import FeatureExtractor,linear_block, FeatureExtractor_cbp,linear_block_cbp

class GradientReverseLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod 
    def backward(ctx, grad):
        output = grad.neg() * ctx.alpha
        return output, None
 
class CRL_Extracted_feats(nn.Module):
    def __init__(self, network, replacement_rate,init,maturity_threshold,input_size=[64,3,128,128],pretrained=True):
        super(CRL_Extracted_feats, self).__init__()
        self.network = network
        self.input_size = input_size
        if network == 'VGG':
            self.model = timm.create_model('vgg19_bn', pretrained=pretrained)
        elif network == 'Resnet':
            self.model = timm.create_model('resnet50d', pretrained=pretrained)
        elif network == 'Densenet':
            self.model = timm.create_model('densenet169', pretrained=pretrained)
        elif network == 'Inception':
            self.model = timm.create_model('inception_v3', pretrained=pretrained)
        elif network == 'Alexnet':
            self.model = tvmodels.alexnet(pretrained=pretrained, progress = False)
        elif network == 'Googlenet':
            self.model = tvmodels.googlenet(pretrained= pretrained, progress= False)
        elif network == 'Mobilenet':
            self.model = tvmodels.mobilenet_v2(pretrained=pretrained, progress = False)
         ## Freeze layers
        if pretrained: 
            for param in self.model.parameters():
                param.requires_grad = False
        ##
        # self.self_attn1 = SelfAttention(256)
        # self.self_attn2_cbp = SelfAttention_cbp(256,replacement_rate=replacement_rate,init=init,maturity_threshold=maturity_threshold)
       
        if network in ['VGG']:
            self.features = self.model.features
            self.avgpool = self.model.avgpool
            in_features = self.model.get_classifier().in_features
            self.extractor1 = FeatureExtractor(in_features)
            self.extractor2 = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
           
        elif network in ['Alexnet','Mobilenet']: #The classification part consists of multiple fully connected layers
            self.features = self.model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            in_features = self.avgpool(self.features(torch.rand(*self.input_size))).shape[1]
            self.extractor = FeatureExtractor(in_features)
            self.extractor_cbp = FeatureExtractor_cbp(in_features,replacement_rate=replacement_rate,init=init,maturity_threshold=maturity_threshold)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer 
           
        elif network == 'Densenet': #The classification part consists of multiple fully connected layers
            self.features = self.model.features
            in_features = self.model.classifier.in_features 
            self.extractor1 = FeatureExtractor(in_features)
            self.extractor2 = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
           
        elif network == 'Googlenet':
            self.features = nn.Sequential(*list(self.model.children())[:-1])
            in_features = self.model.fc.in_features
            self.extractor1 = FeatureExtractor(in_features) 
            self.extractor2 = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
      
        else:
            self.features = nn.Sequential(*list(self.model.children())[:-1])
            in_features = self.model.get_classifier().in_features
            self.extractor1 = FeatureExtractor(in_features)
            self.extractor2 = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
            
    def extract_feats(self, x1, x2):
        b1 = x1.size(0)
        b2 = x2.size(0)
        if self.network in ['Alexnet','Mobilenet']:
            x1_inputs = self.avgpool(self.features(x1)).reshape(b1,-1)
            x2_inputs = self.avgpool(self.features(x2)).reshape(b2,-1)
        else:
            x1_inputs = self.features(x1)
            x2_inputs = self.features(x2) 

        x1_feats_adv = self.extractor(x1_inputs)
        x2_feats_adv,features1 = self.extractor_cbp(x2_inputs)
        combined_feats = torch.cat((x1_feats_adv,x2_feats_adv),dim=0)
        return combined_feats,x1_feats_adv,x2_feats_adv,features1

class CRL_Meta_extractor(CRL_Extracted_feats):
    def __init__(self, network, num_classes, replacement_rate,init,maturity_threshold,input_size=[64,3,128,128],pretrained=True,frozen=True):
        super(CRL_Meta_extractor, self).__init__(network,replacement_rate,init,maturity_threshold,input_size,pretrained)
        self.domain_clf = linear_block(256, num_classes)
        self.clf = linear_block(256, num_classes)
        self.clf_cbp = linear_block_cbp(256, num_classes,replacement_rate=replacement_rate,init=init,maturity_threshold=maturity_threshold)
        #self.rl = linear_block(256, num_classes)
        if pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
            if hasattr(self, 'avgpool'):
                for param in self.avgpool.parameters():
                    param.requires_grad = False
        if frozen:
            for name, module in self.extractor.named_modules():
                if not isinstance(module, (nn.ReLU, nn.SELU)):  # Check for activation functions
                    for param in module.parameters():
                        param.requires_grad = False
            
            for name, module in self.domain_clf.named_modules():
                if not isinstance(module, (nn.ReLU, nn.SELU)):
                    for param in module.parameters():
                        param.requires_grad = False

            for name, module in self.clf.named_modules():
                if not isinstance(module, (nn.ReLU, nn.SELU)):
                    for param in module.parameters():
                        param.requires_grad = False
       
    def forward(self, x1, x2,alpha=None): 
        combined_feats,x1_feats_adv,x2_feats_adv,features1 = super().extract_feats(x1,x2)
        reverse_feats = GradientReverseLayer.apply(combined_feats, alpha)
        domains = self.domain_clf(reverse_feats)
        src_preds = self.clf(x1_feats_adv)
        tgt_preds,features2 = self.clf_cbp(x2_feats_adv)
        #probs = self.rl(x2_feats_adv)
        return src_preds,tgt_preds,domains,x1_feats_adv,x2_feats_adv,features1,features2
    
class RL_model(nn.Module):
    def __init__(self, num_classes):
        super(RL_model, self).__init__()
        self.rl = linear_block(256, num_classes)
    def forward(self, x):
        probs = self.rl(x)
        return probs

class RL_Extracted_feats(nn.Module):
    def __init__(self, network, replacement_rate,init,maturity_threshold,input_size=[64,3,128,128],pretrained=True):
        super(RL_Extracted_feats, self).__init__()
        self.network = network
        self.input_size = input_size
        if network == 'VGG':
            self.model = timm.create_model('vgg19_bn', pretrained=pretrained)
        elif network == 'Resnet':
            self.model = timm.create_model('resnet50d', pretrained=pretrained)
        elif network == 'Densenet':
            self.model = timm.create_model('densenet169', pretrained=pretrained)
        elif network == 'Inception':
            self.model = timm.create_model('inception_v3', pretrained=pretrained)
        elif network == 'Alexnet':
            self.model = tvmodels.alexnet(pretrained=pretrained, progress = False)
        elif network == 'Googlenet':
            self.model = tvmodels.googlenet(pretrained= pretrained, progress= False)
        elif network == 'Mobilenet':
            self.model = tvmodels.mobilenet_v2(pretrained=pretrained, progress = False)
         ## Freeze layers
        if pretrained: 
            for param in self.model.parameters():
                param.requires_grad = False
        ##
        #self.self_attn = SelfAttention(256)
       
        if network in ['VGG']:
            self.features = self.model.features
            self.avgpool = self.model.avgpool
            in_features = self.model.get_classifier().in_features
            self.extractor = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
           
        elif network in ['Alexnet','Mobilenet']: #The classification part consists of multiple fully connected layers
            self.features = self.model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            in_features = self.avgpool(self.features(torch.rand(*self.input_size))).shape[1]
            self.extractor_cbp = FeatureExtractor_cbp(in_features,replacement_rate=replacement_rate,init=init,maturity_threshold=maturity_threshold)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer 
           
        elif network == 'Densenet': #The classification part consists of multiple fully connected layers
            self.features = self.model.features
            in_features = self.model.classifier.in_features 
            self.extractor = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
           
        elif network == 'Googlenet':
            self.features = nn.Sequential(*list(self.model.children())[:-1])
            in_features = self.model.fc.in_features
            self.extractor = FeatureExtractor(in_features) 
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
      
        else:
            self.features = nn.Sequential(*list(self.model.children())[:-1])
            in_features = self.model.get_classifier().in_features
            self.extractor = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
    def extract_feats(self, x):
        bs = x.size(0)
        if self.network in ['Alexnet','Mobilenet']:
            x_inputs = self.avgpool(self.features(x)).reshape(bs,-1)
        else:
            x_inputs = self.features(x) 
        x1_feats_adv, features1 = self.extractor_cbp(x_inputs)
        return x1_feats_adv, features1

class RL_Meta_extractor(RL_Extracted_feats):
    def __init__(self, network, num_classes,replacement_rate,init,maturity_threshold, input_size=[64,3,128,128],pretrained=True):
        super(RL_Meta_extractor, self).__init__(network,replacement_rate,init,maturity_threshold,input_size,pretrained)
        self.clf_cbp = linear_block_cbp(256, num_classes,replacement_rate=replacement_rate,init=init,maturity_threshold=maturity_threshold)
        self.rl_cbp = linear_block_cbp(256, num_classes,replacement_rate=replacement_rate,init=init,maturity_threshold=maturity_threshold)
        if pretrained:
            for param in self.features.parameters():
                param.requires_grad = False
            if hasattr(self, 'avgpool'):
                for param in self.avgpool.parameters():
                    param.requires_grad = False
       
    def forward(self, x):       
        extracted_feats, features1 = super().extract_feats(x)
        preds, features2 = self.clf_cbp(extracted_feats)
        rewards, features3 = self.rl_cbp(extracted_feats)
        return preds,features1, features2,rewards,features3

# class Extracted_feats(nn.Module):
#     def __init__(self, network, input_size=[64,3,128,128],pretrained=True):
#         super(Extracted_feats, self).__init__()
#         self.network = network
#         self.input_size = input_size
#         if network == 'VGG':
#             self.model = timm.create_model('vgg19_bn', pretrained=pretrained)
#         elif network == 'Resnet':
#             self.model = timm.create_model('resnet50d', pretrained=pretrained)
#         elif network == 'Densenet':
#             self.model = timm.create_model('densenet169', pretrained=pretrained)
#         elif network == 'Inception':
#             self.model = timm.create_model('inception_v3', pretrained=pretrained)
#         elif network == 'Alexnet':
#             self.model = tvmodels.alexnet(pretrained=pretrained, progress = False)
#         elif network == 'Googlenet':
#             self.model = tvmodels.googlenet(pretrained= pretrained, progress= False)
#         elif network == 'Mobilenet':
#             self.model = tvmodels.mobilenet_v2(pretrained=pretrained, progress = False)
#          ## Freeze layers
#         if pretrained: 
#             for param in self.model.parameters():
#                 param.requires_grad = False
#         ##
#         #self.self_attn = SelfAttention(256)
       
#         if network in ['VGG']:
#             self.features = self.model.features
#             self.avgpool = self.model.avgpool
#             in_features = self.model.get_classifier().in_features
#             self.extractor = FeatureExtractor(in_features)
#             self.model.fc = nn.Identity()  # Identity layer to skip the head layer
           
#         elif network in ['Alexnet','Mobilenet']: #The classification part consists of multiple fully connected layers
#             self.features = self.model.features
#             self.avgpool = nn.AdaptiveAvgPool2d((1,1))
#             in_features = self.avgpool(self.features(torch.rand(*self.input_size))).shape[1]
#             self.extractor = FeatureExtractor(in_features)
#             self.model.fc = nn.Identity()  # Identity layer to skip the head layer 
           
#         elif network == 'Densenet': #The classification part consists of multiple fully connected layers
#             self.features = self.model.features
#             in_features = self.model.classifier.in_features 
#             self.extractor = FeatureExtractor(in_features)
#             self.model.fc = nn.Identity()  # Identity layer to skip the head layer
           
#         elif network == 'Googlenet':
#             self.features = nn.Sequential(*list(self.model.children())[:-1])
#             in_features = self.model.fc.in_features
#             self.extractor = FeatureExtractor(in_features) 
#             self.model.fc = nn.Identity()  # Identity layer to skip the head layer
      
#         else:
#             self.features = nn.Sequential(*list(self.model.children())[:-1])
#             in_features = self.model.get_classifier().in_features
#             self.extractor = FeatureExtractor(in_features)
#             self.model.fc = nn.Identity()  # Identity layer to skip the head layer
#     def extract_feats(self, x):
#         bs = x.size(0)
#         if self.network in ['Alexnet','Mobilenet']:
#             x_inputs = self.avgpool(self.features(x)).reshape(bs,-1)
#         else:
#             x_inputs = self.features(x) 
#         x1_feats_adv = self.extractor(x_inputs)
#         return x1_feats_adv

# class Meta_extractor(Extracted_feats):
#     def __init__(self, network, num_classes, input_size=[64,3,128,128],pretrained=True):
#         super(Meta_extractor, self).__init__(network, input_size,pretrained)
#         self.clf = linear_block(256, num_classes)
#         if pretrained:
#             for param in self.features.parameters():
#                 param.requires_grad = False
#             if hasattr(self, 'avgpool'):
#                 for param in self.avgpool.parameters():
#                     param.requires_grad = False
       
#     def forward(self, x):       
#         extracted_feats = super().extract_feats(x)
#         preds = self.clf(extracted_feats)
#         return preds
