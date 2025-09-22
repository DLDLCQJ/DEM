import timm
import torch
import torch.nn as nn
import torchvision.models as tvmodels

from modules.attention import CrossAttention
from modules.extractor import FeatureExtractor, linear_block


class Loading_pretrained(nn.Module):
    def __init__(self, network, num_classes, hidden_size,input_size=[64,3,128,128],pretrained=True):
        super(Loading_pretrained, self).__init__()
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
        ##
        self.cross_attn1 = CrossAttention(256,256)
        self.cross_attn2 = CrossAttention(256,256)
        ## Freeze pretrained layers
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
        ##
        if network in ['VGG']:
            self.features = self.model.features
            self.avgpool = self.model.avgpool
            in_features = self.model.get_classifier().in_features
            self.extractor1 = FeatureExtractor(in_features)
            self.extractor2 = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
            self.cls = linear_block(256, num_classes,hidden_size,nn.SELU)
        elif network in ['Alexnet','Mobilenet']: #The classification part consists of multiple fully connected layers
            self.features = self.model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
            in_features = self.avgpool(self.features(torch.rand(*self.input_size))).shape[1]
            self.extractor1 = FeatureExtractor(in_features)
            self.extractor2 = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
            self.cls = linear_block(256, num_classes)
        elif network == 'Densenet': #The classification part consists of multiple fully connected layers
            self.features = self.model.features
            in_features = self.model.classifier.in_features
            self.extractor1 = FeatureExtractor(in_features)
            self.extractor2 = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
            self.cls = linear_block(256, num_classes,hidden_size,nn.SELU)
        elif network == 'Googlenet':
            self.features = nn.Sequential(*list(self.model.children())[:-1])
            in_features = self.model.fc.in_features
            self.extractor1 = FeatureExtractor(in_features)
            self.extractor2 = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
            self.cls = linear_block(256, num_classes,hidden_size,nn.SELU)
        else:
            self.features = nn.Sequential(*list(self.model.children())[:-1])
            in_features = self.model.get_classifier().in_features
            self.extractor1 = FeatureExtractor(in_features)
            self.extractor2 = FeatureExtractor(in_features)
            self.model.fc = nn.Identity()  # Identity layer to skip the head layer
            self.cls = linear_block(256, num_classes,hidden_size,nn.SELU)

    # Modify the classification head
    def forward(self, x1, x2=None,mode='train'):
        if self.network in ['Alexnet','Mobilenet']:
            src_inputs = self.avgpool(self.features(x1)).squeeze()
            tgt_inputs = self.avgpool(self.features(x2)).squeeze()
        else:
            src_inputs = self.features(x1)
            tgt_inputs = self.features(x2)
        #!!!!
        src_feats_adv = self.extractor1(src_inputs)
        #!!!!
        src_feats_con = self.extractor1(src_inputs)
        # #!!!!
        tgt_feats_con = self.extractor1(tgt_inputs)
        #!!!!
        tgt_feats_adv = self.extractor1(tgt_inputs)
        
        src_attn_feats = self.cross_attn1(src_feats_con,src_feats_adv)
        ts_attn_feats = self.cross_attn2(tgt_feats_con,src_feats_adv)

        combined_feats = torch.cat((src_attn_feats,ts_attn_feats),dim=0)
        preds = self.cls(combined_feats)
        return preds,src_feats_con,tgt_feats_con,src_feats_adv, tgt_feats_adv

