import torch
import torch.nn as nn

from modules.cbp_linear import CBPLinear

# class FeatureExtractor(nn.Module):
#     def __init__(self,input_size,output_size=256):
#         super(FeatureExtractor, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_size, 768),
#             nn.LayerNorm(768),
#             nn.ReLU(),
#             nn.Linear(768, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Linear(512,256), 
#             nn.LayerNorm(256),
#             nn.ReLU(),
#         )
#     def forward(self, x):
#         return self.layers(x) 

class FeatureExtractor(nn.Module):
    def __init__(self,input_size,dropout=0.5):
        super(FeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.layers(x)
    
# def FeatureExtractor(input_size,dropout=0.5):
#     layers = [
#         nn.Linear(input_size, 768),
#         nn.BatchNorm1d(768),
#         nn.ReLU(),
#         nn.Dropout(dropout),
#         nn.Linear(768, 512),
#         nn.BatchNorm1d(512),
#         nn.ReLU(),
#         nn.Dropout(dropout),
#         nn.Linear(512,256),
#         nn.BatchNorm1d(256),
#         nn.ReLU(),
#     ]
#     return nn.Sequential(*layers)


# def linear_block(input_size,output_size):
#     layers = []
#     layers.append(nn.Linear(input_size,128))
#     layers.append(nn.BatchNorm1d(128))
#     layers.append(nn.ReLU())
#     layers.append(nn.Linear(128, 64))
#     layers.append(nn.BatchNorm1d(64))
#     layers.append(nn.ReLU())
#     layers.append(nn.Linear(64, output_size))
#     return nn.Sequential(*layers)

def linear_block(input_size, output_size, dropout=0.5):
    layers = [
        nn.Linear(input_size, 128), 
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, output_size)
        ]
    return nn.Sequential(*layers)

# class linear_block_cbp(nn.Module):
#     def __init__(self,input_size,output_size):
#         super(linear_block_cbp, self).__init__()
#         self.fc1 = nn.Linear(input_size,128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_size)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.act = nn.ReLU()
#         self.layers = nn.ModuleList()
#         self.layers.append(self.fc1)
#         self.layers.append(self.bn1)
#         self.layers.append(self.act)
#         self.layers.append(self.fc2)
#         self.layers.append(self.bn2)
#         self.layers.append(self.act)
#         self.layers.append(self.fc3)
       
#     def forward(self, x):
#         x1 = self.act(self.bn1(self.fc1(x)))
#         x2 = self.act(self.bn2(self.fc2(x1)))
#         x3 = self.fc3(x2)
#         return x3,[x1,x2]

class FeatureExtractor_cbp(nn.Module):
    def __init__(self,input_size,replacement_rate=1e-3,init='kaiming',maturity_threshold=10,dropout=0.5):
        super(FeatureExtractor_cbp, self).__init__()
        self.fc1 = nn.Linear(input_size, 768)
        self.bn1 = nn.BatchNorm1d(768)
        self.fc2 = nn.Linear(768, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512,256)
        self.bn3 = nn.BatchNorm1d(256) 
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            self.fc1, self.bn1, self.act, self.dropout,
            self.fc2, self.bn2, self.act, self.dropout,
            self.fc3, self.bn3, self.act
            ])

    def forward(self, x):
        x1 = self.layers[3](self.layers[2](self.layers[1](self.layers[0](x))))
        x2 = self.layers[7](self.layers[6](self.layers[5](self.layers[4](x1))))
        x3 = self.layers[10](self.layers[9](self.layers[8](x2)))
        return x3, [x1,x2]

class linear_block_cbp(nn.Module):
    def __init__(self,input_size,output_size,replacement_rate=1e-3,init='kaiming',maturity_threshold=10,dropout=0.5):
        super(linear_block_cbp, self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, output_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            self.fc1, self.bn1,self.act,
            self.dropout, self.fc2
            ])
    def forward(self, x):
        x1 = self.layers[3](self.layers[2](self.layers[1](self.layers[0](x))))
        x2 = self.layers[4](x1)
        return x2,[x1]
    
# class FeatureExtractor_cbp(nn.Module):
#     def __init__(self,input_size,replacement_rate=1e-3,init='kaiming',maturity_threshold=10,dropout=0.5):
#         super(FeatureExtractor_cbp, self).__init__()
#         self.fc1 = nn.Linear(input_size, 768)
#         self.bn1 = nn.BatchNorm1d(768)
#         self.fc2 = nn.Linear(768, 512)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512,256)
#         self.bn3 = nn.BatchNorm1d(256)

#         self.act = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)
        
#         self.cbp1 = CBPLinear(in_layer=self.fc1, out_layer=self.fc2, bn_layer=self.bn1, replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
#         self.cbp2 = CBPLinear(in_layer=self.fc2, out_layer=self.fc3, bn_layer=self.bn2,replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)
         
#         self.layers = nn.ModuleList([
#             self.fc1, self.bn1, self.act, self.dropout,
#             self.fc2, self.bn2, self.act, self.dropout,
#             self.fc3, self.bn3, self.act
#             ])

#     def forward(self, x):
#         x1 = self.cbp1(self.act(self.bn1(self.fc1(x))))
#         x2 = self.dropout(x1)
#         x3 = self.cbp2(self.act(self.bn2(self.fc2(x2))))
#         x4 = self.act(self.bn3(self.fc3(self.dropout(x3))))
#         return x4, [x1,x3]

# class linear_block_cbp(nn.Module):
#     def __init__(self,input_size,output_size,replacement_rate=1e-3,init='kaiming',maturity_threshold=10,dropout=0.5):
#         super(linear_block_cbp, self).__init__()
#         self.fc1 = nn.Linear(input_size,128)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, output_size)

#         self.act = nn.ReLU()
#         self.dropout = nn.Dropout(dropout)

#         self.cbp1 = CBPLinear(in_layer=self.fc1, out_layer=self.fc2,bn_layer=self.bn1,replacement_rate=replacement_rate, maturity_threshold=maturity_threshold, init=init)

#         self.layers = nn.ModuleList([
#             self.fc1, self.bn1,self.act,
#             self.dropout,self.fc2
#             ])
       
#     def forward(self, x):
#         x1 = self.cbp1(self.act(self.bn1(self.fc1(x))))
#         x2 = self.fc2(self.dropout(x1))
#         return x2,[x1]

