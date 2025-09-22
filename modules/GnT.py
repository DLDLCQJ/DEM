from torch.nn import Linear, BatchNorm1d, LayerNorm, Module
from torch import where, rand, topk, long, empty, zeros, no_grad, tensor
import torch
from torch import nn
import sys
from math import sqrt
from utils.AdamGnT import AdamGnT
from torch.nn.init import calculate_gain

# def get_layer_std(layer, gain):
#     if isinstance(layer, Linear):
#         return gain * sqrt(1 / layer.in_features)
    
def get_layer_bound(layer, init, gain):
  
    if isinstance(layer, Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound
    
class FFGnT(object):
    def __init__(self, net, hidden_activation, opt, decay_rate=0.99, replacement_rate=1e-4, init='kaiming',
                 util_type='contribution', maturity_threshold=100, device='cpu',accumulate=False):
        super(FFGnT, self).__init__()

        self.net = net
        self.bn_layers = []
        self.ln_layers = []
        self.weight_layers = []
        self.get_weight_layers(nn_module=self.net)
        self.num_hidden_layers = len(self.weight_layers) - 1
        self.device = device
        self.accumulate = accumulate
        self.opt = opt
        self.opt_type = 'sgd'
        if isinstance(self.opt, AdamGnT):
            self.opt_type = 'adam'
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        # Initialize utility tracking variables
        self.util, self.bias_corrected_util, self.ages, self.mean_feature_mag = [], [], [], []
        for i in range(self.num_hidden_layers):
            self.util.append(zeros(self.weight_layers[i].out_features, dtype=torch.float32, device=self.device))
            self.bias_corrected_util.append(zeros(self.weight_layers[i].out_features,dtype=torch.float32, device=self.device))
            self.ages.append(zeros(self.weight_layers[i].out_features, dtype=torch.float32, device=self.device))
            self.mean_feature_mag.append(zeros(self.weight_layers[i].out_features, dtype=torch.float32, device=self.device))
        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]

        # Calculate bounds for each layer
        #self.stds = self.compute_std(hidden_activation=hidden_activation)
        self.bounds = self.compute_bounds(hidden_activation=hidden_activation, init=init)

    def get_weight_layers(self, nn_module):
        # Recursively navigate submodules
        for layer in nn_module.children():
            if isinstance(layer, Linear):
                self.weight_layers.append(layer)
            elif isinstance(layer, BatchNorm1d):
                self.bn_layers.append(layer)
            elif isinstance(layer, LayerNorm):
                self.ln_layers.append(layer)
            elif isinstance(layer, Module):
                self.get_weight_layers(layer)

    def compute_bounds(self, hidden_activation, init='kaiming'):
        bounds = []
        gain = calculate_gain(nonlinearity=hidden_activation)
        for i in range(self.num_hidden_layers):
            bounds.append(get_layer_bound(layer=self.weight_layers[i], init=init,gain=gain))
        bounds.append(get_layer_bound(layer=self.weight_layers[-1],init=init, gain=1))
        return bounds
    
    def update_utility(self, layer_idx=0, features=None):
        with torch.no_grad():
            self.util[layer_idx] *= self.decay_rate
            """
            Adam-style bias correction
            """
            bias_correction = 1 - self.decay_rate ** self.ages[layer_idx]

            self.mean_feature_mag[layer_idx] *= self.decay_rate
            self.mean_feature_mag[layer_idx] -= - (1 - self.decay_rate) * features.mean(dim=0)
            bias_corrected_act = self.mean_feature_mag[layer_idx] / bias_correction

            current_layer = self.weight_layers[layer_idx]
            next_layer = self.weight_layers[layer_idx+1]
            input_wight_mag = current_layer.weight.data.abs().mean(dim=1)
            output_wight_mag = next_layer.weight.data.abs().mean(dim=0)

            if self.util_type == 'weight':
                new_util = output_wight_mag
            elif self.util_type == 'contribution':
                new_util = output_wight_mag * features.abs().mean(dim=0)
            elif self.util_type == 'adaptation':
                new_util = 1/input_wight_mag
            elif self.util_type == 'zero_contribution':
                new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0)
            elif self.util_type == 'adaptable_contribution':
                new_util = output_wight_mag * (features - bias_corrected_act).abs().mean(dim=0) / input_wight_mag
            elif self.util_type == 'feature_by_input':
                input_wight_mag = current_layer.weight.data.abs().mean(dim=1)
                new_util = (features - bias_corrected_act).abs().mean(dim=0) / input_wight_mag
            else:
                new_util = 0

            self.util[layer_idx] += (1 - self.decay_rate) * new_util

            self.bias_corrected_util[layer_idx] = self.util[layer_idx] / bias_correction

            if self.util_type == 'random':
                self.bias_corrected_util[layer_idx] = torch.rand(self.util[layer_idx].shape)

    def test_features(self, features):
        features_to_replace = [empty(0, dtype=long) for _ in range(self.num_hidden_layers)]
        num_features_to_replace = [0 for _ in range(self.num_hidden_layers)]
        if self.replacement_rate == 0:
            return features_to_replace, num_features_to_replace
        #print('length_hidden:', self.num_hidden_layers)
        for i in range(self.num_hidden_layers):
            self.ages[i] += 1
            '''Update feature utility'''
            self.update_utility(layer_idx=i, features=features[i])
            '''Find the no. of features to replace'''
            #print(self.ages[i])
            eligible_feature_indices = where(self.ages[i] > self.maturity_threshold)[0]
            if eligible_feature_indices.shape[0] == 0: continue
            num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
            self.accumulated_num_features_to_replace[i] += num_new_features_to_replace
            '''Case when the number of features to be replaced is between 0 and 1.'''
            if self.accumulate:
                num_new_features_to_replace = int(self.accumulated_num_features_to_replace[i])
                self.accumulated_num_features_to_replace[i] -= num_new_features_to_replace
            else:
                if num_new_features_to_replace < 1:
                    if torch.rand(1) <= num_new_features_to_replace:
                        num_new_features_to_replace = 1
                num_new_features_to_replace = int(num_new_features_to_replace)
            if num_new_features_to_replace == 0:    continue
            # select the indices of the features with the lowest utility (lowest importance).
            new_features_to_replace = topk(-self.bias_corrected_util[i][eligible_feature_indices], num_new_features_to_replace)[1]
            new_features_to_replace = eligible_feature_indices[new_features_to_replace]
            '''Initialize utility for new features'''
            self.util[i][new_features_to_replace] = 0
            self.mean_feature_mag[i][new_features_to_replace] = 0.
            features_to_replace[i] = new_features_to_replace
            num_features_to_replace[i] = num_new_features_to_replace
        return features_to_replace, num_features_to_replace

    def gen_new_features(self, features_to_replace, num_features_to_replace):
        '''Generate new features: Reset input and output weights for low utility features'''
        with torch.no_grad():
            for i in range(self.num_hidden_layers):
                if num_features_to_replace[i] == 0:
                    continue
            
                current_layer, next_layer = self.weight_layers[i], self.weight_layers[i+1]
                current_layer.weight.data[features_to_replace[i], :] *= 0.0
                current_layer.weight.data[features_to_replace[i], :] += \
                    empty(num_features_to_replace[i], current_layer.in_features).uniform_(-self.bounds[i], self.bounds[i]).to(self.device)
                    #empty([num_features_to_replace[i]] + list(current_layer.weight.shape[1:]), device=self.device).normal_(std=self.stds[i])
                current_layer.bias.data[features_to_replace[i]] *= 0
                '''Update bias to correct for the removed features and set the outgoing weights and ages to zero'''
                next_layer.weight.data[:, features_to_replace[i]] = 0
                next_layer.bias.data += (next_layer.weight.data[:, features_to_replace[i]] * \
                                                self.mean_feature_mag[i][features_to_replace[i]] / \
                                                (1 - self.decay_rate ** self.ages[i][features_to_replace[i]])).sum(dim=1)
                
                self.ages[i][features_to_replace[i]] = 0

    def update_optim_params(self, features_to_replace, num_features_to_replace):
        if self.opt_type == 'adam':
            for i in range(self.num_hidden_layers):
                # input weights
                if num_features_to_replace[i] == 0:
                    continue
                self.opt.state[self.weight_layers[i].weight]['exp_avg'][features_to_replace[i], :] = 0.0
                self.opt.state[self.weight_layers[i].bias]['exp_avg'][features_to_replace[i]] = 0.0
                self.opt.state[self.weight_layers[i].weight]['exp_avg_sq'][features_to_replace[i], :] = 0.0
                self.opt.state[self.weight_layers[i].bias]['exp_avg_sq'][features_to_replace[i]] = 0.0
                self.opt.state[self.weight_layers[i].weight]['step'][features_to_replace[i], :] = 0
                self.opt.state[self.weight_layers[i].bias]['step'][features_to_replace[i]] = 0
                # output weights
                self.opt.state[self.weight_layers[i+1].weight]['exp_avg'][:, features_to_replace[i]] = 0.0
                self.opt.state[self.weight_layers[i+1].weight]['exp_avg_sq'][:, features_to_replace[i]] = 0.0
                self.opt.state[self.weight_layers[i+1].weight]['step'][:, features_to_replace[i]] = 0

    def gen_and_test(self, features):
        if not isinstance(features, list):
            print('features passed to generate-and-test should be a list')
            sys.exit()
        features_to_replace, num_features_to_replace = self.test_features(features=features)
        self.gen_new_features(features_to_replace, num_features_to_replace)
        self.update_optim_params(features_to_replace, num_features_to_replace)



