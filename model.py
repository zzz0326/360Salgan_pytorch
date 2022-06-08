import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

bgr_mean = np.array([103.939, 116.779, 123.68]).astype(np.float32)
class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.sigmoid = nn.Sigmoid()

        # conv_1
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), padding=1)
        # conv_2
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), padding=1)

        # conv_3
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), padding=1)

        # conv_4
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), padding=1)

        # conv_5
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), padding=1)

        # deconv_5
        self.uconv5_3 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.uconv5_2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.uconv5_1 = nn.Conv2d(512, 512, (3, 3), padding=1)

        # deconv_4
        self.uconv4_3 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.uconv4_2 = nn.Conv2d(512, 512, (3, 3), padding=1)
        self.uconv4_1 = nn.Conv2d(512, 512, (3, 3), padding=1)

        # deconv_3
        self.uconv3_3 = nn.Conv2d(512, 256, (3, 3), padding=1)
        self.uconv3_2 = nn.Conv2d(256, 256, (3, 3), padding=1)
        self.uconv3_1 = nn.Conv2d(256, 256, (3, 3), padding=1)

        # deconv_2
        self.uconv2_2 = nn.Conv2d(256, 128, (3, 3), padding=1)
        self.uconv2_1 = nn.Conv2d(128, 128, (3, 3), padding=1)

        # deconv_1
        self.uconv1_2 = nn.Conv2d(128, 64, (3, 3), padding=1)
        self.uconv1_1 = nn.Conv2d(64, 64, (3, 3), padding=1)

        # output
        self.output = nn.Conv2d(64, 1, (1, 1), padding=0)

    def forward(self, x):
        #assert(x.size()[1:] == (3, 192, 256))
        #print 'input', x.size()
        # todo insert data pre-processing: subtract image net mean image

        # conv_1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.max_pool2d(x)
        # print 'pool1: ', x.size()

        # conv_2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.max_pool2d(x)
        # print 'pool2: ', x.size()

        # conv_3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.max_pool2d(x)
        # print 'pool3: ', x.size()

        # conv_4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.max_pool2d(x)
        # print 'pool4: ', x.size()

        # conv_5
        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        # print 'conv5: ', x.size()


        # deconv_5
        x = self.relu(self.uconv5_3(x))
        x = self.relu(self.uconv5_2(x))
        x = self.relu(self.uconv5_1(x))
        #print 'uconv5: ', x.size()

        # pool 4
        x = self.up_sample(x)
        #print 'upool4: ', x.size()

        # deconv 4
        x = self.relu(self.uconv4_3(x))
        x = self.relu(self.uconv4_2(x))
        x = self.relu(self.uconv4_1(x))
        #print 'uconv4: ', x.size()

        # pool 3
        x = self.up_sample(x)
        #print 'upool3: ', x.size()

        # deconv 3
        x = self.relu(self.uconv3_3(x))
        x = self.relu(self.uconv3_2(x))
        x = self.relu(self.uconv3_1(x))
        #print 'uconv3: ', x.size()

        # pool 2
        x = self.up_sample(x)
        #print 'upool2: ', x.size()

        # deconv 2
        x = self.relu(self.uconv2_2(x))
        x = self.relu(self.uconv2_1(x))
        #print 'uconv2: ', x.size()

        # pool 1
        x = self.up_sample(x)
        #print 'upool1: ', x.size()

        # deconv 1
        x = self.relu(self.uconv1_2(x))
        x = self.relu(self.uconv1_1(x))
        #print 'uconv1: ', x.size()

        # output
        x = self.sigmoid(self.output(x))
        #print 'output: ', x.size()

        return x


'''
model structure
Input: (3, 192, 256)
conv1_1: (64, 192, 256)
conv1_2: (64, 192, 256)
pool1: (64, 96, 128)
conv2_1: (128, 96, 128)
conv2_2: (128, 96, 128)
pool2: (128, 48, 64)
conv3_1: (256, 48, 64)
conv3_2: (256, 48, 64)
conv3_3: (256, 48, 64)
pool3: (256, 24, 32)
conv4_1: (512, 24, 32)
conv4_2: (512, 24, 32)
conv4_3: (512, 24, 32)
pool4: (512, 12, 16)
conv5_1: (512, 12, 16)
conv5_2: (512, 12, 16)
conv5_3: (512, 12, 16)
uconv5_3: (512, 12, 16)
uconv5_2: (512, 12, 16)
uconv5_1: (512, 12, 16)
upool4: (512, 24, 32)
uconv4_3: (512, 24, 32)
uconv4_2: (512, 24, 32)
uconv4_1: (512, 24, 32)
upool3: (512, 48, 64)
uconv3_3: (256, 48, 64)
uconv3_2: (256, 48, 64)
uconv3_1: (256, 48, 64)
upool2: (256, 96, 128)
uconv2_2: (128, 96, 128)
uconv2_1: (128, 96, 128)
upool1: (128, 192, 256)
uconv1_2: (64, 192, 256)
uconv1_1: (64, 192, 256)
output: (1, 192, 256)
'''


def theano_conv_2_torch_tensor(torch_dict, weights_dict, w_name, b_name, conv_name, flip_filter=False):
    weight = weights_dict[w_name]
    bias = weights_dict[b_name]
    if flip_filter:
        weight = weight[:, :, ::-1, ::-1].copy()  # important
        print(weight.shape)
        weight = torch.from_numpy(weight)
    else:
        print(weight.shape)
        weight = torch.from_numpy(weight)

    bias = torch.from_numpy(bias)
    print(weight.size(), bias.size())
    print(torch_dict[conv_name + '.weight'].size(), torch_dict[conv_name + '.bias'].size())
    assert torch_dict[conv_name + '.weight'].size() == weight.size()
    assert torch_dict[conv_name + '.bias'].size() == bias.size()
    torch_dict[conv_name + '.weight'] = weight
    torch_dict[conv_name + '.bias'] = bias


def load_npz_weights(torch_dict, weights):
    conv_name = ['conv1_1', 'conv1_2',
                 'conv2_1', 'conv2_2',
                 'conv3_1', 'conv3_2', 'conv3_3',
                 'conv4_1', 'conv4_2', 'conv4_3',
                 'conv5_1', 'conv5_2', 'conv5_3']
    uconv_name = ['u' + name for name in conv_name[::-1]]

    # convert encoder
    for i, name in zip(range(len(conv_name)), conv_name):
        print('arr_%d, arr_%d, %s' % (2 * i, 2 * i + 1, name))
        theano_conv_2_torch_tensor(torch_dict, weights,
                                   'arr_%d' % (2 * i), 'arr_%d' % (2 * i + 1), name, False)

    # convert decoder
    offset = len(conv_name) * 2
    for i, name in zip(range(len(uconv_name)), uconv_name):
        print('arr_%d, arr_%d, %s' % (2 * i + offset, 2 * i + 1 + offset, name))
        theano_conv_2_torch_tensor(torch_dict, weights,
                                   'arr_%d' % (2 * i + offset), 'arr_%d' % (2 * i + 1 + offset), name, True)

    # convert output
    theano_conv_2_torch_tensor(torch_dict, weights, 'arr_52', 'arr_53', 'output', True)


def save_pytorch_model():

    model = VGG16()
    torch.save(model.state_dict(), './90.pkl')

    npz_path = 'E:\code\salgan-master\scripts\\1845_90_gen_modelWeights0180.npz'
    pkl_path = '.\\90.pkl'

    npz_weight = np.load(npz_path)

    pkl_weight = torch.load(pkl_path, map_location='cpu')


    load_npz_weights(pkl_weight, npz_weight)

    torch.save(pkl_weight, '.\\90.pkl')
    del pkl_weight




# model = VGG16().cpu()
# weight = torch.load(
#     "./360.pkl", map_location='cpu')
#
# model.load_state_dict(weight, strict=True)
# del weight
#
#
# bgr_mean = np.array([103.939, 116.779, 123.68]).astype(np.float32)
# test = cv2.imread('im360.jpg', cv2.IMREAD_COLOR)
# image = cv2.resize(test, (256, 192), interpolation=cv2.INTER_AREA)
#
# image = np.transpose(image, [2, 0, 1])
# image = image.astype(np.float32)
# image -= bgr_mean[:, np.newaxis, np.newaxis]
# X = torch.from_numpy(image)
# X = X.unsqueeze(0)
#
# out = model(X).detach().numpy()
# out = out[0,]
# out = out.transpose(1, 2, 0)
# out = (out * 255).astype(np.uint8)
#
# out = cv2.resize(out, (5000, 2500), interpolation=cv2.INTER_CUBIC)
# out = cv2.GaussianBlur(out, (5, 5), 0)
# out = np.clip(out, 0, 255)
#
# cv2.imwrite('pytorch_output.jpg', out)
