from caffe2.python import core, workspace, model_helper
from caffe2.python import brew, optimizer, utils
from caffe2.proto import caffe2_pb2

import os
import numpy as np

core.GlobalInit(['caffe2', '--caffe2_log_level=-1']) # change to 0 to disable

############### Definition part ###############

def AddInput(model, batch_size, db):
    # load the data
    data_uint8, label = model.TensorProtosDBInput(
        [], ["data_uint8", "label"], batch_size=batch_size,
        db=db, db_type='leveldb')
    # cast the data to float
    data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
    # scale data from [0,255] down to [0,1]
    data = model.Scale(data, data, scale=float(1./256))
    # don't need the gradient for the backward pass
    data = model.StopGradient(data, data)
    return data, label

def AddLeNetModel(model, data):
    '''
    This part is the standard LeNet model: from data to the softmax prediction.
    
    For each convolutional layer we specify dim_in - number of input channels
    and dim_out - number or output channels. Also each Conv and MaxPool layer changes the
    image size. For example, kernel of size 5 reduces each side of an image by 4.

    While when we have kernel and stride sizes equal 2 in a MaxPool layer, it divides
    each side in half.
    '''
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8 -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3 = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, fc3, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')
    return softmax

def AddTrainingOperators(model, softmax, label):
    """Adds training operators to the model."""
    xent = model.LabelCrossEntropy([softmax, label], 'xent')
    # compute the expected loss
    loss = model.AveragedLoss(xent, "loss")
    # track the accuracy of the model
    accuracy = model.Accuracy([softmax, label], "accuracy")
    # use the average loss we just computed to add gradient operators to the model
    model.AddGradientOperators([loss])
    # do a simple stochastic gradient descent
    opt = optimizer.build_sgd(model, base_learning_rate=0.01, policy="step", stepsize=1, gamma=0.999)  # , momentum=0.9
    #with core.DeviceScope(device_opts):
    #    brew.add_weight_decay(train_model, 0.001)  # any effect???

############### Training part ###############

dev_opts = core.DeviceOption(caffe2_pb2.CPU)

arg_scope = {"order": "NCHW", "use_cudnn" : False}
data_folder = os.curdir

train_model = model_helper.ModelHelper(name="mnist_train", arg_scope=arg_scope)
data, label = AddInput(train_model, batch_size=64,
        db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'))
softmax = AddLeNetModel(train_model, data)
AddTrainingOperators(train_model, softmax, label)

train_model.Print('accuracy', [], to_file=1)
train_model.Print('loss', [], to_file=1)

# now the net is defined lets run training
workspace.RunNetOnce(train_model.param_init_net)
workspace.CreateNet(train_model.net)

epochs = 1
print '\ntraining for', epochs, 'epochs'

# 937 * 64 is nearly the size of training set
for j in range(0, epochs):
    workspace.RunNet(train_model.net, 937)

############### Testing part ###############

test_model = model_helper.ModelHelper(
    name="mnist_test", arg_scope=arg_scope, init_params=False)
data, label = AddInput(test_model, batch_size=100,
    db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'))
softmax = AddLeNetModel(test_model, data)
accuracy = test_model.Accuracy([softmax, label], "accuracy")

workspace.RunNetOnce(test_model.param_init_net)
workspace.CreateNet(test_model.net)

test_accuracy = np.zeros(100) # 100 * 100 is the size of test corpora
for i in range(100):
    workspace.RunNet(test_model.net)
    test_accuracy[i] = workspace.FetchBlob('accuracy')

print 'test_accuracy: %f' % test_accuracy.mean()

############### Saving part ###############

from caffe2.python.predictor.mobile_exporter import Export

INIT_NET = 'mnist_init_net.pb'
PREDICT_NET = 'mnist_predict_net.pb'

deploy_model = model_helper.ModelHelper(
    name="mnist_deploy", arg_scope=arg_scope, init_params=False)
AddLeNetModel(deploy_model, "data")

init_net, predict_net = Export(workspace, deploy_model.net, deploy_model.params)

with open(PREDICT_NET, 'wb') as f:
    # for human readable version use: f.write(str(predict_net))
    f.write(predict_net.SerializeToString())

with open(INIT_NET, 'wb') as f:
    f.write(init_net.SerializeToString())

############### Loading part ###############

# let's first clear current workspace to make sure we get blobs from
# loading and not some leftovers (we might also switch workspace to a new
# one)

workspace.ResetWorkspace()
# now workspace.Blobs() returns []

init_def = caffe2_pb2.NetDef()
with open(INIT_NET, 'r') as f:
    init_def.ParseFromString(f.read())
    init_def.device_option.CopyFrom(dev_opts)

workspace.RunNetOnce(init_def.SerializeToString())
# now workspace.Blobs() returns blobs from net definition

net_def = caffe2_pb2.NetDef()
with open(PREDICT_NET, 'r') as f:
    net_def.ParseFromString(f.read())
    net_def.device_option.CopyFrom(dev_opts)

workspace.CreateNet(net_def.SerializeToString(), overwrite=True)

############### Testing part 2 ###############

import leveldb

db = leveldb.LevelDB('mnist-test-nchw-leveldb')

# let's pick first image
iprot, lprot = caffe2_pb2.TensorProtos.FromString(db.Get('00000000')).protos

# As of writing this feeding TensorProto is not yet supported
#workspace.FeedBlob("data", iprot)
img = np.fromstring(iprot.byte_data, dtype=np.uint8)
workspace.FeedBlob('data', (img/256.).reshape([1,1,28,28]).astype(np.float32))
workspace.RunNet(net_def.name)
softmax = workspace.FetchBlob('softmax')
print "Output:", softmax
print "Class:", np.argmax(softmax), "expected:", lprot.int32_data[0]

# You can view the image with the following (ipython'ish)
#%matplotlib
#import matplotlib.pyplot as plt
#plt.figure()
#plt.imshow(img.reshape([28, 28]))
