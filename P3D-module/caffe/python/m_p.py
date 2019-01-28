from caffe.proto import caffe_pb2

def toPrototxt(modelName, deployName):
    with open(modelName, 'rb') as f:
        caffemodel = caffe_pb2.NetParameter()
        caffemodel.ParseFromString(f.read())

    for item in caffemodel.layers:
        item.ClearField('blobs')
    for item in caffemodel.layer:
        item.ClearField('blobs')
        
    # print(caffemodel)
    with open(deployName, 'w') as f:
        f.write(str(caffemodel))

if __name__ == '__main__':
    modelName = '/data/wjc/TCNN_STCNN/davis_240_320.caffemodel'
    deployName = 'tcnn.prototxt'
    toPrototxt(modelName, deployName)
