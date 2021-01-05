from tensorflow.keras.layers import Input, Conv2D, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, GlobalAvgPool2D, Dropout
from tensorflow.keras.layers import Add, ReLU, Dense
from tensorflow.keras import Model

#Conv-batchNorm_relu block
def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

#Identity block
def identity_block(tensor, filters):
    
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    
    x = Add()([tensor,x])    #skip connection
    x = ReLU()(x)
    
    return x

#Projection block
def projection_block(tensor, filters, strides):
    
    #left stream
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4*filters, kernel_size=1, strides=1)(x)
    x = BatchNormalization()(x)
    
    #right stream
    shortcut = Conv2D(filters=4*filters, kernel_size=1, strides=strides)(tensor)
    shortcut = BatchNormalization()(shortcut)
    
    x = Add()([shortcut,x])    #skip connection
    x = ReLU()(x)
    
    return x

#Resnet block
def resnet_block(x, filters, reps, strides):
    
    x = projection_block(x, filters, strides)
    for _ in range(reps-1):
        x = identity_block(x,filters)
        
    return x

def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return  image, label

def build_resnet():
    input_layer = Input(shape=(224,224,3)) #input shape

    x = conv_batchnorm_relu(input_layer,filters=64,kernel_size=7,strides=2)
    x = MaxPool2D(pool_size=3,strides=2)(x)
    x = resnet_block(x,filters=64,reps=3,strides=1)
    x = resnet_block(x,filters=128,reps=4,strides=2)
    x = resnet_block(x,filters=256,reps=6,strides=2)
    x = resnet_block(x,filters=512,reps=3,strides=2)
    x = GlobalAvgPool2D()(x)
    x = Dense(1000,activation='relu')(x)
    x = Dropout(.2)(x)
    output_layer = Dense(2,activation='softmax')(x)

    model = Model(inputs=input_layer,outputs=output_layer)
    return model


if __name__ == "__main__":
    #build model
    resnet = build_resnet()

