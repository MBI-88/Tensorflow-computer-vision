"""
Created on July 23 12:02:10 2021
@author:MBI

Descripcion:
Script para la implementacion del modelo Houglass  (deteccion de pose humana)
"""
#%% Modulos
from tensorflow.keras.models import Model,load_model,model_from_json
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPooling2D,Add,Input,SeparableConv2D,UpSampling2D
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
#%%  Modelo

def create_houglass_network(num_class,num_stacks,num_channels,inres,outres,bottleneck):
    input = Input(shape=(inres[0],inres[1],3))
    front_features = create_front_module(input,num_channels,bottleneck)
    head_next_stage = front_features
    outputs = []
    for i in range(num_stacks):
        head_next_stage,head_to_loss = houglass_model(head_next_stage,num_class,num_channels,bottleneck,i)
        outputs.append(head_to_loss)

    model = Model(inputs=input,outputs=outputs)
    rms = RMSprop(lr=5e-4)
    model.compile(optimizer=rms,loss=mean_squared_error,metrics=['accuracy'])
    return model

def houglass_model(bottom,num_classes,num_channels,bottleneck,hgid):
    left_features = create_left_half_blocks(bottom,bottleneck,hgid,num_channels)
    rf1 = create_right_half_blocks(left_features,bottleneck,hgid,num_channels)
    head_next_stage,head_parts = create_heads(bottom,rf1,num_classes,hgid,num_channels)
    return head_next_stage,head_parts

def bottleneck_block(bottom,num_out_channels,block_name):
    if  K.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = Conv2D(num_out_channels,kernel_size=(1,1),activation='relu', padding='same',name=block_name + 'skip')(bottom)

    _x = Conv2D(num_out_channels//2,kernel_size=(1,1),activation='relu',padding='same',name=block_name + '_conv_1x1_x1')(bottom)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels//2,kernel_size=(3,3),activation='relu',padding='same',name=block_name + '_conv_3x3_x2')(_x)
    _x = BatchNormalization()(_x)
    _x = Conv2D(num_out_channels//2,kernel_size=(1,1),activation='relu',padding='same',name=block_name + '_cov_1x1_x3')(_x)
    _x = BatchNormalization()(_x)
    _x = Add(name=block_name + '_residual')([_skip,_x])
    return _x

def bottleneck_mobile(bottom,num_out_channels,block_name):
    if K.int_shape(bottom)[-1] == num_out_channels:
        _skip = bottom
    else:
        _skip = SeparableConv2D(num_out_channels,kernel_size=(1,1),activation='relu',padding='same',name=block_name + '_skiip')(bottom)

    _x = SeparableConv2D(num_out_channels//2,kernel_size=(1,1),activation='relu',padding='same',name=block_name + '_conv_1x1_x1')(bottom)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_channels//2,kernel_size=(3,3),activation='relu',padding='same',name=block_name + '_conv_3x3_x2')(_x)
    _x = BatchNormalization()(_x)
    _x = SeparableConv2D(num_out_channels//2,kernel_size=(1,1),activation='relu',padding='same',name=block_name + '_cov_1x1_x3')(_x)
    _x = BatchNormalization()(_x)
    _x = Add(name=block_name + '_residual')([_skip, _x])
    return _x

def create_front_module(input,num_channels,bottleneck):
    _x = Conv2D(64,kernel_size=(7,7),strides=(2,2),padding='same',activation='relu',name='Front_conv_1x1')(input)
    _x = BatchNormalization()(_x)
    _x = bottleneck(_x,num_channels//2,'front_conv_x1')
    _x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(_x)
    _x = bottleneck(_x, num_channels // 2, 'front_conv_x2')
    _x = bottleneck(_x, num_channels // 2, 'front_conv_x3')
    return _x

def create_left_half_blocks(bottom,bottleneck,hglayer,num_channels):
    hgname = 'hg'+ str(hglayer)
    f1 = bottleneck(bottom,num_channels,hgname + '_l1')
    _x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(f1)
    f2 = bottleneck(_x,num_channels,hgname, + '_l2')
    _x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(f2)
    f4 = bottleneck(_x,num_channels,hgname + '_l4')
    _x = MaxPooling2D(pool_size=(2,2),strides=(2,2))(f4)
    f8 = bottleneck(_x,num_channels,hgname + '_l8')
    return (f1,f2,f4,f8)

def connect_lef_to_right(left,right,bottleneck,name,num_channels):
    _xleft = bottleneck(left,num_channels,name+'_connect')
    _xright = UpSampling2D()(right)
    add = Add()([_xleft,_xright])
    out = bottleneck(add,num_channels,name+'connect_conv')
    return out

def bottom_layer(lf8,bottleneck,hgid,num_channels):
    lf8_connect = bottleneck(lf8,num_channels,str(hgid)+'_lf8')
    _x = bottleneck(lf8,num_channels,str(hgid)+'_lf8_x1')
    _x = bottleneck(_x,num_channels,str(hgid)+'_lf8_x2')
    _x = bottleneck(_x,num_channels,str(hgid)+ '_lf8_x3')
    rf8 = Add()([_x,lf8_connect])
    return rf8

def create_right_half_blocks(leftfeatures,bottleneck,hglayer,num_channels):
    lf1,lf2,lf4,lf8 = leftfeatures
    rf8 = bottom_layer(lf8,bottleneck,hglayer,num_channels)
    rf4 = connect_lef_to_right(lf4, rf8,bottleneck,'hg'+ str(hglayer) + '_rf4',num_channels)
    rf2 = connect_lef_to_right(lf2, rf4, bottleneck, 'hg' + str(hglayer) + '_rf2', num_channels)
    rf1 = connect_lef_to_right(lf1, rf2, bottleneck, 'hg' + str(hglayer) + '_rf1', num_channels)
    return rf1

def create_heads(prelayerfeatures,rf1,num_classes,hgid,num_channels):
    head = Conv2D(num_channels,kernel_size=(1,1),activation='relu', padding='same',name=str(hgid)+'_conv_1x1_x1')(rf1)
    head = BatchNormalization()(head)
    head_parts = Conv2D(num_classes,kernel_size=(1,1),activation='relu',padding='same',name=str(hgid)+'_conv_1x1_parts')(head)
    head = Conv2D(num_channels,kernel_size=(1,1),activation='relu',padding='same',name=str(hgid)+'_conv_1x1_x2')(head)
    head_m = Conv2D(num_channels,kernel_size=(1,1),activation='linear',padding='same',name=str(hgid)+'_conv_1x1_x3')(head_parts)
    
    head_next_stage = Add()([head,head_m,prelayerfeatures])
    return head_next_stage,head_parts

def euclidian_loss(x,y):
    return K.sqrt(K.sum(K.square(x-y)))




