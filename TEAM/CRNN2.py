
from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, Bidirectional
from keras.layers import Input, Dense, Activation
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.layers.recurrent import LSTM, GRU

def CRNN3(input_shape, num_classes, prediction_only=False, gru=False):
    """CRNN architecture.
    
    # Arguments
        input_shape: Shape of the input image, (256, 32, 1).
        num_classes: Number of characters in alphabet, including CTC blank.
        
    # References
        https://arxiv.org/abs/1507.05717
        123123434156526926
    """
    
    act = 'relu'
    
    # KERAS API를 사용한 모델 구현
    x = image_input = Input(shape=input_shape, name='image_input')
    
    x = Conv2D(64, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv1_1')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool1', padding='same')(x)
    
    x = Conv2D(128, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv2_1')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool2', padding='same')(x)
    
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv3_2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), name='pool3', padding='same')(x)
    
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv4_1')(x)
    x = BatchNormalization(name='batchnorm1')(x)
    
    x = Conv2D(512, (3, 3), strides=(1, 1), activation=act, padding='same', name='conv5_1')(x)
    x = BatchNormalization(name='batchnorm2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(1, 2), name='pool5', padding='valid')(x)
    
    x = Conv2D(512, (2, 2), strides=(1, 1), activation=act, padding='valid', name='conv6_1')(x)
    x = Reshape((-1,512))(x)
    
    if gru:
        x = Bidirectional(GRU(256, return_sequences=True))(x)
        x = Bidirectional(GRU(256, return_sequences=True))(x)
    
    else:
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
    
    x = Dense(num_classes, name='dense1')(x)
    
    # output은 softmax함수를 사용하여 라벨에대한 확률값이 나온다.
    x = y_pred = Activation('softmax', name='softmax')(x)
    
    #모델을 정의 : Model(input, output)
    model_pred = Model(image_input, x)
    
    # train모델이아닌 preiction 모델의 output은 softmax activation을 적용한 값
    if prediction_only:
        return model_pred

    #최대 글자수
    max_string_len = int(y_pred.shape[1])

    # CTC LOSS 함수 정의
    def ctc_lambda_func(args):
        labels, y_pred, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    # CTC LOSS를 계산할때 사용하는 INPUT 정의
    labels = Input(name='label_input', shape=[max_string_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Lambda를 사용하여 ctc loss 구한다
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([labels, y_pred, input_length, label_length])
    
    # 최종 학습모델의 인풋은 4가지이고, 아웃풋은 ctc loss 값
    model_train = Model(inputs=[image_input, labels, input_length, label_length], outputs=ctc_loss)
    
    return model_train, model_pred

model = CRNN()
