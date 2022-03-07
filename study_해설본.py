import tensorflow as tf, numpy as np, matplotlib.pyplot as plt, csv, pandas as pd

# sunspots -> 흑점

# data = pd.read_csv('sunspots.csv')
# print(data.shape)       # (3235,3)
# print(data[:-5])         # 컬럼은 3개인데 이름없는 컬럼도 있고 뭐가 많다.
# Unnamed: 0        Date  Monthly Mean Total Sunspot Number 이게 뭘까


def plot_series(time, series, format='-', start=0, end=None):
    plt.figure(figsize=(10,6))
    plt.plot(time[start:end],series[start:end],format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    # plt.show()
    
time_step = []                                  # 3235
sunspots = []                                   # 3235

with open('sunspots.csv') as csvfile:           # csv파일을 csvfile이라는 변수명으로 열고
    reader = csv.reader(csvfile, delimiter=',') # csv.reader로 변수를 읽고 reader에 담는다.
    # print(type(reader))                         현재 reader은 <class '_csv.reader'> type이고 읽을수도(<_csv.reader object at 0x0000013F8F0C8940>) for문 돌릴수도 없다.
    next(reader)                                # 반복문 돌리기 위해서 next()로 reader의 0행을 넘기고 그 다음값부터 반환해준다.
    for row in reader:
        sunspots.append(float(row[2]))
        time_step.append(int(row[0]))
        
series = np.array(sunspots)                     # 위에서 만든 리스트를 넘파이형태로 바꿔준다.
time = np.array(time_step)                      # 위에서 만든 리스트를 넘파이형태로 바꿔준다.
plot_series(time,series)                        # 위에서 만들어놓은 함수 실행.

# 시계열데이터는 train과 test(또는 valid)를 나눌때 랜덤하게 나눠주면 안된다. 시간의 흐름이 유지되게 나눠줘야한다.
split_time = 3000                               # train, valid 나눠주는 시점을 의미.
time_train = time[:split_time]                  # 0부터 3000까지 -> index번호이므로 2999까지가 담긴다.
x_train = series[:split_time]                   # 상동.
time_valid = time[split_time:]                  # 3000부터 마지막까지.
x_valid = series[split_time:]                   # 상동.

window_size = 30
batch_size = 32
shuffle_buffer_size = 1000

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)                            # numpy형태가 된 series의 차원을 늘려준다. (3235,) -> (3235,1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

train_set = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
print(train_set)
print(x_train.shape)


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1, padding='causal', activation='relu', input_shape=[None,1]),
        tf.keras.layers.LSTM(64, return_sequences=True),              
        tf.keras.layers.LSTM(64, return_sequences=True),              
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)])

model.summary()