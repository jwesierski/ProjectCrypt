from pylsl import StreamInlet, resolve_stream
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque
import cv2
import os
import random
import tensorflow as tf

#script being used to test a model and make new data. While the model is running it will be saving what you are thinking
MODEL_NAME = "C:\\Users\\wesie\\OneDrive\\Desktop\\ProjectCrypt\\new_models\\41.3-acc-64x3-batch-norm-9epoch-1647915410-loss-4.97.model" # your model path here.
#comes in chunks of 125-125 data points per channel, 16 channels. 16 channels of 125 data points. We keep cycling through that
model = tf.keras.models.load_model(MODEL_NAME)
reshape = (-1, 16, 60)
#reshape = (-1, 16, 128, 1)

model.predict( np.zeros((32,16,60)).reshape(reshape) )

ACTION = 'left' # THIS IS THE ACTION YOU'RE THINKING
#because of nyquist theory even though we get about 125 data ponts we only want to use 1/2
FFT_MAX_HZ = 60

HM_SECONDS = 10  # this is approximately 10 seconds of thinking this direction (though the stream ends up being closer to 40 for some reason)
TOTAL_ITERS = HM_SECONDS*25  # ~25 iters means 25 frames of 125 datapoints from hz (truncated to 60). Each file holds these 25 seconds
BOX_MOVE = "model"  # random or model

last_print = time.time()
fps_counter = deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

WIDTH = 800
HEIGHT = 800
SQ_SIZE = 50
MOVE_SPEED = 1

square = {'x1': int(int(WIDTH)/2-int(SQ_SIZE/2)),
          'x2': int(int(WIDTH)/2+int(SQ_SIZE/2)),
          'y1': int(int(HEIGHT)/2-int(SQ_SIZE/2)),
          'y2': int(int(HEIGHT)/2+int(SQ_SIZE/2))}


box = np.ones((square['y2']-square['y1'], square['x2']-square['x1'], 3)) * np.random.uniform(size=(3,))
horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))

total = 0
left = 0
right = 0
none = 0
correct = 0

channel_datas = []

for i in range(TOTAL_ITERS):  # how many iterations. Eventually this would be a while True
    channel_data = []
    for i in range(16): # each of the 16 channels here
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[:FFT_MAX_HZ])

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)

    env = np.zeros((WIDTH, HEIGHT, 3))

    env[:,HEIGHT//2-5:HEIGHT//2+5,:] = horizontal_line
    env[WIDTH//2-5:WIDTH//2+5,:,:] = vertical_line
    env[square['y1']:square['y2'], square['x1']:square['x2']] = box

    cv2.imshow('', env)
    cv2.waitKey(1)
#checking if we got action right or wrong
    network_input = np.array(channel_data).reshape(reshape)
    out = model.predict(network_input)
    print(out[0])

    if BOX_MOVE == "random":
        move = random.choice([-1,0,1])
        square['x1'] += move
        square['x2'] += move

    elif BOX_MOVE == "model":
        choice = np.argmax(out)
        if choice == 0:
            if ACTION == "left":
                correct += 1
            square['x1'] -= MOVE_SPEED
            square['x2'] -= MOVE_SPEED
            left += 1

        elif choice == 2:
            if ACTION == "right":
                correct += 1
            square['x1'] += MOVE_SPEED
            square['x2'] += MOVE_SPEED
            right += 1

        else:
            if ACTION == "none":
                correct += 1
            none += 1

    total += 1


    channel_datas.append(channel_data)

#plt.plot(channel_datas[0][0])
#plt.show()

datadir = "data"
#if not os.path.exists(datadir):
#    os.mkdir(datadir)
#here we save the data
actiondir = f"C:\\Users\\wesie\\OneDrive\\Desktop\\ProjectCrypt\\BCI-master\\data_V3\\{datadir}\\{ACTION}"
if not os.path.exists(actiondir):
    os.mkdir(actiondir)

print(len(channel_datas))

print(f"saving {ACTION} data...")
np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
print("done.")

for action in ['left', 'right', 'none']:
    #print(f"{action}:{len(os.listdir(f'data/{action}'))}")
    print(action, sum(os.path.getsize(f'C:\\Users\\wesie\\OneDrive\\Desktop\\ProjectCrypt\\BCI-master\\data_V3\\{datadir}\\{action}\\{f}') for f in os.listdir(f'C:\\Users\\wesie\\OneDrive\\Desktop\\ProjectCrypt\\BCI-master\\data_V3\\{datadir}\\{action}'))/1_000_000, "MB")

print(ACTION, correct/total)
print(f"left: {left/total}, right: {right/total}, none: {none/total}")

with open("accuracies.csv", "a") as f:
    f.write(f"{int(time.time())},{ACTION},{correct/total},{MODEL_NAME},{left/total},{right/total},{none/total}\n")
