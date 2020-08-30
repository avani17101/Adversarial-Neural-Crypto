class TimeHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.times = []

    def on_epoch_begin(self, batch, logs={}):

        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):

        self.times.append(time.time() -                        self.epoch_time_start)
            
            


time_callback = TimeHistory()

model.fit(..., callbacks=[..., time_callback],...)

times = time_callback.times
            
            
# https://github.com/keras-team/keras/issues/2850

# https://stackoverflow.com/questions/39124676/show-progress-bar-for-each-epoch-during-batchwise-training-in-keras
