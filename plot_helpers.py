import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_loss_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def plot_image(x, y, label_names):
    figure(figsize=(1,1 ), dpi=80)
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(str(y[0]) + ": " + label_names[y[0]])
    plt.show()
