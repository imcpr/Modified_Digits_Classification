# coding=utf-8
# Authors:
#   Yann Long
#
# Coding began novembre 2, 2015

def heatmap(predictions, actual, filename):
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt

    cm = confusion_matrix(actual, predictions)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    i = 0
    while os.path.exists('{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(filename, i))