import numpy as np
import itertools
import matplotlib.pyplot as plt
from DataAssignment.src.py.config.config import PATH_PLOTS
from sklearn.metrics import roc_auc_score, auc, precision_score, roc_curve, accuracy_score, confusion_matrix


def create_list_dict(list_in):
    """
    Function to create a dictionary of index-values based on any given list

    Parameters
    -----------
    list_in: list
        list from which extract dictionary

    Returns
    -----------
    dict
    """
    # dictionary init
    dict_out = {}

    # dictionary creation
    for i in list_in:  # go over the list
        if i not in dict_out.keys():  # checking if value already in dict
            dict_out[i] = len(dict_out) + 1  # if not, add it
        else:
            pass
    # eof
    return dict_out


def cumulative_gain(y_true, y_score):
    """Generate the cumulative gain chart
        y_true:  true values
        y_score: predicted values
    """
    pos_label = 1
    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true.index = [i for i in range(0, len(desc_score_indices))]
    y_true = y_true[desc_score_indices]

    total_p = sum(y_true)

    weight = 1

    # cg: Cumulative gain
    cg_pop_perc = np.linspace(0.0, 1.0, num=len(y_true))
    cg_cap_perc = (y_true * weight).cumsum() / (total_p + 0.0)
    cg_cap_perc.index = [i for i in range(0, len(cg_cap_perc))]
    cg_wizard_perc = np.array([min(i + 1, total_p) for i in range(0, len(y_score))]) / (total_p + 0.0)
    return cg_pop_perc, cg_cap_perc, cg_wizard_perc


def roc_charts(y_true, y_score, plot_name):
    """
    Compute the ROC curve, ROC area, CG curve, and CG area.
    :param y_true:  true values
    :param y_score: predicted values
    :param plot_name: string with model name
    :return:
    """

    def _get_decile(cg_cap_perc, cg_pop_perc, decile):
        interval, = np.where((cg_pop_perc >= decile) & (cg_pop_perc < decile + 0.01))
        gain = round(np.mean(cg_cap_perc[interval]), 2)
        return interval, gain

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr[1], tpr[1], _ = roc_curve(y_true, y_score)
    roc_auc[1] = auc(fpr[1], tpr[1])

    # Plot of a ROC curve for a specific class
    plt.figure()

    ax = plt.subplot(1, 2, 1)
    plt.plot(fpr[1], tpr[1], label='ROC (area = %0.3f)' % roc_auc[1], color='#ee503b')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Cumulative gain
    ax = plt.subplot(1, 2, 2)
    cg_pop_perc, cg_cap_perc, cg_wizard_perc = cumulative_gain(y_true, y_score)
    cg_auc = auc(cg_pop_perc, cg_cap_perc)

    # Plot of a CG curve for a specific class
    plt.plot(cg_pop_perc, cg_cap_perc, label='CG (a = %0.3f)' % cg_auc, color='#ee503b')
    plt.plot(cg_pop_perc, cg_wizard_perc, label='Wizard', color='#636360')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('% of population')
    plt.ylabel('% of captured response (Total: {0})'.format(np.bincount(y_true)[1]))

    # calculate decile 0.1 value:
    interval, dec_01 = _get_decile(cg_cap_perc, cg_pop_perc, 0.1)
    plt.plot(0.1, dec_01, 'o', color='#37a75a')
    plt.annotate(str(dec_01), xy=(0.1, dec_01), color='#636360')

    # calculate decile 0.2 value:
    interval, dec_02 = _get_decile(cg_cap_perc, cg_pop_perc, 0.2)
    plt.plot(0.2, dec_02, 'o', color='#37a75a')
    plt.annotate(str(dec_02), xy=(0.2, dec_02), color='#636360')

    # calculate decile 0.3 value:
    interval, dec_03 = _get_decile(cg_cap_perc, cg_pop_perc, 0.4)
    plt.plot(0.4, dec_03, 'o', color='#37a75a')
    plt.annotate(str(dec_03), xy=(0.4, dec_03), color='#636360')

    plt.title('Cumulative gain')
    plt.legend(loc="lower right")

    plt.subplots_adjust(left=.02, right=.98)
    plt.tight_layout()
    plt.savefig(PATH_PLOTS + '{0}_roc.png'.format(plot_name))
    plt.close()
    return roc_auc[1], cg_auc


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        pass

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(PATH_PLOTS + 'confusion_matrix.png')
    plt.close()
