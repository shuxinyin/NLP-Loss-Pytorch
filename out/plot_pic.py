import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_data(file):
    df = pd.read_csv(file, header=0, sep='\t', encoding='utf-8')
    return df


def get_time(df):
    time_list = df['time'].to_list()
    return np.mean(time_list)


def draw_plot(l1, l2, l3, l4, l5, name='train_loss', location="upper right"):
    plt.plot(l1)
    plt.plot(l2)
    plt.plot(l3)
    plt.plot(l4)
    plt.plot(l5)
    legend = plt.legend(['Normal', 'FGSM', 'FGM', 'PGD', 'FreeAT'], title=name, loc=location)


if __name__ == '__main__':
    file_dic = {"Normal": "ATTypeFGM_UseATFalse.csv",
                "FGSM": "ATTypeFGSM_UseATTrue.csv",
                "FGM": "ATTypeFGM_UseATTrue.csv",
                "PGD": "ATTypePGD_UseATTrue.csv",
                "FreeAT": "ATTypeFreeAT_UseATTrue_epsilon0.8.csv"}

    df_normal = read_data(file_dic['Normal'])
    df_fgsm = read_data(file_dic['FGSM'])
    df_fgm = read_data(file_dic['FGM'])
    df_pgd = read_data(file_dic['PGD'])
    df_freeat = read_data(file_dic['FreeAT'])
    print(df_normal.head(3))

    t1, t2, t3, t4, t5 = get_time(df_normal), get_time(df_fgsm), \
                         get_time(df_fgm), get_time(df_pgd), \
                         get_time(df_freeat)
    print(t1, t2, t3, t4, t5)

    ax1 = plt.subplot(1, 2, 1)
    train_loss1 = df_normal['train_loss'].to_list()
    train_loss2 = df_fgsm['train_loss'].to_list()
    train_loss3 = df_fgm['train_loss'].to_list()
    train_loss4 = df_pgd['train_loss'].to_list()
    train_loss5 = df_freeat['train_loss'].to_list()

    draw_plot(train_loss1, train_loss2, train_loss3, train_loss4, train_loss5, name='train_loss')

    # eval_loss1 = df_normal['eval_loss'].to_list()
    # eval_loss2 = df_fgsm['eval_loss'].to_list()
    # eval_loss3 = df_fgm['eval_loss'].to_list()
    # eval_loss4 = df_pgd['eval_loss'].to_list()
    # eval_loss5 = df_freeat['eval_loss'].to_list()
    # draw_plot(eval_loss1, eval_loss2, eval_loss3, eval_loss4, eval_loss5, name='eval_loss')
    ax2 = plt.subplot(1, 2, 2)
    acc1 = df_normal['acc'].to_list()
    acc2 = df_fgsm['acc'].to_list()
    acc3 = df_fgm['acc'].to_list()
    acc4 = df_pgd['acc'].to_list()
    acc5 = df_freeat['acc'].to_list()
    draw_plot(acc1, acc2, acc3, acc4, acc5, name='acc', location="lower right")
    print(max(acc1), max(acc2), max(acc3), max(acc4), max(acc5))
    plt.show()
