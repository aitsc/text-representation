from collections import OrderedDict
from pprint import pprint
from scipy import stats


results = '''
TF-IDF & 0.1532 & 0.0571 & 0.0831 & 0.0773 & 0.1751 & 0.5666
avg-GloVe & 0.1747 & 0.0634 & 0.0931 & 0.0875 & 0.1983 & 0.5786
avg-word2vec & 0.2551 & 0.0913 & 0.1345 & 0.1492 & 0.2810 & 0.6241
WMD-GloVe & 0.1393 & 0.0514 & 0.0751 & 0.0660 & 0.1659 & 0.5603
WMD-word2vec & 0.2223 & 0.0827 & 0.1206 & 0.1291 & 0.2556 & 0.6095
LSA & 0.0289 & 0.0098 & 0.0146 & 0.0073 & 0.0312 & 0.4912
LDA & 0.0361 & 0.0132 & 0.0193 & 0.0093 & 0.0367 & 0.4944
doc2vec & 0.1272 & 0.0462 & 0.0677 & 0.0514 & 0.1420 & 0.5491
Doc2VecC & 0.2825 & 0.1030 & 0.1510 & 0.1727 & 0.3056 & 0.6358
CTPE-word2vec & 0.3066 & 0.1124 & 0.1645 & 0.1893 & 0.3233 & 0.6474
TF-IDF & 0.2564 & 0.3360 & 0.2908 & 0.1692 & 0.3257 & 0.6475
avg-GloVe & 0.1821 & 0.2387 & 0.2066 & 0.1084 & 0.2368 & 0.5974
avg-word2vec & 0.2163 & 0.2834 & 0.2453 & 0.1367 & 0.2786 & 0.6211
WMD-GloVe & 0.2225 & 0.2915 & 0.2524 & 0.1510 & 0.2993 & 0.6308
WMD-word2vec & 0.2527 & 0.3310 & 0.2866 & 0.1733 & 0.3311 & 0.6490
LSA & 0.0420 & 0.0549 & 0.0476 & 0.0188 & 0.0579 & 0.5039
LDA & 0.0220 & 0.0287 & 0.0249 & 0.0072 & 0.0267 & 0.4890
doc2vec & 0.1554 & 0.2036 & 0.1763 & 0.0857 & 0.2027 & 0.5790
Doc2VecC & 0.2579 & 0.3380 & 0.2926 & 0.1719 & 0.3239 & 0.6475
CTPE-word2vec & 0.2808 & 0.3680 & 0.3185 & 0.1846 & 0.3476 & 0.6595
TF-IDF & 0.1280 & 0.2419 & 0.1674 & 0.0745 & 0.1681 & 0.5624
avg-GloVe & 0.1036 & 0.1959 & 0.1355 & 0.0596 & 0.1383 & 0.5464
avg-word2vec & 0.1275 & 0.2419 & 0.1670 & 0.0710 & 0.1629 & 0.5607
WMD-GloVe & 0.1185 & 0.2226 & 0.1547 & 0.0719 & 0.1596 & 0.5577
WMD-word2vec & 0.1376 & 0.2605 & 0.1801 & 0.0837 & 0.1822 & 0.5707
LSA & 0.0350 & 0.0653 & 0.0456 & 0.0189 & 0.0487 & 0.4994
LDA & 0.0309 & 0.0589 & 0.0406 & 0.0141 & 0.0401 & 0.4954
doc2vec & 0.1014 & 0.1914 & 0.1326 & 0.0566 & 0.1350 & 0.5444
Doc2VecC & 0.1635 & 0.3094 & 0.2140 & 0.0964 & 0.2062 & 0.5841
CTPE-word2vec & 0.1889 & 0.3609 & 0.2480 & 0.1146 & 0.2381 & 0.6020
'''


def significance_test(x, y):
    '''
    t检验, 判断x是否等于y
    :param x: list; 一个总体, 比如对比算法
    :param y: list; 一个总体, 比如本算法
    :return:
    '''
    x_N = len(x)
    x_µ = sum(x) / x_N
    y_N = len(y)
    y_µ = sum(y) / y_N
    out = {'x-mean': x_µ, 'y-mean': y_µ}
    if x_N == y_N:
        xy = [i - j for i, j in zip(x, y)]  # 差值
        xy_µ = sum(xy) / x_N
        xy_D = sum([(xy_µ - i) ** 2 for i in xy]) / (x_N - 1)  # 样本方差
        if xy_D == 0:
            t_v = float('nan')
            tp_v = 1. if x_µ == y_µ else 0.
        else:
            t_v = (xy_µ - 0) / (xy_D / x_N) ** 0.5
            tp_v = min(1., (1 - stats.t.cdf(x=abs(t_v), df=x_N - 1)) * 2)
        out['t-value'] = t_v
        out['p-value'] = tp_v
        if tp_v > 0.05:
            out['describe'] = 'x和y没有显著性差异'
        else:
            if tp_v <= 0.001:
                p = 99.9
            elif tp_v <= 0.005:
                p = 99.5
            elif tp_v <= 0.01:
                p = 99
            elif tp_v <= 0.025:
                p = 97.5
            else:
                p = 95
            if x_µ > y_µ:
                out['describe'] = 'x大于y的置信度在' + str(p) + '%以上'
            else:
                out['describe'] = 'x小于y的置信度在' + str(p) + '%以上'
    return out


if __name__ == '__main__':
    method_rets_D = OrderedDict()
    for line in results.strip().split('\n'):
        line = line.split(' & ')
        method_rets_D.setdefault(line[0], []).extend([float(i) for i in [line[3], line[5]]])
    # check
    len_ret = None
    for k, v in method_rets_D.items():
        if len_ret is None:
            len_ret = len(v)
        assert len(v) == len_ret
    pprint(method_rets_D)
    print()
    my = 'CTPE-word2vec'
    for k in method_rets_D.keys():
        if k == my:
            continue
        out = significance_test(method_rets_D[k], method_rets_D[my])
        print(f"{k} & {round(out['x-mean'], 4)} & {round(out['t-value'], 4)} & {round(out['p-value'], 4)} & {round(1-out['p-value'], 4)} \\\\")
        my_mean = out['y-mean']
    print('my_mean:', round(my_mean, 4))
