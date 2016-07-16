from __future__ import division

import numpy as np
import pymc3 as pm

from scipy import stats
from collections import defaultdict

def post_summary(sample, cred_mass=0.95, cmp_val=None, rope=None):
    post = defaultdict(float)
    #
    post['mean']   = np.mean(sample)
    post['std']    = np.std(sample)
    #
    post['median'] = np.median(sample)
    post['mode']   = stats.mode(sample)[0]
    #
    post['mc_error'] = pm.stats.mc_error(sample)
    #
    hdi = pm.stats.hpd(sample, 1-cred_mass)
    post['cred_mass'] = cred_mass
    post['hdi'] = hdi
    #
    if cmp_val is not None:
        pct_gt = float(np.sum(sample>cmp_val))/len(sample)
        pct_lt = 1 - pct_gt
        post['cmp_val'] = cmp_val
        post['pct_cmp_val'] = (pct_lt, pct_gt)
    #
    if rope is not None:
        pct_in_rope = float(np.sum((sample>rope[0]) & (sample<rope[1])))/len(sample)
        post['rope'] = rope 
        post['pct_in_rope'] = pct_in_rope
        post['decision'] = decision(hdi, rope)
    #
    return post

def interval_str(interval):
    return "[{:+0.03f}, {:+0.03f}]".format(interval[0], interval[1])
    
def pct_cmp_val_str(cmp_val, pct_cmp_val):
    pct_lt, pct_gt = pct_cmp_val
    if cmp_val == 0:
        return "{:.1%} < {:.0f} < {:.1%}".format(pct_lt, cmp_val, pct_gt)
    else:
        return "{:.1%} < {:.2f} < {:.1%}".format(pct_lt, cmp_val, pct_gt)

def decision(hdi, rope):
    if rope[0] <= hdi[0] <= hdi[1] <= rope[1]:
        return '='
    if hdi[0] <= hdi[1] <= rope[0] <= rope[1]:
        return '<<'
    if rope[0] <= rope[1] <= hdi[0] <= hdi[1]:
        return '>>'
    if (hdi[0]+hdi[1])/2 < (rope[0]+rope[1])/2:
        return '<'
    if (hdi[0]+hdi[1])/2 > (rope[0]+rope[1])/2:
        return '>'
    return '?'

def micro_s_test(scores_a, scores_b):
    p_value = stats.binom_test([sum(scores_a<scores_b), sum(scores_a>scores_b)])
    return p_value

def micro_t_test(scores_a, scores_b):
    t, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
    return p_value

def macro_s_test(scores_a, scores_b):
    p_value = stats.binom_test([sum(scores_a<scores_b), sum(scores_a>scores_b)])
    return p_value

def macro_t_test(scores_a, scores_b):
    t, p_value = stats.ttest_rel(scores_a, scores_b)
    return p_value

def bfsd(prio_sample, post_sample, cmp_val=0):
    prio_pdf_kde = stats.kde.gaussian_kde(prio_sample)
    post_pdf_kde = stats.kde.gaussian_kde(post_sample)
    bayes_factor = post_pdf_kde(cmp_val)[0] / prio_pdf_kde(cmp_val)[0]
    return bayes_factor
    
def post_analysis(trace, var, burn=0, cred_mass=0.95, cmp_val=None, rope=None):
    post = post_summary(trace[burn:][var], cred_mass, cmp_val, rope)
    analysis = "{:s}".format(var)
    analysis += "\t{:+0.3f}\t{:0.3f}".format(post['mean'], post['std'])
    analysis += "\t{:0.3f}".format(post['mc_error'])
    if cmp_val is not None:
        analysis += "\t{:s}".format(pct_cmp_val_str(cmp_val, post['pct_cmp_val']))
    if rope is not None:
        analysis += "\t{:5.1%}".format(post['pct_in_rope'])
    analysis += "\t{:s}".format(interval_str(post['hdi']))
    if rope is not None:
        analysis += "\t{:s}".format(post['decision'])
    return analysis
