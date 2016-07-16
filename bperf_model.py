from __future__ import division

import numpy as np
import pymc3 as pm

def build_model_paired(y_true, y_pred_a, y_pred_b, model_type=3):
    N = len(y_true)
    out = list(zip(y_pred_a, y_pred_b))
    pos, neg = True, False
    out_map = {
        (neg, neg): 0,
        (neg, pos): 1,
        (pos, neg): 2,
        (pos, pos): 3,
    }
    out_p = np.array([out_map[out[i]] for i in range(N) if y_true[i] == pos])
    out_n = np.array([out_map[out[i]] for i in range(N) if y_true[i] == neg])
    cnt_p = [sum(out_p==k) for k in range(len(out_map))]
    cnt_n = [sum(out_n==k) for k in range(len(out_map))]
    num_p = sum(y_true)
    num_n = N - num_p
    #
    with pm.Model() as model:
        if model_type == 0 or model_type == 1 or model_type == 2:
            # define the prior
            mu = pm.Beta('mu', 1.0, 1.0)
            alpha = np.repeat(1.0, 4)
            theta_p = pm.Dirichlet('theta_p', a=alpha, shape=4)
            theta_n = pm.Dirichlet('theta_n', a=alpha, shape=4)
            # define the likelihood
            if model_type == 1:
                y = pm.Bernoulli('y', p=mu, observed=y_true)
                o_p = pm.Categorical('o_p', p=theta_p, observed=out_p)
                o_n = pm.Categorical('o_n', p=theta_n, observed=out_n)
            elif model_type == 2:
                n_p = pm.Binomial('n_p', n=N, p=mu, observed=num_p)
                c_p = pm.Multinomial('c_p', n=num_p, p=theta_p, observed=cnt_p)
                c_n = pm.Multinomial('c_n', n=num_n, p=theta_n, observed=cnt_n)
        elif model_type == 3:
            # define the posterior directly
            mu = pm.Beta('mu', 1+num_p, 1+num_n)
            alpha = np.repeat(1.0, 4)
            theta_p = pm.Dirichlet('theta_p', a=alpha+cnt_p, shape=4)
            theta_n = pm.Dirichlet('theta_n', a=alpha+cnt_n, shape=4)
        #
        nu = 1 - mu
        F1 = lambda P, R: 2.0*(P*R)/(P+R)
        #
        tp_a = mu * (theta_p[out_map[(pos, neg)]] + theta_p[out_map[(pos, pos)]])
        fn_a = mu * (theta_p[out_map[(neg, neg)]] + theta_p[out_map[(neg, pos)]])
        fp_a = nu * (theta_n[out_map[(pos, neg)]] + theta_n[out_map[(pos, pos)]])
        tn_a = nu * (theta_n[out_map[(neg, neg)]] + theta_n[out_map[(neg, pos)]])
        P_a = tp_a/(tp_a+fp_a)
        R_a = tp_a/(tp_a+fn_a)
        F1_a = pm.Deterministic('F1_a', F1(P_a,R_a))
        tp_b = mu * (theta_p[out_map[(neg, pos)]] + theta_p[out_map[(pos, pos)]])
        fn_b = mu * (theta_p[out_map[(neg, neg)]] + theta_p[out_map[(pos, neg)]])
        fp_b = nu * (theta_n[out_map[(neg, pos)]] + theta_n[out_map[(pos, pos)]])
        tn_b = nu * (theta_n[out_map[(neg, neg)]] + theta_n[out_map[(pos, neg)]])
        P_b = tp_b/(tp_b+fp_b)
        R_b = tp_b/(tp_b+fn_b)
        F1_b = pm.Deterministic('F1_b', F1(P_b,R_b))
        #
        delta = pm.Deterministic('delta', F1_a-F1_b)
    return model

def build_model_unpaired(y_true, y_pred_a, y_pred_b):
    N = len(y_true)
    con_a = list(zip(y_true, y_pred_a))
    con_b = list(zip(y_true, y_pred_b))
    pos, neg = True, False
    con_map = {
        (neg, neg): 0,
        (neg, pos): 1,
        (pos, neg): 2,
        (pos, pos): 3,
    }
    out_a = np.array([con_map[con_a[i]] for i in range(N)])
    out_b = np.array([con_map[con_b[i]] for i in range(N)])
    cnt_a = [sum(out_a==k) for k in range(len(con_map))]
    cnt_b = [sum(out_b==k) for k in range(len(con_map))]
    #num_p = sum(y_true)
    #num_n = N - num_p
    #
    with pm.Model() as model:
        # define the posterior directly
        #mu = pm.Beta('mu', 1+num_p, 1+num_n)
        alpha = np.repeat(1.0, 4)
        theta_a = pm.Dirichlet('theta_a', a=alpha+cnt_a, shape=4)
        theta_b = pm.Dirichlet('theta_b', a=alpha+cnt_b, shape=4)
        #
        #nu = 1 - mu
        F1 = lambda P, R: 2.0*(P*R)/(P+R)
        #
        tp_a = theta_a[con_map[(pos, pos)]]
        fn_a = theta_a[con_map[(pos, neg)]]
        fp_a = theta_a[con_map[(neg, pos)]]
        tn_a = theta_a[con_map[(neg, neg)]]
        P_a = tp_a/(tp_a+fp_a)
        R_a = tp_a/(tp_a+fn_a)
        F1_a = pm.Deterministic('F1_a', F1(P_a,R_a))
        tp_b = theta_b[con_map[(pos, pos)]]
        fn_b = theta_b[con_map[(pos, neg)]]
        fp_b = theta_b[con_map[(neg, pos)]]
        tn_b = theta_b[con_map[(neg, neg)]]
        P_b = tp_b/(tp_b+fp_b)
        R_b = tp_b/(tp_b+fn_b)
        F1_b = pm.Deterministic('F1_b', F1(P_b,R_b))
        #
        delta = pm.Deterministic('delta', F1_a-F1_b)
    return model

def learn_model(model, draws=50000):
    with model:
        start = pm.find_MAP()
        #step = pm.Slice()  # It is very slow when the model has many parameters
        #step = pm.HamiltonianMC(scaling=start)  # It leads to constant samples
        #step = pm.NUTS(scaling=start)           # It leads to constant samples
        step = pm.Metropolis()
        trace = pm.sample(draws, step, start=start)
    return trace
