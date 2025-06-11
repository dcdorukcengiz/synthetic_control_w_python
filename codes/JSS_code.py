#%%

import pandas as pd
import numpy as np
import os
from plotnine import ggtitle
import sys
import io as IO
from contextlib import redirect_stdout


from scpi_pkg.scdata import scdata
from scpi_pkg.scdataMulti import scdataMulti
import scpi_pkg.scest as scest
from scpi_pkg.scpi import scpi
from scpi_pkg.scplot import scplot
from scpi_pkg.scplotMulti import scplotMulti
from IPython.utils import io



data = pd.read_csv("Data/scpi_germany.csv")
np.random.seed(12345)

id_var = "country"
outcome_var = "gdp"
time_var = "year"
period_pre = np.arange(1960, 1991)
period_post = np.arange(1991, 2004)
unit_tr = "West Germany"
unit_co = list(set(data[id_var].to_list()))
unit_co = [cou for cou in unit_co if cou != "West Germany"]
constant = True
cointegrated_data = True



data_prep = scdata(df = data, id_var = id_var, time_var = time_var,
 outcome_var = outcome_var, period_pre = period_pre,
 period_post = period_post, unit_tr = unit_tr,
 unit_co = unit_co, cointegrated_data = cointegrated_data,
 constant = constant)

with io.capture_output() as captured:
    with redirect_stdout(IO.StringIO()):
        est_si = scest.scest(data_prep, w_constr={"name": "simplex"})


e_method = "gaussian"
plot = scplot(est_si, x_lab = "Year", e_method = e_method,
 y_lab = "GDP per capita (thousand US dollars)")
plot + ggtitle("")


weights_and_constant = est_si.b.reset_index()


w_constr = {"name": "simplex", "Q": 1}
u_missp = True
u_sigma = "HC1"
u_order = 1
u_lags = 0
e_method = "gaussian"
e_order = 1
e_lags = 0
sims = 1000
cores = 1


est_lasso2 = scest.scest(data_prep, w_constr={'p': 'L1', 'dir': '<=', 'Q': 1, 'lb': -np.inf})




for mtd in ["ridge", ""]:
    np.random.seed(8894)
    pi_si = scpi(data_prep, sims = sims, w_constr = {"name": mtd},
    u_order = u_order, u_lags = u_lags, e_order = e_order,
    e_lags = e_lags, e_method = e_method, u_missp = u_missp,
    u_sigma = u_sigma, cores = cores)
plot = scplot(pi_si, x_lab = "Year", e_method = e_method,
    y_lab = "GDP per capita (thousand US dollars)")
plot + ggtitle("")
