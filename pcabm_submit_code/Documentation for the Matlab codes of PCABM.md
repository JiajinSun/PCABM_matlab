#### Documentation for the Matlab codes of PCABM

#### The functions for the model and the algorithms

The folder `cpl_m_code` contains functions for the main algorithms as well as model definition, model generation, and error calculation. Among them, the important ones are:

`CAdcBlkMod.m`: defines the covariate adjusted degree corrected block model.

`CAgenCAdcBM1.m`: generates an observed adjacency matrix from the covariate adjusted degree corrected block model.

`CA_estimgamma.m`: calculates the MLE for $\pmb\gamma$ given an estimate of community assignment $\mathbf e_0$, as is described in **equation (2)** in the paper.

`CA_SCWA.m`: the function for the spectral clustering with adjustment algorithm (**Algorithm 2** in the paper). While all the arguments and options are explained in comments in the beginning of the file itself, the most important ones to note is `'perturb'` and `'SC_r'` in options. `'perturb'` = false: no regularization; `'perturb'` = true, `'SC_r'` = false: regularization of the form in Amini et al., 2013; `'perturb'` = true, `'SC_r'` = true: regularization of the form in this paper.

`CA_plEM.m`: the function for the pseudo likelihood EM algorithm (**Algorithm 1** in the paper).  All the arguments and options are explained in comments in the beginning of the file itself.

​              

#### Simulations

The codes for simulations are in the main folder, and are explained as follows:

`PCABM_demo.m`: The code for simulations in **Section 7.1 and 7.2** of the paper. The beginning of the code file sets up the parameter settings of $n,c_\rho,c_\gamma$, etc.. For each setting, `NUM_rep` = 100 realizations are simulated. For the simulation of **Section 7.1**: The estimation of $\pmb\gamma$ (of the 100 realizations of 1 parameter setting) is stored in the variable `gammahat`, and its mean and variance over 100 realizations is calculated and printed in the variable `result_gamma`. These results correspond to **Figure 2** and **Table 1** in the paper. For the simulation of **Section 7.2**: The NMI and ARI for the SCWA algorithm are stored in the variables `init_nmi` and `init_ari`, while the NMI and ARI for the PLEM algorithm are stored in the variables `CAplEM_nmi` and `CAplEM_ari`. Their mean and variance over the 100 realizations are calculated and printed in the variables `result_err` and `result_std`. These results correspond to **Figure 3(a)(b)(c)** in the paper.

`PCABM_demo_initaccu.m`: The code for simulations in **Section 7.3** of the paper. Everything is the same as in `PCABM_demo.m`, except that we control the initial accuracy at a certain level (line 91 in the code), and track the accuracy of SCWA and PLEM in addition to NMI and ARI. The variables `init_gen_accu`, `SCWA_accu` and `CAplEM_accu` store the accuracies of the initial estimate, the SCWA estimate and the PLEM estimate, respectively. The results of this file correspond to **Figure 3(d)** in the paper.

`PCABM_DCBM_demo.m`: The code for simulations in **Section 7.4** of the paper. The framework of the code is the same as in `PCABM_demo.m`; the differences are: we set the degree correction parameters as $\{1,4\}$ with equal probability of 0.5 (in line 13); we construct the covariates as $z_{ij} = \log(d_id_j) $ (in line 52). The results are, again, stored in the variables `result_gamma`, `result_err` and `result_std`. Results of this file correspond to **Figure 4** in the paper.

`ecv_chooseK.m`: The code for simulations in **Section 7.5** of the paper. For each true model generated (`realization` in line 32-54), the node pairs are subsampled for `Nrep` folds (line 72-78); for each fold, the model is fitted with different $k$'s (line 80-105), and then each fitted model is evaluated on the held-out test set (line 107-108). The variables `Khat_lik_scaled` and `Khat_se` store the $k$ chosen by scaled likelihood and scale $L_2$ loss in each realization. The number of times each $k$ is selected in the 100 realizations are printed as tabulate in the output of the file, whose results correspond to **Table 2** in the paper.

`ecv_variableselect_sppB.m`: The code for simulations in **Section B** of the supplementary material. The code itself is run in parallel; each loop in the `parfor` in line 35 corresponds to a value of $r$, the correlation between the false covariate $Z'$ and $P = [B_{c_ic_j}]$. Then, for each realization of generated model (line 44-60), we go over all the not yet selected variables (line 74-145) until adding any variable to the selected variable set (`Seled`) does not increase the test set likelihood by a threshold. `Shat_lik(realization,:)` is the set of indices of variables selected in this realization, whose result correspond to **Table B.2** in the supplementary material. After that we compare 5 cases: 1. adjusting for all covariates, 2. adjusting for only the true covariate, 3. adjusting for the selected covariates, 4. adjusting for no covariates, 5. adjusting for only the false covariate. The variable `gamh_specresult` stores the $\hat{\pmb\gamma}$ for all the 5 cases; the average $\hat{\pmb\gamma}$ when adjusting for all covariates is calculated in `gamhmean_overr(r_ind,1,:)`, and corresponds to **Table B.1** in the supplementary material. The variables `MuI_specresult` and `ARI_specresult` store the clustering accuracies for all 5 cases; their averages are calculated in `MuI_overr` and `ARI_overr`, which correspond to **Figure 6** in the supplementary material. 

(For the convenience of future users, each file only runs the simulation of **one** setting for 100 times.)



#### Real Data

`pbdata_demo.m`: The code for the analysis of the political blog data in **Section 8.1**. The output gives the number of misclassified nodes, ARI, and NMI for SCWA and PLEM, respectively.

