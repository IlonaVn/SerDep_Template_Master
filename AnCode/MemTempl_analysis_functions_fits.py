def glm_fit(data, bins, bin_sel):
    """
    Function to fit a binomial curve to data (specifically judgment task response data).
    
    :param data: single-subject response distribution data depending on distance between Gabor and Template.
    :param bins: How many bins judgment task response data is averaged into.
    :param bin_sel: Which of these bins should be used to fit binomial.
    
    """
    import numpy as np
    import statsmodels.api as sm
    
    conditions = []
    
    for i in range(len(data)):
        conditions.append(data[i,:])
    
    glm_binom = []
    res = []
    y_preds = np.zeros((len(conditions), len(bins[bin_sel[0]:bin_sel[1]])))
    
    for condi, cond in enumerate(conditions):
        glm_binom_tmp = sm.GLM(cond[bin_sel[0]:bin_sel[1]], bins[bin_sel[0]:bin_sel[1]], family=sm.families.Binomial())
        res_tmp = glm_binom_tmp.fit()
        print(res_tmp.summary())
        y_preds[condi, :] = res_tmp.fittedvalues
        glm_binom.append(glm_binom_tmp)
        res.append(res_tmp)
    
    return glm_binom, res, y_preds

def Derivative_of_Gaussian(x, a, w): #taken from Darinka
    """
    :param x: in the context of SD, the relative position of the past trial
    :param a: in the context of SD, the amplitude of the curve, the parameter to be estimated
    :param w: in the context of SD, the width of the curve, treated here as a free parameter
    
    """ 
    import numpy as np
    
    c = np.sqrt(2) / np.exp(-0.5) #constant associated with the DOG function
    
    return x * a * w * c * np.exp(-(w*x)**2)


def compute_SSE(y_pred, y): #SSE (Sum of squares due to error, the smaller, the better) ### from Darinka 
    
    """
    :param y_pred: predicted values
    :param y: actual values
    """ 
    import numpy as np

    return np.sum((y-y_pred)**2)

def fit_DoG(y, x, fittingSteps, plot= False, ax = None, rangelim = 90, SD_type = ''): #modified from Darinka
    
    """
    :param y: observations to be fit, in the context of SD: response errors
    :param x: predictor, in the context of SD: relative angular distance
    :param fittingSteps: how many iterations of the fitting procedure to perform
    :param plot: Whether to plot for control.
    :param ax: If true, control plot is plotted onto existing figure.
    """ 
    import numpy as np
    from scipy.optimize import least_squares
    from MemTempl_analysis_functions_fits import compute_SSE, Derivative_of_Gaussian
    
    y_temp = y.copy()

    if (SD_type == 'SD_resp') & (rangelim == 90):
        first_lim = int(90-45)
        last_lim = int(45-90)
        y_temp[first_lim:last_lim] = 0
    
    #We need to minimize the residuals (aka, data-model)
    def _solver(params):
        a, w = params
        return y_temp-Derivative_of_Gaussian(x, a, w)
    
    #Initialize bookkeeping variables
    gof = np.zeros((5)) #to store measures of goodness of fit
    
    #Range of plausible values to be tried for amplitude parameter
    min_a = -10
    max_a = 10
    
    #Range of values to be tried for width parameter
    if rangelim == 0:
        min_w = 0.02
        max_w = 0.2
        
    elif rangelim > 45:
        min_w = 0.01 #a la Fritsche 2017
        max_w = 0.04 

    elif rangelim <= 45:
        min_w = 0.1
        max_w = 0.4
    
    min_cost = np.inf
    
    x = np.linspace(-90,90,181)

    for i in range(fittingSteps):
        #Determine random starting positions of parameters within specified range
        params_0 = [np.random.rand() * (max_a - min_a) + min_a,
                    np.random.rand() * (max_w - min_w) + min_w] 
        
        try:
            result = least_squares(_solver, params_0, bounds=([min_a, min_w], [max_a, max_w]))
        except ValueError:
            continue
        
        #Check whether the residual error is smaller than the previous one
        if result['cost'] < min_cost:
            best_params, min_cost, y_res = result['x'], result['cost'], result.fun
    
    #after best parameters have been determined, generate data for curve and determine fit with real data
    y_pred = Derivative_of_Gaussian(x, best_params[0], best_params[1])
    gof[0] = compute_SSE(y_pred, y_temp) #SSE
    
    if plot:
        y_model = Derivative_of_Gaussian(x, best_params[0],best_params[1] )
        ax.plot(x, y_model, c='k', linestyle='None', marker='o')#, label = parameter_info) #optimized DoG starting witih current parameters
    try:
        return best_params[0], best_params[1], min_cost, gof
    except UnboundLocalError:
        return np.nan, np.nan, min_cost, gof

def return_dogparams(data, xrange = 90, rangelim = 90, plot = False, fittingsteps = 5):
    """
    Receives arrays with SD curve data, returns arrays of same length with optimal DoG parameters and goodness of fit parameters.
    
    :param xrange: range over which DoG is plotted
    :param rangelim: range over which goodness of fit is determined, also the range defining DoG initial width
    :param plot: Whether to plot for control.
    :param fittingsteps: How many trials to fit DoG curve to data to perform.
    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from MemTempl_analysis_functions_fits import fit_DoG, Derivative_of_Gaussian
    
    if len(data) > 30:
        plot = False
    
    #determine shape of variable storing DoG parameters (depending on number of curves to be fit)
    shapes = np.shape(data)
    dim3or2 = len(shapes)
    
    if dim3or2 == 3: #if the incoming data is 3D, then there is more than 1 subject
        dog_params = np.zeros((shapes[0],shapes[1],4))
        xrange = np.linspace(-xrange,xrange, len(data[0,0,:]))
        rangesubs = shapes[0]
        rangepasses = shapes[1]

    elif dim3or2 == 2: #if incoming data is 2D, that means that there is only one subject's data
        dog_params = np.zeros((shapes[0],4))
        rangesubs = 1
        rangepasses = shapes[0]
    
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot()
        
    for i in range(rangesubs):
        for k in range(rangepasses):
            #for each subject and session, take curve data from array and determine best-fitting DoG parameters
            if dim3or2 == 3:
                y = data[i,k,:]
                amp, width, _, gof = fit_DoG(y, xrange, fittingSteps=fittingsteps, plot= False, ax = None, rangelim = rangelim, SD_type = SD_type)
                gof1 = gof[0]
                gof2 = gof[1]
                dog_params[i,k,:] = amp, width, gof1, gof2
            
            elif dim3or2 == 2:
                y = data[k,:]
                amp, width, _, gof = fit_DoG(y, xrange, fittingSteps=fittingsteps, plot= False, ax = None, rangelim = rangelim, SD_type = SD_type)
                gof1 = gof[0]
                gof2 = gof[1]
                dog_params[k,:] = amp, width, gof1, gof2
            
            if plot:
                data_model = Derivative_of_Gaussian(xrange, amp, width)
                ax.plot(xrange, data[i], c = 'r')
                ax.plot(xrange, data_model, c='k', linestyle = 'None', marker = 'o')
            
    return dog_params

def permute_modfree_SD_significance(data_clean, SD_type, split, n_permutations = 50):
    """
    Function to permute every subject's individual SD over their model free SD measure (median as in Gallagher 2022).
    Creates array of n_subsXn_splitXn_permutations, then returns array of n_subsXn_split (for SD) with median and SD over re-sampled data.
    
    :param data_clean: Full dataframe with all subjects' cleaned data (cleaned by remove_outliers_SD() function).
    :param SD_type: Which dependence's significance to determine.
    :param split: How many sections the data is to be split into.
    :param n_permutations: How many permutations to perform to arrive at significance.

    """
    import numpy as np
    from numpy.random import permutation
    from MemTempl_analysis_functions_basic import pulldata
    
    subs = np.unique(data_clean.Subject)
    
    p_onestar = 0.05
    p_twostars = 0.01
    p_threestars = 0.001
    
    #Prepare variable to store all dependence measures for each subject, each split section of data, and each permutation.
    permuted_medians = np.zeros((len(subs),len(split),n_permutations))
    
    #Prepare final variable to return significance measures for each subject and section of data
    permuted_measures = np.zeros((len(subs),len(split)))
        
    
    for subi, sub in enumerate(subs):
        #For each subject, pull correct independent variable sequence and response errors.
        x, titlename,_,_ = pulldata(data_clean[data_clean['Subject'] == sub], SD_type, n_back = 4, subject = '')

        if SD_type == 'SD_resp' or SD_type == 'SD_currori':
            y = data_clean[data_clean['Subject'] == sub].Corr_MemItem_Resp_Error.values
        else:
            y = data_clean[data_clean['Subject'] == sub].Resp_error_demeaned.values
        
        y[np.isnan(x)] = np.nan
        x[np.isnan(y)] = np.nan
        
        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]
        
        #Prepare variable to store all permuted data sequences + the "real" one.
        sub_perm = np.zeros((n_permutations+1,len(y)))

        #Fill that variable with permuted versions of data sequence by detaching 
        #the sequence of response errors from the sequence of independent variable expression.
        for permi in range(n_permutations):
            tmp_y = permutation(y)
            sub_perm[permi,:] = tmp_y
        
        #Store real version of data sequence at last spot in variable.
        sub_perm[-1,:] = y
        
        #Iterate through permuted data sequences and real data sequence.
        for dati in range(len(sub_perm)):
            print("Permutation " + str(dati))
            
            #For each data sequence, split data up into mirrored sections of length specified in split 
            #(analogously to procedure in return_modelfreemeasure_nonsmoothed() function).
            splitdataccw = [] 
            splitdatacw = []
            
            if len(split) == 1:
                splitdataccw.append(sub_perm[dati,:][(x > -split[0]) & (x < 0)])
            elif len(split) == 2:
                splitdataccw.append(sub_perm[dati,:][(x > -split[0]) & (x < -split[1])])
                splitdataccw.append(sub_perm[dati,:][(x > -split[1]) & (x < 0)])
            elif len(split) == 3:
                splitdataccw.append(sub_perm[dati,:][(x > -split[0]) & (x < -split[1])])
                splitdataccw.append(sub_perm[dati,:][(x > -split[1]) & (x < -split[2])])
                splitdataccw.append(sub_perm[dati,:][(x > -split[2]) & (x < 0)])
            
            if len(split) == 1:
                splitdatacw.append(sub_perm[dati,:][(x < split[0]) & (x > 0)])
            elif len(split) == 2:
                splitdatacw.append(sub_perm[dati,:][(x < split[0]) & (x > split[1])])
                splitdatacw.append(sub_perm[dati,:][(x < split[1]) & (x > 0)])
            elif len(split) == 3:
                splitdatacw.append(sub_perm[dati,:][(x < split[0]) & (x > split[1])])
                splitdatacw.append(sub_perm[dati,:][(x < split[1]) & (x > split[2])])
                splitdatacw.append(sub_perm[dati,:][(x < split[2]) & (x > 0)])
            
            #Prepare variable for last iteration
            real_measure = np.zeros((len(split)))
            
            #Determine each data section's dependence measure (analogously to procedure in return_modelfreemeasure_nonsmoothed() function)
            for k in range(len(split)):
                ccwsec = splitdataccw[k]
                cwsec = splitdatacw[k]
                
                #Store this measure in one variable for permuted sequences.
                if dati < len(sub_perm)-1:
                    permuted_medians[subi,k,dati] = np.median(ccwsec) - np.median(cwsec)
                
                #Store it in another variable for the real sequence.
                else:
                    real_measure[k] = np.median(ccwsec) - np.median(cwsec)
            
        """
        This last section was indented to the right one step further - I do not think
        it changed anything, but this way makes more sense.
        
        """
        
        #After for each subject, dependence measures have been computed for every permuted data sequence,
        #their distribution is compared to the real measure. The significance is determined from the distance
        #of the real measure to the mean of the permuted measures.
        #This is done for each section of data specified in split.
        
        for k in range(len(split)):
            #Sort all dependence measures for permuted data sequences.
            sorted_res = np.sort(permuted_medians[subi,k,:])
            
            #Determine the percentage of permuted measures that are higher or lower than the real measure.
            lower_perc = (sorted_res < real_measure[k]).sum()/len(sorted_res)
            upper_perc = (sorted_res > real_measure[k]).sum()/len(sorted_res)
            
            #Determine significance based on star-definition from the start.
            if lower_perc < p_threestars or upper_perc < p_threestars:
                permuted_measures[subi,k] = 3
            elif lower_perc < p_twostars or upper_perc < p_twostars:
                permuted_measures[subi,k] = 2
            elif lower_perc < p_onestar or upper_perc < p_onestar:
                permuted_measures[subi,k] = 1
            else:
                permuted_measures[subi,k] = 0
                
    return permuted_measures