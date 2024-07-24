import sys

sys.path.append('/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/Code')
from Training import data_objects
from Training.cluster_run_average import TrainArgs
from Training.cluster_run_average import ModularArithmeticDataset
import inspect
import functools
from activations_Ising import *
#from cluster_run_bias import CNN_nobias
import torch.nn.functional as F







def plot_traincurves(single_run,single_run_ng):
    titles=['Grok loss','No grok loss','Grok Accuracy','No grok accuracy']
    fig=make_subplots(rows=2,cols=2,subplot_titles=titles)


    #losses
    fig.add_trace(go.Scatter(x=list(range(len(single_run.train_losses))),y=single_run.train_losses,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=1,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(single_run.test_losses))),y=single_run.test_losses,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=1,col=1)
    #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(single_run_ng.test_losses))),y=single_run_ng.train_losses,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=1,col=2)
    fig.add_trace(go.Scatter(x=list(range(len(single_run_ng.train_losses))),y=single_run_ng.test_losses,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=1,col=2)

    #acc
    fig.add_trace(go.Scatter(x=list(range(len(single_run.test_accuracies))),y=single_run.train_accuracies,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=2,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(single_run.test_accuracies))),y=single_run.test_accuracies,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=2,col=1)
    #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(single_run_ng.test_accuracies))),y=single_run_ng.train_accuracies,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=2,col=2)
    fig.add_trace(go.Scatter(x=list(range(len(single_run_ng.train_accuracies))),y=single_run_ng.test_accuracies,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=2,col=2)
    #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)

    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_yaxes(title_text="Loss",type='log', row=1, col=2)

    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)

    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    fig.update_yaxes(title_text="Accuracy", row=2, col=2)

    fig.update_layout(title_text=f'Training curves: hidden layers={(single_run.trainargs.hiddenlayers,single_run_ng.trainargs.hiddenlayers)},wd={single_run.trainargs.weight_decay,single_run_ng.trainargs.weight_decay},wm={single_run.trainargs.weight_multiplier,single_run_ng.trainargs.weight_multiplier},train size={(single_run.trainargs.train_size,single_run_ng.trainargs.train_size)}, lr={(single_run.trainargs.lr,single_run_ng.trainargs.lr)}')
    fig.show()

from bisect import bisect_left

def find_nearest_index(sorted_values, target):
    # Find the position where the target should be inserted
    pos = bisect_left(sorted_values, target)
    # Handle edge cases for start and end of list
    if pos == 0:
        return 0
    if pos == len(sorted_values):
        return len(sorted_values) - 1
    # Check if the target is closer to the current position or the previous one
    before = pos - 1
    if abs(sorted_values[before] - target) <= abs(sorted_values[pos] - target):
        return before
    return pos

def find_closest(sorted_list, target):
    left, right = 0, len(sorted_list) - 1
    best_index = left
    while left <= right:
        mid = (left + right) // 2
        if abs(sorted_list[mid] - target) < abs(sorted_list[best_index] - target):
            best_index = mid
        
        if sorted_list[mid] < target:
            left = mid + 1
        elif sorted_list[mid] > target:
            right = mid - 1
        else:
            return mid  # Return immediately if exact match is found
    
    return best_index

def find_closest_descending(sorted_list, target):
    left, right = 0, len(sorted_list) - 1
    best_index = left
    while left <= right:
        mid = (left + right) // 2
        if abs(sorted_list[mid] - target) < abs(sorted_list[best_index] - target):
            best_index = mid
        
        # The main difference is in how we adjust the left and right pointers
        if sorted_list[mid] > target:
            left = mid + 1
        elif sorted_list[mid] < target:
            right = mid - 1
        else:
            return mid  # Return immediately if exact match is found
    
    return best_index
# Example usage:
def find_index_of_max_value(values):
    return max(enumerate(values), key=lambda x: x[1])[0]

def find_flat(run_object,start,window):
    ccdf_epochs=[]
    ccds_at_mins=[]
    print(f'epoch count: {len(run_object.model_epochs())}')
    for epoch in tqdm(run_object.model_epochs()):
        grok_state_dic=run_object.models[epoch]['model']
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        ccdf_mins=[]
        prob_at_min=[]
        for i in range(len(weights_grok)):
            flattened_gw=np.abs(torch.flatten(weights_grok[i]).detach().numpy())
            sorted_gw=np.sort(flattened_gw)+np.pi*10**(-30)
            test=np.isclose(sorted_gw,0)
            # if sum(test)>0:
            #     print(f' problematic gw: {sorted_gw}')
            log_sorted_gw=np.log(sorted_gw)
            cdf_gw=np.arange(1, len(flattened_gw) + 1) / len(flattened_gw)
            ccdf_gw=1-cdf_gw
            ccdf_gw+=np.pi*10**-(40)
            log_ccdf_gw=np.log(ccdf_gw)

            start_ind=find_closest_descending(sorted_list=ccdf_gw,target=start)
            if start_ind<len(log_ccdf_gw)-window-1:
                start_ind=start_ind
            else:
                start_ind=0
            # if epoch==14600 and i==2:
            #     print(ccdf_gw)
            #     print(start_ind)
            #     print(ccdf_gw[find_nearest_index(sorted_values=ccdf_gw,target=start)])
            #     print(len(ccdf_gw))
                # exit()
            if len(log_ccdf_gw)-window-1>start_ind:
                window=window
            else:
                window=len(log_ccdf_gw)-start_ind-2
            log_ccdf_gw_grads=[(log_sorted_gw[i+window]-log_sorted_gw[i])/np.log(10) for i in range(start_ind,len(log_ccdf_gw)-window-1)]
            abs_values=np.abs(log_ccdf_gw_grads)
            ccdf_mins.append(max(abs_values))
            prob_at_min.append(ccdf_gw[log_ccdf_gw_grads.index(max(abs_values))+start_ind])
        ccdf_epochs.append(ccdf_mins)
        ccds_at_mins.append(prob_at_min)
    return ccdf_epochs,ccds_at_mins





def across_lrs_mins(non_grok_filenames,start,window):
    
    titles=[]
    for filename in non_grok_filenames:
        print(filename)
        with open(filename, 'rb') as in_strm:
            single_run_ng=torch.load(in_strm,map_location=torch.device('cpu'))
        
        for i in range(4):
            titles.append(f'lr {single_run_ng.trainargs.lr}, Layer {i}')
    fig=make_subplots(rows=len(non_grok_filenames),cols=4,subplot_titles=titles)    

    count=1
    for filename in non_grok_filenames:
        print(filename)
        with open(filename, 'rb') as in_strm:
        #single_run_ng = dill.load(in_strm)
            single_run_ng=torch.load(in_strm,map_location=torch.device('cpu'))
        test_epochs,test_probs=find_flat(run_object=single_run_ng,start=start,window=window)
        first=True
        for i in range(len(test_epochs[0])):
            if first:
                fig.add_trace(go.Scatter(x=single_run_ng.model_epochs(),y=[t[i] for t in test_epochs],marker=dict(color='blue'),showlegend=first,name=f'{single_run_ng.trainargs.lr}'),row=count,col=i+1)
                first=False
            else:
                fig.add_trace(go.Scatter(x=single_run_ng.model_epochs(),y=[t[i] for t in test_epochs],marker=dict(color='blue'),showlegend=first,name=f'{single_run_ng.trainargs.lr}'),row=count,col=i+1)
            #fig.add_trace(go.Scatter(x=single_run_ng.model_epochs(),y=[t[i] for t in test_probs],marker=dict(color='blue'),showlegend=True,name='No grok'),row=2,col=i+1)
            fig.update_xaxes(title_text="Epoch", row=count, col=i+1)
            fig.update_yaxes(title_text=f"Max 1/grad", row=count, col=i+1)
        count+=1
        
    fig.show()





import matplotlib.pyplot as plt
from scipy.optimize import minimize




# Initial guess for the breakpoints (excluding the first and last points)


def piecewise_linear(x, breakpoints, slopes, intercepts):
    """
    Calculate the piecewise linear fit.
    :param x: The x values.
    :param breakpoints: The breakpoints between linear segments.
    :param slopes: The slopes of the linear segments.
    :param intercepts: The intercepts of the linear segments.
    :return: The y values corresponding to the piecewise linear fit.
    """
    # Number of segments is one more than the number of breakpoints
    n_segments = len(breakpoints) + 1
    y = np.zeros(x.shape)
    for i in range(n_segments):
        if i == 0:
            mask = x < breakpoints[0]
        elif i == n_segments - 1:
            mask = x >= breakpoints[-1]
        else:
            mask = (x >= breakpoints[i-1]) & (x < breakpoints[i])
        y[mask] = slopes[i] * x[mask] + intercepts[i]
    return y

def objective(params,x,y,n_parts):
    """
    Objective function to minimize.
    :param params: Array containing the breakpoints, slopes, and intercepts.
    :return: Sum of squared residuals between the actual y and the fitted y.
    """
    n_breakpoints = n_parts - 1
    breakpoints = params[:n_breakpoints]
    slopes = params[n_breakpoints:2*n_breakpoints+1]
    intercepts = params[2*n_breakpoints+1:]
    fitted_y = piecewise_linear(x, breakpoints, slopes, intercepts)
    return np.sum((y - fitted_y) ** 2)


def fit_piecewise_linear(x, y, n_parts):
    """
    Fit a piecewise linear model to the data.
    
    :param x: Array of x data points.
    :param y: Array of y data points.
    :param n_parts: Number of linear parts for the model.
    :return: A tuple containing the optimized breakpoints, slopes, intercepts, and a function to calculate fitted y.
    """
    initial_breakpoints = np.linspace(x.min(), x.max(), n_parts + 1)[1:-1]

    def piecewise_linear(x, breakpoints, slopes, intercepts):
        n_segments = len(breakpoints) + 1
        y = np.zeros(x.shape)
        for i in range(n_segments):
            if i == 0:
                mask = x < breakpoints[0]
            elif i == n_segments - 1:
                mask = x >= breakpoints[-1]
            else:
                mask = (x >= breakpoints[i-1]) & (x < breakpoints[i])
            y[mask] = slopes[i] * x[mask] + intercepts[i]
        return y

    def objective(params):
        n_breakpoints = n_parts - 1
        breakpoints = params[:n_breakpoints]
        slopes = params[n_breakpoints:2*n_breakpoints+1]
        intercepts = params[2*n_breakpoints+1:]
        fitted_y = piecewise_linear(x, breakpoints, slopes, intercepts)
        return np.sum((y - fitted_y) ** 2)

    initial_slopes = np.ones(n_parts)
    initial_intercepts = np.linspace(y.min(), y.max(), n_parts)

    initial_guess = np.concatenate([initial_breakpoints, initial_slopes, initial_intercepts])

    result = minimize(objective, initial_guess, method='Powell')

    optimized_breakpoints = result.x[:n_parts-1]
    optimized_slopes = result.x[n_parts-1:2*n_parts-1]
    optimized_intercepts = result.x[2*n_parts-1:]
    final_loss = result.fun


    def fitted_y_func(x_new):
        return piecewise_linear(x_new, optimized_breakpoints, optimized_slopes, optimized_intercepts)

    return optimized_breakpoints, optimized_slopes, optimized_intercepts,final_loss, fitted_y_func


def fit_piecewise_linear_with_loss_and_variable_gradients(x, y, n_parts):
    initial_breakpoints = np.linspace(x.min(), x.max(), n_parts + 1)[1:-1]

    def piecewise_linear(x, breakpoints, slopes, intercepts):
        n_segments = len(breakpoints) + 1
        y = np.zeros(x.shape)
        for i in range(n_segments):
            if i == 0:
                mask = x < breakpoints[0]
            elif i == n_segments - 1:
                mask = x >= breakpoints[-1]
            else:
                mask = (x >= breakpoints[i-1]) & (x < breakpoints[i])
            y[mask] = slopes[i] * x[mask] + intercepts[i]
        return y

    def objective(params):
        n_breakpoints = n_parts - 1
        breakpoints = params[:n_breakpoints]
        slopes = params[n_breakpoints:2*n_breakpoints+1]
        intercepts = params[2*n_breakpoints+1:]
        fitted_y = piecewise_linear(x, breakpoints, slopes, intercepts)
        return np.sum((y - fitted_y) ** 2)

    # Improved approach for initial guesses
    # Generate dynamic initial slopes based on divided sections of the dataset
    initial_slopes = np.diff(y) / np.diff(x)
    initial_slopes = initial_slopes[::len(initial_slopes) // n_parts][:n_parts]
    
    # Generate dynamic initial intercepts based on the start and end points of the dataset
    initial_intercepts = y[::len(y) // n_parts][:n_parts]

    initial_guess = np.concatenate([initial_breakpoints, initial_slopes, initial_intercepts])

    result = minimize(objective, initial_guess, method='Powell')

    optimized_breakpoints = result.x[:n_parts-1]
    optimized_slopes = result.x[n_parts-1:2*n_parts-1]
    optimized_intercepts = result.x[2*n_parts-1:]
    final_loss = result.fun

    def fitted_y_func(x_new):
        return piecewise_linear(x_new, optimized_breakpoints, optimized_slopes, optimized_intercepts)

    return optimized_breakpoints, optimized_slopes, optimized_intercepts, fitted_y_func, final_loss

# Example usage with the previously generated data (remove or replace this part when using your own data)


# Generate fitted y values for the original x data (for plotting)

def fit_piecewise_linear_with_negative_slopes(x, y, n_parts):
    initial_breakpoints = np.linspace(x.min(), x.max(), n_parts + 1)[1:-1]
    # initial_breakpoints=[(1/(n_parts+1-i))*((max(x)-min(x))) for i in range(n_parts)]
    # initial_breakpoints=[-3,-1.8,-1.5,-1,-0.5]

    def piecewise_linear(x, breakpoints, slopes, intercepts):
        n_segments = len(breakpoints) + 1
        y = np.zeros(x.shape)
        for i in range(n_segments):
            if i == 0:
                mask = x < breakpoints[0]
            elif i == n_segments - 1:
                mask = x >= breakpoints[-1]
            else:
                mask = (x >= breakpoints[i-1]) & (x < breakpoints[i])
            y[mask] = slopes[i] * x[mask] + intercepts[i]
        return y

    def objective(params):
        n_breakpoints = n_parts - 1
        breakpoints = params[:n_breakpoints]
        slopes = params[n_breakpoints:2*n_breakpoints+1]
        intercepts = params[2*n_breakpoints+1:]
        fitted_y = piecewise_linear(x, breakpoints, slopes, intercepts)
        penalty = 10000 * np.sum(np.maximum(0, slopes))  # Penalize positive slopes
        return np.sum((y - fitted_y) ** 2) + penalty

    # Adjust initial slopes to start from a negative value, assuming a decreasing trend
    # initial_slopes = -np.abs(np.diff(y) / np.diff(x))
    # initial_slopes = initial_slopes[::len(initial_slopes) // n_parts][:n_parts]
    initial_slopes=[(1/(n_parts+1-i))*((max(y)-min(y))/(max(x)-min(x))) for i in range(n_parts)]
    
    initial_intercepts = y[::len(y) // n_parts][:n_parts]

    initial_guess = np.concatenate([initial_breakpoints, initial_slopes, initial_intercepts])

    result = minimize(objective, initial_guess, method='Powell')

    optimized_breakpoints = result.x[:n_parts-1]
    optimized_slopes = result.x[n_parts-1:2*n_parts-1]
    optimized_intercepts = result.x[2*n_parts-1:]
    final_loss = result.fun

    def fitted_y_func(x_new):
        return piecewise_linear(x_new, optimized_breakpoints, optimized_slopes, optimized_intercepts)

    return optimized_breakpoints, optimized_slopes, optimized_intercepts, fitted_y_func, final_loss

# Re-fit the piecewise linear model enforcing negative slopes







def fit_piecewise_weights(run_object,start,end,n_parts):
    fit_ys=[]
    line_values=[]
    xs=[]
    ys=[]
    print(f'epoch count: {len(run_object.model_epochs())}')
    for epoch in tqdm(run_object.model_epochs()[:100]):
        grok_state_dic=run_object.models[epoch]['model']
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        line_data=[]
        fitted_values=[]
        x_ep=[]
        y_ep=[]
        for i in range(1,len(weights_grok)):
            flattened_gw=np.abs(torch.flatten(weights_grok[i]).detach().numpy())
            sorted_gw=np.sort(flattened_gw)+np.pi*10**(-30)

            log_sorted_gw=np.log(sorted_gw)
            cdf_gw=np.arange(1, len(flattened_gw) + 1) / len(flattened_gw)
            ccdf_gw=1-cdf_gw
            ccdf_gw+=np.pi*10**-(40)
            log_ccdf_gw=np.log(ccdf_gw)

            start_ind=find_closest_descending(sorted_list=ccdf_gw,target=start)
            if start_ind<len(log_ccdf_gw):
                start_ind=start_ind
            else:
                start_ind=0
            end_ind=find_closest_descending(sorted_list=ccdf_gw,target=end)
            
            y=log_ccdf_gw[start_ind:end_ind]
            x=log_sorted_gw[start_ind:end_ind]
            breakpoints, slopes, intercepts, fitted_y_func, loss = fit_piecewise_linear_with_negative_slopes(x, y, n_parts=n_parts)
            fitted_y_example = fitted_y_func(x)
            line_data.append([breakpoints,slopes,intercepts,loss,fitted_y_func])
            fitted_values.append(fitted_y_example)
            x_ep.append(x)
            y_ep.append(y)
        line_values.append(line_data)
        fit_ys.append(fitted_values)
        xs.append(x_ep)
        ys.append(y_ep)
    return fit_ys,line_values,xs,ys

def open_files_in_leaf_directories(root_dir):
    all_files=[]
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check if the current directory is a leaf directory
        if not dirnames:
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'rb') as in_strm:
                            single_run = torch.load(in_strm,map_location=device)
                                            # Do something with the content if needed
                            all_files.append(single_run)
                except Exception as e:
                    print(f"Failed to open {file_path}: {e}")
    return all_files





if __name__=="__main__":
    print('herro')
    #No bias
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/grok_ising/clusterdata/grok_True_time_1714759182/data_seed_0_time_1714762834_train_500_wd_0.08_lr0.0001"
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/grok_ising/clusterdata/grok_False_time_1714759214/data_seed_0_time_1714762715_train_500_wd_0.05_lr0.0001"

    #100 neurons final layers
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/LossCurves/grok_True_standard_param.torch"#<--
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/LossCurves/grok_False_standard_param.torch"
    
    
    #10 neurons final layers
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/fclayer_runs/grok_True_time_1714254742/data_seed_0_time_1714257653_train_500_wd_0.08_lr0.0001"#<--
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/fclayer_runs/grok_False_time_1714254764/data_seed_0_time_1714257737_train_500_wd_0.05_lr0.0001"#<--
    
    #Modadd no reg
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/ModAdd/ModAdd_512_grok_True.torch"
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/ModAdd/ModAdd_512_grok_False.torch"
    
    #dynamical data
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/clusterdata/hiddenlayer_[256]_desc_test_moadadd/grok_Truedataseed_0_sgdseed_0_initseed_0_wd_0.0003_wm_500.0_time_1717370830"
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/clusterdata/hiddenlayer_[256]_desc_test_moadadd/grok_Truedataseed_0_sgdseed_0_initseed_0_wd_0.0003_wm_1.0_time_1717371339"

    #mod add
    data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/oppositetest/hiddenlayer_[512]_desc_opp_modadd_wm_10.0/grok_Falsedataseed_0_sgdseed_0_initseed_0_wd_3e-05_wm_10.0_time_1719661186"
    data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/oppositetest/hiddenlayer_[512]_desc_opp_modadd_wm_1.0/grok_Falsedataseed_0_sgdseed_0_initseed_0_wd_3e-05_wm_1.0_time_1719658785"

    #data_object_file_name="clusterdata/grok_False_time_1712763706_wm_1/data_seed_0_time_1712763732_train_100_wd_0.0_lr0.004"
    #data_object_file_name_ng=data_object_file_name#"clusterdata/grok_False_time_1712762395_wm_1/data_seed_0_time_1712762659_train_100_wd_0.0_lr0.001"
    
    with open(data_object_file_name, 'rb') as in_strm:
        #single_run = dill.load(in_strm)
        single_run = torch.load(in_strm,map_location=torch.device('cpu'))
        single_run.norms=single_run.l2norms
    with open(data_object_file_name_ng, 'rb') as in_strm:
        #single_run = dill.load(in_strm)
        single_run_ng = torch.load(in_strm,map_location=torch.device('cpu'))
        single_run_ng.norms=single_run_ng.l2norms

    single_run.traincurves_and_iprs(single_run_ng).show()
    exit()
    
    # print(len(single_run_ng.iprs))
    # print(vars(single_run.trainargs))
    # print(vars(single_run_ng.trainargs))
    # exit()
    # single_run.traincurves_and_iprs(single_run_ng).show()
    # single_run.weights_histogram_epochs2(non_grokked_object=single_run_ng).show()
    

    

    dataset_filename="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/Data/IsingML_L16_traintest.pickle"
    with open(dataset_filename, "rb") as handle:
        dataset = dill.load(handle)[1]

    # single_run.plot_traincurves(single_run_ng).show()
    # single_run.weights_histogram_epochs2(non_grokked_object=single_run_ng).show()
    
    


    def share_of_zero_weights(runobject):
        epochs=runobject.model_epochs()
        # print(runobject.models[epochs[-1]]['model'].keys())
        # tempmodel=runobject.modelclass(**runobject.modelconfig)
        # tempmodel.load_state_dict(runobject.models[epochs[-1]]['model'])
        statedic=runobject.models[epochs[-1]]['model']
        flattened_weights=torch.cat([torch.abs(torch.flatten(statedic[key])) for key in statedic.keys()])
        meanweight=torch.mean(flattened_weights).item()
        threshold=meanweight*(0.001)
        zeropeakshare=torch.sum(flattened_weights<threshold).item()/len(flattened_weights)
        maxweight=torch.max(flattened_weights).item()
        # print(f' multiplier {runobject.trainargs.weight_multiplier}')
        # print(f' zero peak share of weights: {zeropeakshare}')
        # print(flattened_weights.shape)
        # print(f'initial multiplier: ')
        # print(f'mean weight {torch.mean(flattened_weights).item()}')
        # print(f'max weight {torch.max(flattened_weights).item()}')
        # print(f'min weight {torch.min(flattened_weights).item()}')
        #runobject.weights_histogram_epochs2(runobject)
        return zeropeakshare,maxweight
    
    foldername_seedaverage='/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/oppositetest'
    all_files=open_files_in_leaf_directories(foldername_seedaverage)
    print(all_files[0].trainargs)
    print(all_files[0].trainargs)

  
    # print(f'unequal args')
    # for arg in vars(all_files[0].trainargs).keys():
    #     if vars(all_files[0].trainargs)[arg]!=vars(single_run.trainargs)[arg]:
    #         print(f'Arg: {arg} cluster arg {vars(all_files[0].trainargs)[arg]}; single arg {vars(single_run.trainargs)[arg]}')
    
    # exit()


    print('cluster train args')
    for i in range(len(all_files)):
        if all_files[i].trainargs.weight_multiplier > 1:
            # print(all_files[i].trainargs.weight_multiplier)
            # print(all_files[i].trainargs)
            # print(all_files[i].l2norms[-1])
            # print(single_run.trainargs)
            # fig=make_subplots(rows=1,cols=1)
            # fig.add_trace(go.Scatter(x=list(range(len(all_files[i].l2norms))),y=np.array(all_files[i].l2norms)),row=1,col=1)
            # fig.show()
            

            fig=all_files[i].plot_traincurves(all_files[i])
            fig.update_layout(title_text=f'weight multiplier {all_files[i].trainargs.weight_multiplier}')
            #fig.show()
    
    #single_run.plot_traincurves(single_run_ng).show()
    #You should plot alongside the Ising and you'll see it's basically the same.
    print('all files length: ',len(all_files))
    maxaccuracies=[]
    minlosses=[]
    multiplier=[]
    for file in all_files:
        print(f'first norm {file.l2norms[0]}',f'last norm {file.l2norms[-1]}')
        maxaccuracies.append(max(file.test_accuracies))
        minlosses.append(min(file.test_losses))
        multiplier.append(file.trainargs.weight_multiplier)
    
    fig=make_subplots(rows=1,cols=2)
    fig.add_trace(go.Scatter(x=multiplier,y=maxaccuracies,name='Max accuracy',mode='markers',marker=dict(color='red')),row=1,col=1)
    fig.add_trace(go.Scatter(x=multiplier,y=minlosses,name='Min losses',mode='markers',marker=dict(color='red')),row=1,col=2)

    #fig.show()
    


    minlosses_tests=[]
    minlosses_train=[]
    norms=[]
    zeropeakshare=[]
    maxweights=[]
    for file in all_files:
        print(f'first norm {file.l2norms[0]}',f'last norm {file.l2norms[-1]}')
        first=True
        if first:
            scalefactor=file.trainargs.weight_multiplier/file.l2norms[-1]
            first=False
        
        minlosses_tests.append(min(file.test_losses))
        minlosses_train.append(min(file.train_losses))
        norms.append(file.l2norms[-1]*scalefactor)
        zeropeak,maxweight=share_of_zero_weights(file)
        zeropeakshare.append(zeropeak)
        maxweights.append(maxweight)
        
    
    # print(len(minlosses_train))
    # print(len(minlosses_tests))
    # print(len(norms))
    # print(minlosses_train)
    # exit()
    fig=make_subplots(rows=2,cols=1,subplot_titles=['LU'],specs=[[{}],[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=norms,y=minlosses_tests,name='Min test loss',mode='markers',marker=dict(color='red')),row=1,col=1)
    fig.add_trace(go.Scatter(x=norms,y=np.array(minlosses_train)+10**-15,name='Min train loss',mode='markers',marker=dict(color='blue')),row=1,col=1)
    fig.update_xaxes(title_text="Norm", row=1, col=1)
    fig.update_yaxes(title_text="Minimum Loss",type='log',row=1, col=1)
    fig.add_trace(go.Scatter(x=norms,y=zeropeakshare,name='Zero peak share',mode='markers',marker=dict(color='green')),row=2,col=1)
    #fig.add_trace(go.Scatter(x=norms,y=maxweights,name='Max weight',mode='markers',marker=dict(color='orange',symbol='square')),secondary_y=True,row=1,col=2)
    fig.update_yaxes(title_text="Share of zero weights", row=2, col=1)
    fig.update_xaxes(title_text="Norms", row=2, col=1)
    fig.show()

    exit()
    # fig=make_subplots(rows=1,cols=1)
    # fig.add_trace(go.Scatter(x=list(range(len(single_run.train_accuracies))),y=single_run.train_accuracies,marker=dict(color='red'),name='Grok'),row=1,col=1)
    # fig.add_trace(go.Scatter(x=list(range(len(single_run.test_accuracies))),y=single_run.test_accuracies,marker=dict(color='red'),name='Grok'),row=1,col=1)
    # fig.show()
    # exit()
    

    # print(f"trainargs grok\n {vars(single_run.trainargs)}")
    # print(f"trainargs no grok\n {vars(single_run_ng.trainargs)}")
    # print(single_run.models.keys())

    
    #OK great that function works, now let's try to get a function that calculates the correlation with the model output

    data_object_file_names=["/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/LossCurves/grok_False_standard_param.torch",
                            "/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata4/grok_False_time_1715457544_hiddenlayer_[100]/data_seed_0_time_1715459206_train_500_wd_0.05_lr1e-05",
                            "/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata4/grok_False_time_1715457565_hiddenlayer_[100]/data_seed_0_time_1715458561_train_500_wd_0.05_lr1e-06"]
    def open_obj(filename):
        with open(filename, 'rb') as in_strm:
        #single_run = dill.load(in_strm)
            runobj = torch.load(in_strm,map_location=torch.device('cpu'))
        return runobj

    # runobjs=[open_obj(x) for x in data_object_file_names]
    # fig=make_subplots(rows=1,cols=len(runobjs),subplot_titles=[r'$lr=10^{-4}$',r'$lr=10^{-5}$',r'$lr=10^{-6}$'])
    # count=1
    # for run in runobjs:

    #     fig.add_trace(go.Scatter(x=list(range(len(run.test_accuracies))),y=run.test_accuracies,mode="lines",line=dict(color='blue',dash="solid"),name='Learn test accuracy'),row=1,col=count)
    #     fig.add_trace(go.Scatter(x=list(range(len(run.train_accuracies))),y=run.train_accuracies,mode="lines",line=dict(color='blue',dash="dash"),name='Learn train accuracy'),row=1,col=count)
    #     count+=1
    # fig.show()
    # exit()
    #I want to write a function which calculates the covariances of all the weights. This should converge fairly quickly in the number of samples
    #
    def generalized_covariance_matrix(tensor):
    # Assume tensor is of shape Nxd1xd2x...xdn
        N = tensor.shape[0]
        # Step 1: Flatten tensor from Nxd1xd2x...xdn to Nx(product of other dimensions)
        tensor_reshaped = tensor.reshape(N, -1)  # -1 will automatically calculate the necessary size
        # Step 2: Compute the mean along the first dimension
        mean = tensor_reshaped.mean(dim=0)
        # Step 3: Center the matrix
        tensor_centered = tensor_reshaped - mean
        # Step 4: Compute the covariance matrix
        cov_matrix = (tensor_centered.T @ tensor_centered) / (N - 1)
        return cov_matrix
    


    def pairwise_cross_entropy_optimized(tensor):
        N = tensor.size(0)  # Preserve the size of the first dimension
        # Flatten tensor along all dimensions except the first
        flattened_tensor = tensor.view(N, -1)  # Shape becomes [N, prod(other_dims)]
        
        # Compute log_softmax across the flattened dimension
        log_probs = F.log_softmax(flattened_tensor, dim=0)  # Shape is [N, prod(other_dims)]

        # Expand both flattened tensor and log_probs for pairwise computation
        labels_expanded = flattened_tensor.unsqueeze(1)  # Shape becomes [N, 1, prod(other_dims)]
        log_probs_expanded = log_probs.unsqueeze(2)  # Shape becomes [1, N, prod(other_dims)]

        # Compute cross-entropy
        cross_entropy = -torch.sum(labels_expanded * log_probs_expanded, dim=0)  # Shape becomes [N, N]

        return cross_entropy
    
    def pairwise_relative_entropy(tensor1, tensor2):
        # Flatten tensors along all dimensions except the first
        tensor1_flat = tensor1.view(tensor1.size(0), -1)  # Shape [N, t1]
        tensor2_flat = tensor2.view(tensor2.size(0), -1)  # Shape [N, t2]

        # Normalize along the zeroth dimension (samples)
        probs1 = F.softmax(tensor1_flat, dim=0)
        probs2 = F.softmax(tensor2_flat, dim=0)

        # Add dimensions for broadcasting: [N, t1, 1] and [N, 1, t2]
        probs1_expanded = probs1.unsqueeze(2)
        probs2_expanded = probs2.unsqueeze(1)

        # Ensure probabilities are above a threshold to avoid log(0)
        eps = 1e-10
        probs1_expanded = torch.clamp(probs1_expanded, min=eps)
        probs2_expanded = torch.clamp(probs2_expanded, min=eps)

        # Calculate KL divergence for each pair (broadcasting)
        # KL(P || Q) = sum(P * (log(P) - log(Q)))
        kl_divergences = torch.sum(probs1_expanded * (torch.log(probs1_expanded) - torch.log(probs2_expanded)), dim=0)

        return kl_divergences
    


    def cross_entropy_withinlayer(run_object,epoch,samples,layername):
        model=run_object.modelclass(**single_run.modelconfig)
        statedict=run_object.models[epoch]['model']
        result=model.load_state_dict(statedict,strict=False)
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)
        
        test_dataset=generate_test_set(dataset,samples)
        activations_grok, output, cleanup = get_activations(model,test_dataset[0])#Might be good to compare grokked and c
        cleanup()
        
        layertensor=activations_grok[layername]

        layer_cross_entropy=pairwise_cross_entropy_optimized(layertensor)
        return layer_cross_entropy
    
    def cross_entropy_betweenlayer(run_object,epoch,samples,layerone,layertwo):
        model=run_object.modelclass(**single_run.modelconfig)
        statedict=run_object.models[epoch]['model']
        result=model.load_state_dict(statedict,strict=False)
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)
        
        test_dataset=generate_test_set(dataset,samples)
        activations_grok, output, cleanup = get_activations(model,test_dataset[0])#Might be good to compare grokked and c
        cleanup()
        
        layertensor_one=activations_grok[layerone]
        layertensor_two=activations_grok[layertwo]

        layer_cross_entropy=pairwise_relative_entropy(layertensor_one,layertensor_two)
        return layer_cross_entropy


    epoch=99900
    samples=1000
    layername='conv_layers.3 (Conv2d)'

    def get_cov_tensor(run_object,epoch,samples,layername):
        model=run_object.modelclass(**single_run.modelconfig)
        statedict=run_object.models[epoch]['model']
        result=model.load_state_dict(statedict,strict=False)
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)
        
        test_dataset=generate_test_set(dataset,samples)
        activations_grok, output, cleanup = get_activations(model,test_dataset[0])#Might be good to compare grokked and c
        cleanup()
        
        layertensor=activations_grok[layername]

        layercov=generalized_covariance_matrix(layertensor)
        return layercov
    
    def output_cov(run_object,epoch,samples,layername):
        model=run_object.modelclass(**single_run.modelconfig)
        statedict=run_object.models[epoch]['model']
        result=model.load_state_dict(statedict,strict=False)
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)
        
        test_dataset=generate_test_set(dataset,samples)
        activations_grok, output, cleanup = get_activations(model,test_dataset[0])#Might be good to compare grokked and c
        cleanup()
        layertensor=activations_grok[layername]
        
        
        N=layertensor.shape[0]
        output_diff=output[:,0]-output[:,1]#Probably correlates to the energy!
        #OK now I need to get the covariance:
        # Reshape X to N x (k*k*m*l)
        layertensor_flattened = layertensor.view(N, -1)
        outputdiff_flattened=output_diff.view(-1,1)


        # Calculating means
        mean_layertensor = layertensor_flattened.mean(dim=0, keepdim=True)
        mean_outputdiff = outputdiff_flattened.mean(dim=0, keepdim=True)
        
        
        # Subtract means
        layertensor_centered = layertensor_flattened - mean_layertensor
        outputdiff_centered = outputdiff_flattened - mean_outputdiff

        # Calculate covariance
        covariance = (layertensor_centered * outputdiff_centered).sum(dim=0) / (N - 1)
        

        #layercov=generalized_covariance_matrix(layertensor)
        return covariance
    
    def output_covariance_fig(grokked_object,non_grokked_object,epoch,samples,layername):
        layername='conv_layers.0 (Conv2d)'
        grok_outputvec_0,grok_outputvec_indices=torch.sort(output_cov(grokked_object,epoch,samples,layername),descending=True)
        nogrok_outputvec_0,nogrok_outputvec_indices=torch.sort(output_cov(non_grokked_object,epoch,samples,layername),descending=True)
        layername='conv_layers.3 (Conv2d)'
        grok_outputvec_1,grok_outputvec_indices=torch.sort(output_cov(grokked_object,epoch,samples,layername),descending=True)
        nogrok_outputvec_1,nogrok_outputvec_indices=torch.sort(output_cov(non_grokked_object,epoch,samples,layername),descending=True)
        layername='fc_layers.0 (Linear)'
        grok_outputvec_2,grok_outputvec_indices=torch.sort(output_cov(grokked_object,epoch,samples,layername),descending=True)
        nogrok_outputvec_2,nogrok_outputvec_indices=torch.sort(output_cov(non_grokked_object,epoch,samples,layername),descending=True)

        fig=make_subplots(rows=1,cols=3,subplot_titles=['Conv Layer 1','Conv Layer 2', 'FC layer'])
        fig.add_trace(go.Scatter(x=list(range(len(grok_outputvec_0))),y=grok_outputvec_0.detach().numpy(),marker=dict(color='red'),name='Grok'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(nogrok_outputvec_0))),y=nogrok_outputvec_0.detach().numpy(),marker=dict(color='blue'),name='No grok'),row=1,col=1)
        
        fig.add_trace(go.Scatter(x=list(range(len(grok_outputvec_1))),y=grok_outputvec_1.detach().numpy(),marker=dict(color='red'),name='Grok',showlegend=False),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(nogrok_outputvec_1))),y=nogrok_outputvec_1.detach().numpy(),marker=dict(color='blue'),name='No grok',showlegend=False),row=1,col=2)

        fig.add_trace(go.Scatter(x=list(range(len(grok_outputvec_2))),y=grok_outputvec_2.detach().numpy(),marker=dict(color='red'),name='Grok',showlegend=False),row=1,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(nogrok_outputvec_2))),y=nogrok_outputvec_2.detach().numpy(),marker=dict(color='blue'),name='No grok',showlegend=False),row=1,col=3)
        fig.update_yaxes(title_text='Covariance with output difference')
        fig.update_xaxes(title_text='Rank of neuron')
        return fig
    
    def covariance_matrix_plots(grokked_object,non_grokked_object,layername,epoch,samples):
        #layername='conv_layers.0 (Conv2d)'
        titles=['Cov matrix of activations - grok','Cov matrix of activations - no grok','Eig vals','','Diagonal components of Cov','Off-diagonal components']
        fig=make_subplots(rows=3,cols=2,subplot_titles=titles)
        grok_cov=get_cov_tensor(run_object=grokked_object,epoch=epoch,samples=samples,layername=layername)
        nogrok_cov=get_cov_tensor(run_object=non_grokked_object,epoch=epoch,samples=samples,layername=layername)
        ug,sg,vg=torch.linalg.svd(grok_cov)
        ung,sng,vng=torch.linalg.svd(nogrok_cov)
        g_ofdiag=(grok_cov-torch.diag(grok_cov)).view(-1)
        threshold=10**-10
        g_ofdiag=g_ofdiag[torch.abs(g_ofdiag)>threshold]
        ng_ofdiag=nogrok_cov-torch.diag(nogrok_cov)
        ng_ofdiag=ng_ofdiag[torch.abs(ng_ofdiag)>threshold]

        #Want to set a common scale
        combined_data = grok_cov + nogrok_cov
        zmin = torch.min(combined_data).item()
        zmax = torch.max(combined_data).item()
        fig.add_trace(go.Heatmap(z=grok_cov,zmin=zmin,zmax=zmax),row=1,col=1)
        fig.add_trace(go.Heatmap(z=nogrok_cov,zmin=zmin,zmax=zmax),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(sg))),y=sg/sg[0],marker=dict(color='red'),name='Grok'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(sng))),y=sng/sng[0],marker=dict(color='blue'),name='No grok'),row=2,col=1)
        #fig.add_trace(go.Scatter(x=list(range(len(sg))),y=torch.diag(grok_cov),marker=dict(color='red'),name='Grok'),row=2,col=2)
        #fig.add_trace(go.Scatter(x=list(range(len(sng))),y=torch.diag(nogrok_cov),marker=dict(color='blue'),name='No grok'),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(sg))),y=torch.diag(grok_cov)/torch.mean(torch.diag(grok_cov)),marker=dict(color='red'),name='Grok'),row=3,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(sng))),y=torch.diag(nogrok_cov)/torch.mean(torch.diag(nogrok_cov)),marker=dict(color='blue'),name='No grok'),row=3,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(g_ofdiag))),y=g_ofdiag,marker=dict(color='red'),name='Grok'),row=3,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(ng_ofdiag))),y=ng_ofdiag,marker=dict(color='blue'),name='No grok'),row=3,col=2)
        return fig
    
    #Code for correlations
    def correlation_coefficient(activations, features):
        first=True
        for feature_dim in range(len(features[1])):
            feature_ten=features[:,feature_dim]
            # Calculate means along the first dimension
            activations_mean = torch.mean(activations, dim=0, keepdim=True)
            features_mean = torch.mean(feature_ten, dim=0, keepdim=True)

            
            # Calculate deviations
            activations_dev = activations - activations_mean
            features_dev = feature_ten - features_mean
            new_shape=[-1]+[1] * (len(activations_dev.shape) - 1)
            #features_dev=features_dev.unsqueeze(1)
            features_dev=features_dev.view(new_shape)
            print(activations_dev.shape)
            print(features_dev.shape)
            
            # Calculate correlation coefficient
            numerator = torch.sum(activations_dev * features_dev, dim=0)
            denominator = torch.sqrt(torch.sum(activations_dev ** 2, dim=0) * torch.sum(features_dev ** 2, dim=0))
            correlation = numerator / denominator
            if first:
                feature_correlation=correlation.unsqueeze(-1)
                first=False
                print(feature_correlation.shape)
                
            else:
                correlation=correlation.unsqueeze(-1)
                feature_correlation=torch.cat((feature_correlation,correlation),dim=-1)


        
        return feature_correlation
    
    def correlation_one_epoch_prod(grokked_object,non_grokked_object,epoch,images_tensor,feature_funcs,sortby,dataset,layernames,chosen_neuron_index):
        features_tensor=construct_features_tensor(images_tensor=images_tensor,feature_funcs=feature_funcs)
        def correlation_activations(object):
            model=object.modelclass(**object.modelconfig)
            state_dic=object.models[epoch]['model']
            result=model.load_state_dict(state_dic,strict=False)
            if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
                print(result.missing_keys,result.unexpected_keys)

            activations, output, cleanup = get_activations(model,images_tensor)
            cleanup()
            sorted_activations,acts_indices=get_acts_dict(single_run=object,dataset=dataset,epoch=epoch,sortby=sortby)
            for name in list(activations.keys()):
                if (name in layernames)==False:
                    activations.pop(name)
                    sorted_activations.pop(name)
                    acts_indices.pop(name)
            return activations, sorted_activations, acts_indices
        
        activations_grok,sorted_activations_grok,acts_indices_grok=correlation_activations(object=grokked_object)
        activations_nogrok,sorted_activations_nogrok,acts_indices_nogrok=correlation_activations(object=non_grokked_object)
        correlations_grok={key: correlation_coefficient(activations_grok[key], features_tensor) for key in activations_grok.keys()}
        correlations_nogrok={key: correlation_coefficient(activations_nogrok[key], features_tensor) for key in activations_nogrok.keys()}
        #print(f' acts indices grok shape: {acts_indices_grok[layernames[0]].shape}')
        #print(f'first ten entries of act_indices: {acts_indices_grok[layernames[0]][:10]}')
        #print(f'sorted activation first ten indices: {[sorted_activations_grok[layernames[0]][index].item() for index in acts_indices_grok[layernames[0]][:10]]}')
        #print(f' correlations_grok shape: {correlations_grok[layernames[0]].shape}')
        #print(f' acts_indices shape: {acts_indices_grok[layernames[0]].shape}')
        #print(f' squeezed acts_indices shape: {acts_indices_grok[layernames[0]].squeeze(-1).shape}')
        
        sorted_correlations_grok={key: torch.index_select(correlations_grok[key],0,acts_indices_grok[key].squeeze(-1)) for key in activations_grok.keys()}
        sorted_correlations_nogrok={key: torch.index_select(correlations_nogrok[key],0,acts_indices_nogrok[key].squeeze(-1)) for key in activations_nogrok.keys()}
        #print(f'sorted corr grok shape: {sorted_correlations_grok[layernames[0]].shape}')

        #OK - now I can do the plots
        titles=[r'$\text{(a) Grokking activation-energy correlation for most activated neuron}$',r'$\text{(b) Learning activation-energy correlation for most activated neuron}$',r'$\text{(c) Grokking activation-energy correlation coefficients for all neurons}$',r'$\text{(d) Learning activation-energy correlation coefficients for all neurons}$']
        fig=make_subplots(rows=2,cols=2,subplot_titles=titles)
        activations_index_grok=acts_indices_grok[layernames[0]][chosen_neuron_index]
        grok_neuron_values=activations_grok[layernames[0]][(slice(None),)+tuple(activations_index_grok.tolist())]
        activations_index_nogrok=acts_indices_nogrok[layernames[0]][chosen_neuron_index]
        nogrok_neuron_values=activations_nogrok[layernames[0]][(slice(None),)+tuple(activations_index_nogrok.tolist())]
        #print(grok_neuron_values.shape)
        #print(activations_grok[layernames[0]].shape)
        
        fig.add_trace(go.Scatter(x=features_tensor[:,0],y=grok_neuron_values,mode='markers',marker=dict(color='red'),name=r'$\text{Grokking}$'),row=1,col=1)
        fig.add_trace(go.Scatter(x=features_tensor[:,0],y=nogrok_neuron_values,mode='markers',marker=dict(color='blue'),name=r'$\text{Learning}$'),row=1,col=2)
        fig.update_xaxes(title_text=r'$\text{Normalized energy}$',row=1,col=1)
        fig.update_xaxes(title_text=r'$\text{Normalized energy}$',row=1,col=2)
        fig.update_yaxes(title_text=r'$\text{Neuron activation}$',row=1,col=1)
        fig.update_yaxes(title_text=r'$\text{Neuron activation}$',row=1,col=2)
        
        
        
        fig.add_trace(go.Scatter(x=list(range(len(sorted_correlations_grok[layernames[0]][:,0]))),y=sorted_correlations_grok[layernames[0]][:,0],mode='markers',marker=dict(color='red'),name=r'$\text{Grokking}$',showlegend=False),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(sorted_correlations_nogrok[layernames[0]][:,0]))),y=sorted_correlations_nogrok[layernames[0]][:,0],mode='markers',marker=dict(color='blue'),name=r'\text{Learning}$',showlegend=False),row=2,col=2)
        fig.update_xaxes(title_text=r'$\text{Neuron rank}$',row=2,col=1)
        fig.update_xaxes(title_text=r'$\text{Neuron rank}$',row=2,col=2)
        fig.update_yaxes(title_text=r'$\text{Correlation coefficient}$',row=2,col=1)
        fig.update_yaxes(title_text=r'$\text{Correlation coefficient}$',row=2,col=2)
        fig.update_layout(
        margin=dict(  # Adjust margins to provide more space
            l=20,  # Left margin
            r=150,  # Right margin increased to prevent overlap
            t=50,  # Top margin
            b=20   # Bottom margin
            ))
        
        
        return fig



    def cross_entropy_plot(grokked_object,non_grokked_object,samples,epoch,fig,layerone,layertwo):
        # crossentropy_matrix_g=cross_entropy_withinlayer(run_object=single_run,epoch=epoch,samples=1000,layername='conv_layers.3 (Conv2d)')
        # crossentropy_matrix_ng=cross_entropy_withinlayer(run_object=single_run_ng,epoch=epoch,samples=1000,layername='conv_layers.3 (Conv2d)')
        crossentropy_matrix_g=cross_entropy_betweenlayer(run_object=single_run,epoch=epoch,samples=samples,layerone=layerone,layertwo=layertwo)
        crossentropy_matrix_ng=cross_entropy_betweenlayer(run_object=single_run_ng,epoch=epoch,samples=samples,layerone=layerone,layertwo=layertwo)
        xe_ug,xe_sg,xe_vg=torch.linalg.svd(crossentropy_matrix_g)
        xe_ung,xe_sng,xe_vng=torch.linalg.svd(crossentropy_matrix_ng)


        fig = make_subplots(rows=3, cols=2,
                        specs=[[{}, {}], [{}, {}],
                            [{'colspan': 2}, None]],  # Second row spans two columns
                        subplot_titles=['Grok cross-entropy','Learn cross-entropy','Grok cross entropy histogram','Learn cross entropy histogram','Singular values'])
        
        combined_data = torch.cat((crossentropy_matrix_g,crossentropy_matrix_ng))
        zmin = torch.min(combined_data).item()
        zmax = torch.max(combined_data).item()

        fig.add_trace(go.Heatmap(z=crossentropy_matrix_g,zmin=zmin,zmax=zmax,showscale=False),row=1,col=1)
        fig.add_trace(go.Heatmap(z=crossentropy_matrix_ng,zmin=zmin,zmax=zmax,showscale=True),row=1,col=2)
        fig.add_trace(go.Histogram(x=torch.flatten(crossentropy_matrix_g),nbinsx=128,marker=dict(color='red'),showlegend=False),row=2,col=1)
        fig.add_trace(go.Histogram(x=torch.flatten(crossentropy_matrix_ng),nbinsx=128,marker=dict(color='blue'),showlegend=False),row=2,col=2)
        fig.update_layout(
                xaxis4=dict(matches='x3'),  # Ensures that the second x-axis matches the first
                yaxis4=dict(matches='y3')  # Ensures that the second y-axis matches the first
            )
        fig.add_trace(go.Scatter(x=list(range(len(xe_sg))),y=xe_sg/xe_sg[0],marker=dict(color='red')),row=3,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(xe_sng))),y=xe_sng/xe_sng[0],marker=dict(color='blue')),row=3,col=1)
        return fig 


    def calculate_updates(state_dict,model_params):
        # Constants
        epsilon = state_dict['param_groups'][0]['eps']  # Small constant for numerical stability

        updateslist = []
        for group in state_dict['param_groups']:
            learning_rate = group['lr']
            weight_decay = group['weight_decay']
            for param_id in group['params']:
                param_state = state_dict['state'][param_id]
                if 'exp_avg' in param_state and 'exp_avg_sq' in param_state:
                    # Retrieve the actual parameter tensor from model's parameters using param_id
                    #param_tensor = next((p for p in model_params if id(p) == param_id), None)
                    param_tensor = model_params[param_id]
                    if param_tensor is None:
                        print(f"No matching parameter found for ID {param_id}")
                        continue  # Skip this parameter if no matching tensor is found
                    exp_avg = param_state['exp_avg']
                    exp_avg_sq = param_state['exp_avg_sq']
                    step = param_state['step']

                    bias_correction1 = 1 - 0.9 ** step
                    bias_correction2 = 1 - 0.999 ** step

                    
                    
                    # Calculate update
                    update = (learning_rate * exp_avg / (exp_avg_sq.sqrt() / bias_correction2.sqrt() + epsilon))

                    wd_update = weight_decay * param_tensor
                    updateslist.append((exp_avg,wd_update))
        
        return updateslist

    def optimizer_plots(runobject,epoch):
        model_state_dic=runobject.models[epoch]['model']
        model=runobject.modelclass(**runobject.modelconfig)
        model.load_state_dict(model_state_dic,strict=False)
        model.eval()
        opt_dic=copy.deepcopy(runobject.models[epoch]['optimizer'])
        #print(f" opt dic epoc {epoch} exp avg {opt_dic['state'][7]['exp_avg']}")

        update_list=calculate_updates(opt_dic,list(model.parameters()))
        del opt_dic
        #chosen_dic={key:update_dic[key] for key in chosen_indices}
        return update_list

    def weight_norm_plot(grokked_object,non_grokked_object):
        weight_norms_grokked=[]
        weight_norms_non_grokked=[]
        epochs=[]
        
        for epoch in tqdm(grokked_object.models.keys()):
            grok_state_dic=grokked_object.models[epoch]['model']
            if epoch==10000:
                
                grok_updates_tensor=optimizer_plots(grokked_object,epoch)
                
                
                
                print(grok_opt_dic.keys())
                print(grok_opt_dic['param_groups'])
                print(grok_opt_dic['state'].keys())
                print(grok_opt_dic['state'][0].keys())
                for i in range(7):
                    print(grok_opt_dic['state'][i]['exp_avg'].shape)
                print(grok_opt_dic['state'][4]['exp_avg'].shape)
                print(f"max: {torch.max(grok_opt_dic['state'][4]['exp_avg'])}")
                print(f"min: {torch.min(grok_opt_dic['state'][4]['exp_avg'])}")
                print(f"mean: {torch.mean(grok_opt_dic['state'][4]['exp_avg'])}")
                print(f"var: {torch.var(grok_opt_dic['state'][4]['exp_avg'])**0.5}")
                print(f"state dic update len {len(state_dic_update)}")
                print(f"sizes state dic updates {[x.shape for x in state_dic_update]}")
                
                exit()
            
            grok_weights=[torch.sum(grok_state_dic[key]**2) for key in grok_state_dic.keys() if 'weight' in key]

            nogrok_state_dic=non_grokked_object.models[epoch]['model']
            nogrok_weights=[torch.sum(nogrok_state_dic[key]**2) for key in nogrok_state_dic.keys() if 'weight' in key]

            weight_norms_grokked.append(grok_weights)
            weight_norms_non_grokked.append(nogrok_weights)
            epochs.append(epoch)
        # fig=make_subplots(rows=1,cols=len(weight_norms_grokked[0]))
        # fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_accuracies))),y=grokked_object.test_accuracies,marker=dict(color='red'),name='Grok',showlegend=True),row=1,col=1)
        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='blue'),name='Learn',showlegend=True),row=1,col=1)
        # for i in range(len(weight_norms_grokked[0])-1):
        #     fig.add_trace(go.Scatter(x=epochs,y=np.array(weight_norms_grokked)[:,i],marker=dict(color='red'),name='Grok',showlegend=False),row=1,col=2+i)
        #     fig.add_trace(go.Scatter(x=epochs,y=np.array(weight_norms_non_grokked)[:,i],marker=dict(color='blue'),name='Learn',showlegend=False),row=1,col=2+i)
        #     fig.update_yaxes(type='log',row=1,col=2+i)
        fig=make_subplots(rows=1,cols=2,specs=[[{"secondary_y": True}, {"secondary_y": True}]],subplot_titles=["Grok test accuracy and weight Norm","Learn test accuracy and weight norm"])
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_accuracies))),y=grokked_object.test_accuracies,marker=dict(color='red'),name='Grok Test Accuracy',showlegend=True),row=1,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=torch.sum(torch.tensor(weight_norms_grokked),dim=1),mode="lines",line=dict(color='red',dash="dash"),name='Grok Weight Norm',showlegend=True),row=1,col=1,secondary_y=True)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='blue'),name='Learn Accuracy',showlegend=True),row=1,col=2)
        fig.add_trace(go.Scatter(x=epochs,y=torch.sum(torch.tensor(weight_norms_non_grokked),dim=1),mode="lines",line=dict(color='blue',dash="dash"),name='Learn Weight norm',showlegend=True),row=1,col=2,secondary_y=True)
        fig.update_yaxes(title_text="(Log) Norm", secondary_y=True, type='log',row=1,col=1)
        fig.update_yaxes(title_text="(Log) Norm", secondary_y=True, type='log',row=1,col=2)
        return fig
    
    def weight_distance_plot(grokked_object,non_grokked_object):
        weight_norms_grokked=[]
        weight_norms_non_grokked=[]
        epochs=[]
        grok_distances=[]
        stored_grok_weights=[]
        stored_nogrok_weights=[]
        nogrok_distances=[]
        weight_decay_grok=[]
        wd_grok=0.08
        weight_decay_nogrok=[]
        wd_nogrok=0.05
        for epoch in tqdm(grokked_object.models.keys()):
            if epoch==0:
                grok_state_dic=grokked_object.models[epoch]['model']            
                grok_weights=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
                stored_grok_weights.append(grok_weights)

                nogrok_state_dic=non_grokked_object.models[epoch]['model']
                nogrok_weights=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]
                stored_nogrok_weights.append(nogrok_weights)
                continue
            else:
                grok_state_dic=grokked_object.models[epoch]['model']            
                grok_weights=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
                
                grok_weight_distances=[torch.sum((grok_weights[i]-stored_grok_weights[-1][i])**2) for i in range(len(grok_weights))]
                weight_decay_part_grok=[torch.sum((wd_grok*stored_grok_weights[-1][i])**2) for i in range(len(grok_weights))]
                weight_decay_grok.append(weight_decay_part_grok)
                grok_distances.append(grok_weight_distances)
                
                stored_grok_weights.append(grok_weights)

                nogrok_state_dic=non_grokked_object.models[epoch]['model']            
                nogrok_weights=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]
                
                nogrok_weight_distance=[torch.sum((nogrok_weights[i]-stored_nogrok_weights[-1][i])**2) for i in range(len(nogrok_weights))]
                weight_decay_part_nogrok=[torch.sum((wd_nogrok*stored_nogrok_weights[-1][i])**2) for i in range(len(grok_weights))]
                weight_decay_nogrok.append(weight_decay_part_nogrok)
                nogrok_distances.append(nogrok_weight_distance)
                stored_nogrok_weights.append(nogrok_weights)

                epochs.append(epoch)
            
        # fig=make_subplots(rows=1,cols=len(weight_norms_grokked[0]))
        # fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_accuracies))),y=grokked_object.test_accuracies,marker=dict(color='red'),name='Grok',showlegend=True),row=1,col=1)
        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='blue'),name='Learn',showlegend=True),row=1,col=1)
        # for i in range(len(weight_norms_grokked[0])-1):
        #     fig.add_trace(go.Scatter(x=epochs,y=np.array(weight_norms_grokked)[:,i],marker=dict(color='red'),name='Grok',showlegend=False),row=1,col=2+i)
        #     fig.add_trace(go.Scatter(x=epochs,y=np.array(weight_norms_non_grokked)[:,i],marker=dict(color='blue'),name='Learn',showlegend=False),row=1,col=2+i)
        #     fig.update_yaxes(type='log',row=1,col=2+i)

        fig=make_subplots(rows=2,cols=2,specs=[[{"secondary_y": True}, {"secondary_y": True}],[{"secondary_y": True},{"secondary_y": True}]],subplot_titles=["Grok test accuracy and weight distance","Learn test accuracy and weight distance","Grok test accuracy and loss contribution","Learn test accuracy and loss contribution"])
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_accuracies))),y=grokked_object.test_accuracies,mode="lines",line=dict(color='red',dash="dash"),name='Grok Test Accuracy',showlegend=True),row=1,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=torch.sum(torch.tensor(grok_distances),dim=1),name='Grok Weight Norm',marker=dict(color='red'),showlegend=True),row=1,col=1,secondary_y=True)
        
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_accuracies))),y=grokked_object.test_accuracies,mode="lines",line=dict(color='red',dash="dash"),name='Grok Test Accuracy',showlegend=False),row=2,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=torch.sum(torch.tensor(weight_decay_grok),dim=1),name='Wd part',marker=dict(color='black'),showlegend=True),row=2,col=1,secondary_y=True)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies,mode="lines",line=dict(color='blue',dash="dash"),name='Learn Accuracy',showlegend=True),row=1,col=2)
        fig.add_trace(go.Scatter(x=epochs,y=torch.sum(torch.tensor(nogrok_distances),dim=1),marker=dict(color='blue'),name='Learn Weight norm',showlegend=True),row=1,col=2,secondary_y=True)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies,mode="lines",line=dict(color='blue',dash="dash"),name='Learn Accuracy',showlegend=True),row=2,col=2)
        fig.add_trace(go.Scatter(x=epochs,y=torch.sum(torch.tensor(weight_decay_nogrok),dim=1),name='Wd part',marker=dict(color='black'),showlegend=True),row=2,col=2,secondary_y=True)
        #fig.update_yaxes(title_text="(Log) Norm", secondary_y=True, type='log',row=1,col=1)
        #fig.update_yaxes(title_text="(Log) Norm", secondary_y=True, type='log',row=1,col=2)
        return fig
    
    def optimizer_plot(grokked_object,non_grokked_object,chosen_indices):
        updates_grokked=[]
        updates_non_grokked=[]
        epochs=[]
        
        for epoch in tqdm(grokked_object.models.keys()):
            grok_state_dic=grokked_object.models[epoch]['model']
            grok_updates_tensor=optimizer_plots(grokked_object,epoch)
            # if epoch<200:
            #     print(grok_updates_tensor[7][0])
            # else:
            #     exit()
            non_grok_state_dic=non_grokked_object.models[epoch]['model']
            non_grok_updates_tensor=optimizer_plots(non_grokked_object,epoch)

            updates_grokked.append(grok_updates_tensor)
            updates_non_grokked.append(non_grok_updates_tensor)
            epochs.append(epoch)        
            # if epoch in [1000]:
            #     print(torch.max(grok_updates_tensor[4][0]))
            #     print(torch.min(grok_updates_tensor[4][0]))
            #     exit()
            


        #fig=make_subplots(rows=1,cols=2,specs=[[{"secondary_y": True}, {"secondary_y": True}]],subplot_titles=["Grok test accuracy and weight Norm","Learn test accuracy and weight norm"])
        fig=make_subplots(rows=1+2,cols=2)
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_accuracies))),y=grokked_object.test_accuracies,marker=dict(color='red'),name='Grok Test Accuracy',showlegend=True),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.train_accuracies))),y=grokked_object.train_accuracies,mode="lines",line=dict(color='red',dash="dash"),name='Grok Train Accuracy',showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_losses))),y=grokked_object.test_losses,marker=dict(color='red'),name='Grok test loss',showlegend=True),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.train_losses))),y=grokked_object.train_losses,mode="lines",line=dict(color='red',dash="dash"),name='Grok train loss',showlegend=False),row=1,col=2)
        #fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='blue'),name='Learn Accuracy',showlegend=True),row=2,col=1)
        #fig.add_trace(go.Scatter(x=list(range(len(grokked_object.train_accuracies))),y=grokked_object.train_accuracies,mode="lines",line=dict(color='blue',dash="dash"),name='Learn Train Accuracy',showlegend=False),row=2,col=1)
        fig.update_yaxes(type='log',row=1,col=2)
        count=0
        for i in chosen_indices:
            # print(updates_grokked[0][i][0].shape)
            # print(updates_grokked[0][4][0].shape)
            # print(updates_grokked[0][4][1].shape)
            
            #optupdates=torch.tensor([x[0][i] for x in updates_grokked])
            #print(layerupdates.shape)
            #print([updates_grokked[x][i][0].shape for x in range(len(updates_grokked))][:5])
            
            layer_grad_updates=torch.stack([updates_grokked[x][i][0] for x in range(len(updates_grokked))],dim=0)
            layer_grad_updates_flattened=torch.flatten(layer_grad_updates,start_dim=1)
            
            max_grad_updates=torch.max(layer_grad_updates_flattened,dim=1).values
            min_grad_updates=torch.min(layer_grad_updates_flattened,dim=1).values

            var_grad_updates=torch.var(layer_grad_updates_flattened,dim=1)
            mean_grad_updates=torch.mean(layer_grad_updates_flattened,dim=1)
            print(max_grad_updates.shape,min_grad_updates.shape,var_grad_updates.shape,mean_grad_updates.shape)

            layer_decay_updates=torch.stack([updates_grokked[x][i][1] for x in range(len(updates_grokked))],dim=0)
            layer_decay_updates_flattened=torch.flatten(layer_decay_updates,start_dim=1)
            
            max_decay_updates=torch.max(layer_decay_updates_flattened,dim=1).values
            min_decay_updates=torch.min(layer_decay_updates_flattened,dim=1).values
            var_decay_updates=torch.var(layer_decay_updates_flattened,dim=1)
            mean_decay_updates=torch.mean(layer_decay_updates_flattened,dim=1)


            
            fig.add_trace(go.Scatter(x=epochs,y=max_grad_updates.detach().numpy(),marker=dict(color='red'),name='Max grad'),row=2,col=1)
            fig.add_trace(go.Scatter(x=epochs,y=max_decay_updates.detach().numpy(),marker=dict(color='blue'),name='Max decay'),row=2,col=1)
            fig.update_yaxes(title_text="Max of update",row=2,col=1)
            fig.add_trace(go.Scatter(x=epochs,y=min_grad_updates.detach().numpy(),marker=dict(color='red'),name='Min grad'),row=2,col=2)
            fig.add_trace(go.Scatter(x=epochs,y=min_decay_updates.detach().numpy(),marker=dict(color='blue'),name='Min decay'),row=2,col=2)
            fig.update_yaxes(title_text="Min of update",row=2,col=2)

            fig.add_trace(go.Scatter(x=epochs,y=mean_grad_updates.detach().numpy(),marker=dict(color='red'),name='Min grad'),row=3,col=1)
            fig.add_trace(go.Scatter(x=epochs,y=mean_decay_updates.detach().numpy(),marker=dict(color='blue'),name='Min decay'),row=3,col=1)
            fig.update_yaxes(title_text="Mean of update",row=3,col=1)

            fig.add_trace(go.Scatter(x=epochs,y=var_grad_updates.detach().numpy(),marker=dict(color='red'),name='Min grad'),row=3,col=2)
            fig.add_trace(go.Scatter(x=epochs,y=var_decay_updates.detach().numpy(),marker=dict(color='blue'),name='Min decay'),row=3,col=2)
            fig.update_yaxes(title_text="Var of update",row=3,col=2)

            fig.update_xaxes(title_text='Epochs')
        # fig.update_yaxes(title_text="(Log) Norm", secondary_y=True, type='log',row=1,col=1)
        # fig.update_yaxes(title_text="(Log) Norm", secondary_y=True, type='log',row=1,col=2)
        return fig
    


    def optimizer_plot2(grokked_object,non_grokked_object,chosen_indices):
        updates_grokked=[]
        updates_non_grokked=[]
        epochs=[]
        
        for epoch in tqdm(grokked_object.models.keys()):
            grok_state_dic=grokked_object.models[epoch]['model']
            grok_updates_tensor=optimizer_plots(grokked_object,epoch)
            # if epoch<200:
            #     print(grok_updates_tensor[7][0])
            # else:
            #     exit()
            non_grok_state_dic=non_grokked_object.models[epoch]['model']
            non_grok_updates_tensor=optimizer_plots(non_grokked_object,epoch)

            updates_grokked.append(grok_updates_tensor)
            updates_non_grokked.append(non_grok_updates_tensor)
            epochs.append(epoch)        
            # if epoch in [1000]:
            #     print(torch.max(grok_updates_tensor[4][0]))
            #     print(torch.min(grok_updates_tensor[4][0]))
            #     exit()
            


        #fig=make_subplots(rows=1,cols=2,specs=[[{"secondary_y": True}, {"secondary_y": True}]],subplot_titles=["Grok test accuracy and weight Norm","Learn test accuracy and weight norm"])
        fig=make_subplots(rows=1+2,cols=2)
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_accuracies))),y=grokked_object.test_accuracies,marker=dict(color='red'),name='Grok Test Accuracy',showlegend=True),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.train_accuracies))),y=grokked_object.train_accuracies,mode="lines",line=dict(color='red',dash="dash"),name='Grok Train Accuracy',showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_losses))),y=grokked_object.test_losses,marker=dict(color='red'),name='Grok test loss',showlegend=True),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(grokked_object.train_losses))),y=grokked_object.train_losses,mode="lines",line=dict(color='red',dash="dash"),name='Grok train loss',showlegend=False),row=1,col=2)
        #fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='blue'),name='Learn Accuracy',showlegend=True),row=2,col=1)
        #fig.add_trace(go.Scatter(x=list(range(len(grokked_object.train_accuracies))),y=grokked_object.train_accuracies,mode="lines",line=dict(color='blue',dash="dash"),name='Learn Train Accuracy',showlegend=False),row=2,col=1)
        fig.update_yaxes(type='log',row=1,col=2)
        count=0
        for i in chosen_indices:
            # print(updates_grokked[0][i][0].shape)
            # print(updates_grokked[0][4][0].shape)
            # print(updates_grokked[0][4][1].shape)
            
            #optupdates=torch.tensor([x[0][i] for x in updates_grokked])
            #print(layerupdates.shape)
            #print([updates_grokked[x][i][0].shape for x in range(len(updates_grokked))][:5])
            
            layer_grad_updates=torch.stack([updates_grokked[x][i][0] for x in range(len(updates_grokked))],dim=0)
            layer_grad_updates_flattened=torch.flatten(layer_grad_updates,start_dim=1)
            
            max_grad_updates=torch.max(layer_grad_updates_flattened,dim=1).values
            min_grad_updates=torch.min(layer_grad_updates_flattened,dim=1).values

            var_grad_updates=torch.var(layer_grad_updates_flattened,dim=1)
            mean_grad_updates=torch.mean(layer_grad_updates_flattened,dim=1)
            print(max_grad_updates.shape,min_grad_updates.shape,var_grad_updates.shape,mean_grad_updates.shape)

            layer_decay_updates=torch.stack([updates_grokked[x][i][1] for x in range(len(updates_grokked))],dim=0)
            layer_decay_updates_flattened=torch.flatten(layer_decay_updates,start_dim=1)
            
            max_decay_updates=torch.max(layer_decay_updates_flattened,dim=1).values
            min_decay_updates=torch.min(layer_decay_updates_flattened,dim=1).values
            var_decay_updates=torch.var(layer_decay_updates_flattened,dim=1)
            mean_decay_updates=torch.mean(layer_decay_updates_flattened,dim=1)


            
            fig.add_trace(go.Scatter(x=epochs,y=max_grad_updates.detach().numpy(),marker=dict(color='red'),name='Max grad'),row=2,col=1)
            fig.add_trace(go.Scatter(x=epochs,y=max_decay_updates.detach().numpy(),marker=dict(color='blue'),name='Max decay'),row=2,col=1)
            fig.update_yaxes(title_text="Max of update",row=2,col=1)
            fig.add_trace(go.Scatter(x=epochs,y=min_grad_updates.detach().numpy(),marker=dict(color='red'),name='Min grad'),row=2,col=2)
            fig.add_trace(go.Scatter(x=epochs,y=min_decay_updates.detach().numpy(),marker=dict(color='blue'),name='Min decay'),row=2,col=2)
            fig.update_yaxes(title_text="Min of update",row=2,col=2)

            fig.add_trace(go.Scatter(x=epochs,y=mean_grad_updates.detach().numpy(),marker=dict(color='red'),name='Min grad'),row=3,col=1)
            fig.add_trace(go.Scatter(x=epochs,y=mean_decay_updates.detach().numpy(),marker=dict(color='blue'),name='Min decay'),row=3,col=1)
            fig.update_yaxes(title_text="Mean of update",row=3,col=1)

            fig.add_trace(go.Scatter(x=epochs,y=var_grad_updates.detach().numpy(),marker=dict(color='red'),name='Min grad'),row=3,col=2)
            fig.add_trace(go.Scatter(x=epochs,y=var_decay_updates.detach().numpy(),marker=dict(color='blue'),name='Min decay'),row=3,col=2)
            fig.update_yaxes(title_text="Var of update",row=3,col=2)

            fig.update_xaxes(title_text='Epochs')
        # fig.update_yaxes(title_text="(Log) Norm", secondary_y=True, type='log',row=1,col=1)
        # fig.update_yaxes(title_text="(Log) Norm", secondary_y=True, type='log',row=1,col=2)
        return fig
    #weight_norm_plot(grokked_object=single_run,non_grokked_object=single_run_ng).show()
    #optimizer_plot(single_run,single_run_ng,chosen_indices=[4]).show()


    #single_run.weights_histogram_epochs2(non_grokked_object=single_run_ng)
    
    #single_run.svd_one_epoch_prod(single_run_ng,epoch,None).show()
    # single_run.svd_one_epoch_prod(single_run_ng,epoch,None).show()
    #single_run.svd_epochs(single_run_ng).show()
    #weight_distance_plot(single_run,single_run_ng).show()
    #optimizer_plot(single_run,single_run_ng,chosen_indices=[4]).show()
    

    


    #single_run.make_weights_histogram_modadd(single_run_ng,990,None).show()
    #single_run.svd_one_epoch_prod(single_run_ng,epoch=990,fig=None).show()
    #
    # weight_differences=[]
    # update_differences=[]
    # grad_differences=[]
    # epochs=[]
    # first=True
    # for epoch in tqdm(single_run.models.keys()):
    #     post_update_statedic=single_run.models[epoch]['model']
    #     post_update_statedic_nd=single_run.models[epoch]['decay_update']
    #     pre_update_statedic=single_run.models[epoch]['model_before_opt']

    #     decayed_weights=[post_update_statedic[key] for key in post_update_statedic.keys() if 'weight' in key]
    #     nondecayed_weights=[post_update_statedic_nd[key] for key in post_update_statedic_nd.keys() if 'weight' in key]
    #     pre_update_weights=[pre_update_statedic[key] for key in pre_update_statedic.keys() if 'weight' in key]

    #     difference_weights_wd=[post_update_statedic[key]-post_update_statedic_nd[key] for key in post_update_statedic_nd]
    #     difference_weights_onetensor=torch.cat([torch.flatten(x)**2 for x in difference_weights_wd]).unsqueeze(0)

    #     difference_weights_grad=[post_update_statedic_nd[key]-pre_update_statedic[key] for key in post_update_statedic_nd]
    #     difference_weights_grad_onetensor=torch.cat([torch.flatten(x)**2 for x in difference_weights_grad]).unsqueeze(0)
        

    #     pre_difference_weights=[post_update_statedic[key]-pre_update_statedic[key] for key in post_update_statedic]
    #     update_weights_onetensor=torch.cat([torch.flatten(x)**2 for x in pre_difference_weights]).unsqueeze(0)
        
    #     weight_differences.append(difference_weights_onetensor)
    #     update_differences.append(update_weights_onetensor)
    #     grad_differences.append(difference_weights_grad_onetensor)
    #     # if first:
    #     #     difference_weights_tensor=difference_weights_onetensor
    #     #     first=False
    #     #     continue
    #     # else:
    #     #     difference_weights_tensor=torch.stack((difference_weights_tensor,),dim=0)
    #     epochs.append(epoch)
    # difference_weights_tensor=torch.cat(weight_differences,dim=0)
    # update_weights_tensor=torch.cat(update_differences,dim=0)
    # grad_differences_tensor=torch.cat(grad_differences,dim=0)

    
    
    # max_diff=torch.max(difference_weights_tensor,dim=1)[0]
    # min_diff=torch.min(difference_weights_tensor,dim=1)[0]
    # mean_diff=torch.mean(difference_weights_tensor,dim=1)
    # var_diff=torch.var(difference_weights_tensor,dim=1)

    # max_update=torch.max(update_weights_tensor,dim=1)[0]
    # min_update=torch.min(update_weights_tensor,dim=1)[0]
    # mean_update=torch.mean(update_weights_tensor,dim=1)
    # var_update=torch.var(update_weights_tensor,dim=1)

    # max_grad=torch.max(grad_differences_tensor,dim=1)[0]
    # min_grad=torch.min(grad_differences_tensor,dim=1)[0]
    # mean_grad=torch.mean(grad_differences_tensor,dim=1)
    # var_grad=torch.var(grad_differences_tensor,dim=1)

    # fig=make_subplots(rows=2,cols=2,specs=[[{"secondary_y": True}, {"secondary_y": True}],[{"secondary_y": True}, {"secondary_y": True}]],subplot_titles=['Model update due to weight decay and grad - mean','Model update due to weight decay and grad - var','Model update due to weight decay and grad - Max','Model update due to weight decay and grad - min'])

    # fig.add_trace(go.Scatter(x=epochs,y=mean_diff.detach().numpy(),marker=dict(color='black'),name='Mean decay'),row=1,col=1)
    # fig.add_trace(go.Scatter(x=epochs,y=mean_update.detach().numpy(),marker=dict(color='orange'),name='Mean total'),row=1,col=1)
    # fig.add_trace(go.Scatter(x=epochs,y=mean_grad.detach().numpy(),marker=dict(color='purple'),name='Mean grad'),row=1,col=1)
    # fig.add_trace(go.Scatter(x=list(range(len(single_run.test_accuracies))),y=single_run.test_accuracies,mode="lines",line=dict(color='red',dash="dash"),name='Grok test acc',showlegend=True),row=1,col=1,secondary_y=True)
    # fig.update_yaxes(title_text='Mean gradient update',row=1,col=1)

    # fig.add_trace(go.Scatter(x=epochs,y=var_diff.detach().numpy(),marker=dict(color='black'),name='Var decay'),row=1,col=2)
    # fig.add_trace(go.Scatter(x=epochs,y=var_update.detach().numpy(),marker=dict(color='orange'),name='Var total'),row=1,col=2)
    # fig.add_trace(go.Scatter(x=epochs,y=var_grad.detach().numpy(),marker=dict(color='purple'),name='Var grad'),row=1,col=2)
    # fig.add_trace(go.Scatter(x=list(range(len(single_run.test_accuracies))),y=single_run.test_accuracies,mode="lines",line=dict(color='red',dash="dash"),name='Grok test acc',showlegend=True),row=1,col=2,secondary_y=True)
    # fig.update_yaxes(title_text='Var gradient update',row=1,col=2)


    # fig.add_trace(go.Scatter(x=epochs,y=max_diff.detach().numpy(),marker=dict(color='black'),name='Max decay'),row=2,col=1)
    # fig.add_trace(go.Scatter(x=epochs,y=max_update.detach().numpy(),marker=dict(color='orange'),name='Max total'),row=2,col=1)
    # fig.add_trace(go.Scatter(x=epochs,y=max_grad.detach().numpy(),marker=dict(color='purple'),name='Max grad'),row=2,col=1)
    # fig.add_trace(go.Scatter(x=list(range(len(single_run.test_accuracies))),y=single_run.test_accuracies,mode="lines",line=dict(color='red',dash="dash"),name='Grok test acc',showlegend=True),row=2,col=1,secondary_y=True)
    # fig.update_yaxes(title_text='Max gradient update',row=2,col=1)
    

    # fig.add_trace(go.Scatter(x=epochs,y=min_diff.detach().numpy(),marker=dict(color='black'),name='Min decay'),row=2,col=2)
    # fig.add_trace(go.Scatter(x=epochs,y=min_update.detach().numpy(),marker=dict(color='orange'),name='Min total'),row=2,col=2)
    # fig.add_trace(go.Scatter(x=epochs,y=min_grad.detach().numpy(),marker=dict(color='purple'),name='Grad update'),row=2,col=2)
    # fig.add_trace(go.Scatter(x=list(range(len(single_run.test_accuracies))),y=single_run.test_accuracies,mode="lines",line=dict(color='red',dash="dash"),name='Grok test acc',showlegend=True),row=2,col=2,secondary_y=True)
    # fig.update_yaxes(title_text='Min gradient update',row=2,col=2)

    
    # fig.show()
    exit()
    
    exit()
    # epoch=99900
    # feature_funcs=[functools.partial(compute_energy_torch_batch,J=1),magnetization]
    # test_dataset=generate_test_set(dataset,1000)
    # layernames=['fc_layers.0 (Linear)']
    # chosen_neuron_index=0
    # correlation_one_epoch_prod(grokked_object=single_run,non_grokked_object=single_run_ng, images_tensor=test_dataset[0],
    # epoch=epoch,feature_funcs=feature_funcs,sortby='var',dataset=test_dataset,layernames=layernames,chosen_neuron_index=chosen_neuron_index).show()
    
    #single_run.make_weights_histogram_all2(single_run_ng,epoch,None).show()
    #
    #Use the code
    
    #single_run.plot_traincurves(single_run_ng).show()


    exit()
    
    # #
    # # print(len(single_run.train_losses))
    # # print(single_run.test_losses[-1])
    # # print(single_run.test_accuracies[0])
    # # print(single_run.train_accuracies[0])
    # # single_run.svd_one_epoch(single_run_ng,epoch=19000,fig=None).show()
    

    # single_run.svd_epochs(single_run_ng).show()

    
    
    # #single_run.pca_one_epoch(single_run_ng,epoch=19900,fig=None).show()
    # #single_run.pca_epochs(single_run_ng).show()

    # #Let's see if we can do some pruning. Let's just try to do PCA-based last layer pruning.
    # #1. Load the model
    # #2. access last layer weights
    # #3. do svd
    # #4. Test the accuracy
    # features_list=[functools.partial(compute_energy_torch_batch,J=1),magnetization]#largest_component_sizes
    # #correlation_one_epoch(single_run=single_run,single_run_ng=single_run_ng,sortby='var',epoch=99900,neuron_index=0,images_tensor=test[0],feature_funcs=features_list,fig=None).show()
    # #single_run.correlation_one_epoch(single_run_ng=single_run_ng,sortby='var',epoch=99900,neuron_index=0,images_tensor=test[0],feature_funcs=features_list,fig=None,dataset=test).show()
    # #single_run.correlation_epochs(non_grokked_object=single_run_ng,sortby='var',epoch=99900,neuron_index=0,images_tensor=test[0],feature_funcs=features_list,dataset=test).show()
    # single_run.activations_epochs(single_run_ng,sortby='var',dataset=dataset).show()
    # single_run.activations_epochs(single_run_ng,sortby='all',dataset=dataset).show()
    # single_run.correlation_epochs(non_grokked_object=single_run_ng,sortby='var',neuron_index=0,images_tensor=test[0],feature_funcs=[functools.partial(compute_energy_torch_batch,J=1),magnetization],dataset=test).show()
    exit()







