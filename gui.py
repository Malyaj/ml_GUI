# -*- coding: utf-8 -*-

import PySimpleGUI as sg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
#import matplotlib.pyplot as plt
'''

def train_function(path, model, score):
    if path is None:
        path_to_file = None
    else:
        path_to_file = path

    data = pd.read_excel(path_to_file, sheet_name='Sheet1', parse_dates = [])
    
    ## removing outliers
    lim = 99
    threshold = np.percentile(data[['cost']],lim)
    data['mask'] = data[['cost']] < threshold
    data = data[data['mask'] == True]
    del data['mask']


    ## log transforms
    data['po1_log'] = np.log(data['po1_fe_hrs'][data['po1_fe_hrs'] != 0])
    data['po2_log'] = np.log(data['po2_fe_hrs'][data['po2_fe_hrs'] != 0])
    data['po3_log'] = np.log(data['po3_fe_hrs'][data['po3_fe_hrs'] != 0])
    
    data['po1_ind_log'] = np.log(data['po1_indirect_labour_hrs'][data['po1_indirect_labour_hrs'] != 0])
    data['po2_ind_log'] = np.log(data['po2_indirect_labour_hrs'][data['po2_indirect_labour_hrs'] != 0])
    data['po3_ind_log'] = np.log(data['po3_indirect_labour_hrs'][data['po3_indirect_labour_hrs'] != 0])
    
    data['po1_crf_log'] = np.log(data['po1_craft_hrs'][data['po1_craft_hrs'] != 0])
    data['po2_crf_log'] = np.log(data['po2_craft_hrs'][data['po2_craft_hrs'] != 0])
    data['po3_crf_log'] = np.log(data['po3_craft_hrs'][data['po3_craft_hrs'] != 0])




    ## one-hot encoding categorical features
    categorical_cols = ['brand', 'region', 'technology', 'outage_type',]
    for cat in categorical_cols:
        dummy = pd.get_dummies(data[cat])
        
        for each in dummy.columns:
            data[each] = dummy[each]
    ### deleting the actual categorical features
    for each in categorical_cols:
        del data[each]


    dummy_cols = ['po1_tooling_flag', 'po1_other_flag', 
                  'po2_tooling_flag', 'po2_other_flag', 
                  'po3_tooling_flag', 'po3_other_flag', 
                  'brand_2', 'brand_1', 'ANZ',
           'Asia', 'Europe', 'LATAM', 'MEA', 'US', 'A class', 'CI', 'D Class',
           'GT', 'HGP', 'MI', 'Non A&D', 'Non D or A Class', 'Non-GE HD Gas',
           'Other', 'type_5', 'type_6', 'type_7', 'type_9', 'C', 'Major', 'Minor']


    ## 'COST (USD)', 'PO1_log', 'PO2_log'
    continuous_features = ['po1_log', 'po2_log', 'po3_log', 
                           'po1_ind_log', 'po2_ind_log', 'po3_ind_log', 
                           'po1_crf_log', 'po2_crf_log', 'po3_crf_log', 
                           'outage_duration'
                           ]

    df = data[dummy_cols + continuous_features + ['cost']]
    
    df.columns = dummy_cols + continuous_features + ['cost']
    
    ## missing value treatment
    df = df.fillna(0)



    """end of changes"""
    X = df[dummy_cols + continuous_features]
    y = df[['cost']]
    #y = np.ravel(y)
    
    
    # transforming the target
    y = np.log(y)



    ### let's visualize this
    #plt.scatter(df[['PO1_log']], df[['cost']], c = 'red')
    #plt.scatter(df[['PO2_log']], df[['cost']], c = 'blue')



    ### splitting the data into train and test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    ### scale the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Fit only to the training data
    scaler.fit(X_train)
    # Now apply the transformations to the data:
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)



    ### instantiating the model
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    
    neural_network = MLPRegressor(hidden_layer_sizes=(5,5,5), max_iter=5000)
    random_forest  = RandomForestRegressor(random_state = 7)
    linear_regression = LinearRegression()
    #regressor = SVR(kernel = 'linear')
    support_vector_machine = SVR(kernel = 'rbf')

    ### model instantiation
    if model == 'linear regression':
        model = linear_regression
    elif model == 'random forest':
        model = random_forest
    elif model == 'svm':
        model = support_vector_machine
    elif model == 'neural network':
        model = neural_network
    else:  ### default
        model = support_vector_machine

    
    ### train the model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    def accuracy(y_test, y_pred, hold = 5):
        y_pred = y_pred.reshape(-1,1)

        #if y_pred.shape != y_test.shape:
        if y_pred.shape != y_test.values.shape:
            print("Incorrect args")
            return None
        
        matches = []
        #for i in range(len(y_test)):
        for i in range(len(y_test.values)):
            LSL = (1 - hold / 100) * y_test.values[i]
            USL = (1 + hold / 100) * y_test.values[i]
            if (y_pred[i] > LSL) and (y_pred[i] < USL):
                matches.append(True)
            else:
                matches.append(False)
                
        return np.sum(matches) / len(matches)


    def rmsle(y_test, y_pred):
        y_test = y_test.values
        s = 0
        for i in  range(0,len(y_pred)):
            s += np.power(np.log(float(y_pred[i]) + 1) - np.log(float(y_test[i]) + 1),2)
        s = s * (1/len(y_pred))
        return float(np.sqrt(s))

    if score == 'accuracy':
        scoring_fucntion = accuracy
    elif score == 'rmsle':
        scoring_fucntion = rmsle
    else:
        ## default
        scoring_fucntion = accuracy

    score_result = scoring_fucntion(y_test, y_pred)
    
    return model, scaler, score_result


###############################################################################

def predict_function(path, trained_model, trained_scaler):
    data = pd.read_excel(path, sheet_name='Sheet1', parse_dates = [])
    
    ### log transforms
    data['po1_log'] = np.log(data['po1_fe_hrs'][data['po1_fe_hrs'] != 0])
    data['po2_log'] = np.log(data['po2_fe_hrs'][data['po2_fe_hrs'] != 0])
    data['po3_log'] = np.log(data['po3_fe_hrs'][data['po3_fe_hrs'] != 0])
    
    data['po1_ind_log'] = np.log(data['po1_indirect_labour_hrs'][data['po1_indirect_labour_hrs'] != 0])
    data['po2_ind_log'] = np.log(data['po2_indirect_labour_hrs'][data['po2_indirect_labour_hrs'] != 0])
    data['po3_ind_log'] = np.log(data['po3_indirect_labour_hrs'][data['po3_indirect_labour_hrs'] != 0])
    
    data['po1_crf_log'] = np.log(data['po1_craft_hrs'][data['po1_craft_hrs'] != 0])
    data['po2_crf_log'] = np.log(data['po2_craft_hrs'][data['po2_craft_hrs'] != 0])
    data['po3_crf_log'] = np.log(data['po3_craft_hrs'][data['po3_craft_hrs'] != 0])

    ### one-hot encoding categorical features
    brand_dummy_cols = list(set(data['brand'].values))
    region_dummy_cols = list(set(data['region'].values))
    technology_dummy_cols = list(set(data['technology'].values))
    outage_type_dummy_cols = list(set(data['outage_type'].values))
    
    
    categorical_cols = ['brand', 'region', 'technology', 'outage_type',]
    for cat in categorical_cols:
        dummy = pd.get_dummies(data[cat])
        
        for each in dummy.columns:
            data[each] = dummy[each]
    ### deleting the actual categorical features
    for each in categorical_cols:
        del data[each]

    

    
    
    flag_dummy_cols = ['po1_tooling_flag', 'po1_other_flag', 
                  'po2_tooling_flag', 'po2_other_flag', 
                  'po3_tooling_flag', 'po3_other_flag'] 
    
    dummy_cols = flag_dummy_cols + brand_dummy_cols + region_dummy_cols + technology_dummy_cols + outage_type_dummy_cols 
                  
    ## 'COST (USD)', 'PO1_log', 'PO2_log'
    continuous_features = ['po1_log', 'po2_log', 'po3_log', 
                           'po1_ind_log', 'po2_ind_log', 'po3_ind_log', 
                           'po1_crf_log', 'po2_crf_log', 'po3_crf_log', 
                           ]

    df = data[dummy_cols + continuous_features]
    
    #df.columns = dummy_cols + continuous_features
    
    ## missing value treatment
    df = df.fillna(0)
    
    ### check for missing features and fill 0
    all_dummy_cols = ['po1_tooling_flag', 'po1_other_flag', 
                  'po2_tooling_flag', 'po2_other_flag', 
                  'po3_tooling_flag', 'po3_other_flag', 
                  'brand_2', 'brand_1', 'ANZ',
           'Asia', 'Europe', 'LATAM', 'MEA', 'US', 'A class', 'CI', 'D Class',
           'GT', 'HGP', 'MI', 'Non A&D', 'Non D or A Class', 'Non-GE HD Gas',
           'Other', 'type_5', 'type_6', 'type_7', 'type_9', 'C', 'Major', 'Minor']
    for each in df.columns:
        if each not in all_dummy_cols:
            df[each] = 0



    """end of changes"""
    X = df[dummy_cols + continuous_features]
    
    

    

    ### scale the data with the trained scaler
    # Now apply the transformations to the data:
    X = trained_scaler.transform(X)
    
    y = trained_model.predict(X)
    y = np.exp(y)
    
    df['cost'] = y
    
    return df
###############################################################################


path_to_image = r"D:\Users\703143501\Documents\Genpact Internal\GB\Product\logo.png"
#import PySimpleGUI as sg
 
sg.ChangeLookAndFeel('DarkBlue')
sg.SetOptions(text_justification='right', tooltip_time= 1)


###
##menu_def = [['&File', ['&Select', ['Train Data', 'Unseen Data'], 'Properties', 'Exit']],
##            ['&Help', ['&About'],]]

menu_def = [['&File', ['Properties', 'Exit']],
            ['&Help', ['&About'],]]

layout = [ 
        [sg.Menu(menu_def, tearoff=False)],
        [sg.Txt('This is a tool to train a machine learning model on event data and predict the cost for unseen data')], 
            [sg.Image(path_to_image)], 
            [sg.InputText('Train data', key='_train_path_'), sg.FileBrowse()], 
            #[sg.InputText('Unseen data', key='_unseen_path_'), sg.FileBrowse()], 
            [sg.Text('choose model'), sg.Radio('linear regression', group_id='model', key='_lr_'), sg.Radio('support vector machine', group_id='model', key='_svm_'), sg.Radio('neural network', group_id='model', key='_nn_'), sg.Radio('random forest', group_id='model', key='_rf_')], 
            [sg.Text('choose scoring metric'), sg.Radio('accuracy', group_id='score', key='_acc_'), sg.Radio('rmsle', group_id='score', key='_rmsle_')],
            [sg.RButton('Train', disabled=False)],  #, sg.RButton('Predict', disabled=False),sg.Exit()],
            [sg.InputText('Unseen data', key='_unseen_path_'), sg.FileBrowse()], 
            [sg.Text('Accuracy'), sg.Text('         ', key='_SCORE_')],
            [sg.RButton('Predict', disabled=False), sg.Exit()],
            [sg.RButton('About', tooltip='tool description', size=(150,1), auto_size_button=True)],
            #[canvas_object], 
        ]


window = sg.Window('Event Cost Prediction Tool', grab_anywhere=True).Layout(layout)
#window.UseDictionary = False
window.BackgroundColor = 'black'

window.Resizable = True
window.TextJustification = True



model, score = None, None

while True:
    
    
    event, values = window.Read()
    if event == 'Properties':
        if model is None or score is None:
            sg.Popup("Model not configured properly!")
        else:
            sg.Popup(f"Model: {model} , Scoring Metric: {score}")
    
    if event is None or event == 'Exit':      
        break      
    if event == 'Train':
        path_to_train_data = values['_train_path_']
        
        ### model value
        if values['_lr_'] == True:
            model = 'linear regression'
        if values['_svm_'] == True:
            model = 'svm'
        if values['_nn_'] == True:
            model = 'neural network'
        if values['_rf_'] == True:
            model = 'random forest'
        
        ## score value
        if values['_acc_'] == True:
            score = 'accuracy'
        if values['_rmsle_'] == True:
            score = 'rmsle'
        
        ### call train fucntion
        trained_model, trained_scaler, score_result = train_function(path_to_train_data, model, score)
        window.FindElement('_SCORE_').Update(score_result)
    
    if event == 'About':
        sg.Popup("This tool trains a Machine Learning Algorithm on event data and predicts cost for unseen events")
    
    if event == 'Predict':
        
        ### pop up a window to choose between individual entries or from file
        path_to_unseen_data = values['_unseen_path_']
        original_data = pd.read_excel(path_to_unseen_data, sheet_name='Sheet1', parse_dates = [])
        
        df = predict_function(path_to_unseen_data, trained_model, trained_scaler)
        y = df['cost']
        
        original_data['cost'] = y
        
        path_to_write = r"D:\Users\703143501\Documents\Genpact Internal\GB\Product\result.csv"
        original_data.to_csv(path_to_write, index = None)
        

        ## plot the image
        data = pd.read_csv(path_to_write)

        index = np.arange(len(data['WORKFLOW ID'].values))
        plt.xlabel('WF#', fontsize=12, rotation = 30)
        plt.ylabel('Cost', fontsize=12)
        plt.xticks(index, fontsize=12, rotation=30)
        plt.title(" Cost Estimate ")

        p = plt.bar(index, np.array(data['cost'].values))
        plt.legend(p, loc=2, fontsize=8)
        plt.show()

        sg.Popup("                         Result written !                         ")

window.Close()
