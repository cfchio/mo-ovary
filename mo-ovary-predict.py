import pandas as pd
import joblib
import argparse
import json

feature_names = [['Age','Menopause','US1','US2','US5','US6','US7','CA125','BUN','ALT','AAR','WBC','Lym','PLR'],
                 ['Age','Menopause','US1','US2','US4','US5','US6','US7','CA125','K','RBC','Lym','PLR'],
                 ['Age','Menopause','US1','US2','US4','US5','US6','US7','CA125','K','ALT','WBC','Lym','PLR']
                ]
rmi_names = ['Menopause','US1','US2','US3','US4','US5','US6','CA125']

class MO_Ovary_Model:
    def __init__(self, model_dict):
        self.paras = model_dict
        self.model_name = self.paras["model_name"]
        self.model = joblib.load(self.paras["model_path"])
        self.scaler = joblib.load(self.paras["scaler_path"])
        self.mean = joblib.load(self.paras["mean_path"])
        self.feature_names = feature_names[self.paras["features"]]
        self.cutoff = self.mean['cutoff']
    
    def init_shap(self):
        self.explainer = joblib.load(self.paras["shap_path"])


def CountRMI4(input):
    score_M = [1, 4]
    score_U = [1 ,1, 4, 4, 4, 4]
    score_S = [1, 2]

    M = score_M[int(input[0])]
    U = score_U[int(sum(input[1:6]))]
    S = score_S[int(input[6])]
    CA125 = input[7]

    return int(M * U * CA125 * S)


def SplitStringToDict(info_input):
    ii = info_input.split(',')
    v = {}
    for i in ii:
        e = i.split(':')
        if(len(e)==2):
            v[e[0].strip()] = float(e[1].strip())
    return v


def ExtractValuesFromDict(info_dict, key_name):
    ret = []
    for k in key_name:
        ret.append(info_dict.get(k))
            
    return ret


def PrepareDataInput(info_input, model, input_format):
    input_names = model.feature_names.copy()
    #Features that are always needed
    if 'US3' not in input_names:
        input_names.insert(4, 'US3')
    if 'US4' not in input_names:
        input_names.insert(4, 'US4')
    if 'AST' not in input_names:
        input_names.insert(-1, 'AST')
    if 'Plt' not in input_names:
        input_names.insert(-1, 'Plt')

    dft = pd.DataFrame(columns=input_names)
    if input_format == "csv":
        dft = pd.read_csv(info_input)
    elif input_format == "xls":
        dft = pd.read_excel(info_input)
    else:
        dft.loc[len(dft.index)] = ExtractValuesFromDict(SplitStringToDict(info_input), input_names)

    if ('Plt' in dft.columns) and ('Lym' in dft.columns):
        dft['PLR'] = dft['Plt'] / dft['Lym']
    else:
        dft['PLR'] = float("nan")

    if ('ALT' in dft.columns) and ('AST' in dft.columns):
        dft['AAR'] = dft['AST'] / dft['ALT']
    else:
        dft['AAR'] = float("nan")
    
    for c in model.feature_names:
        dft.fillna({c:model.mean[c]}, inplace=True)

    return dft


def PrintPrediction(df):
    for i, r in enumerate(df.iterrows()):
        print('#%-2d Borderline/Malignant likelihood = %.4f %s risk' %
                (i, r[1]['VTP'], ('High' if r[1]['VT']==1 else 'Low')))
        print('    RMI-4 = %.1f %s risk' % 
                (r[1]['RMI4'], ('High' if r[1]['R4']==1 else 'Low')))


def DoPrediction(df_input, model):
    x_test = model.scaler.transform(df_input[model.feature_names])
    y_test_proba = model.model.predict_proba(x_test)

    df_output = pd.DataFrame()
    if 'ID' in df_input.columns:
        df_output['ID'] = list(df_input['ID'])
    df_output['VTP'] = y_test_proba[:,1]

    r4s = []
    for r in df_input.iterrows():
        r4s.append(CountRMI4(r[1][rmi_names].to_list()))
    df_output['RMI4'] = r4s

    if 'Type' in df_input.columns:
            df_output['Type_ID'] = df_input['Type'].apply(lambda x: 0 if x=='Benign' else 1)
    df_output['VT'] = df_output['VTP'].apply(lambda x: 1 if x>=model.cutoff else 0)
    df_output['R4'] = df_output['RMI4'].apply(lambda x: 1 if x>=450 else 0)

    if 'Type' in df_input.columns:
        df_output['Type'] = df_input['Type']
    if 'Dx' in df_input.columns:
        df_output['Dx'] = df_input['Dx']

    return df_output


def DoSHAP(df_input, fn_output, model):
    import shap

    x_test = model.scaler.transform(df_input[model.feature_names])

    model.init_shap()
    shap_values = model.explainer.shap_values(x_test)
    plot = shap.plots.force(model.explainer.expected_value[1], shap_values[0][:,1],
                            model.scaler.inverse_transform(x_test), 
                            feature_names=model.feature_names, link="logit")
    shap.save_html(fn_output, plot)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model', type=str, default="model-23-01-sk132.json")
    args = parser.parse_args()

    with open(args.model) as f:
        model_config = json.load(f)
   
    m = MO_Ovary_Model(model_config["model"][0])
    
    print("-------------------------------------")
    print("Model: %s" % m.model_name)
    print("-------------------------------------")
    
    if args.input.lower().endswith(".csv"):
        fmt = "csv"
    elif args.input.lower().endswith(".xls") or args.input.lower().endswith(".xlsx"):
        fmt = "xls"
    else:
        fmt = "manual"
    
    df = PrepareDataInput(args.input, m, fmt)
    print(df[m.feature_names])
    print()
   
    df_output = DoPrediction(df, m)

    if fmt == "manual":
        PrintPrediction(df_output)
    else:
        print(df_output.to_string())
        

    if args.output is not None:
        if args.output == 'shap':
            print('\n===== [ SHAP Analysis ] =====')
            if fmt == "manual":
                fn_out = 'shap-forceplot.html'
                DoSHAP(df, fn_out, m)
                print('==> Output to '+fn_out)
            else:
                print("SHAP analysis is NOT available for batch input!")
        elif args.output.lower().endswith(".csv"):
            df_output.to_csv(args.output, index=False)
        elif args.output.lower().endswith(".xls") or args.output.lower().endswith(".xlsx"):
            df_output.to_excel(args.output, index=False)

    print()
