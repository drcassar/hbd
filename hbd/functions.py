import re
import os
import pickle
import pandas as pd

### Chemistry


def composition2atoms(cstr):
    lst = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', cstr)
    dic = {}
    for i in lst:
        if len(i[1]) > 0:
            try:
                dic[i[0]] = int(i[1])
            except ValueError:
                dic[i[0]] = float(i[1])
        else:
            dic[i[0]] = 1
    return dic


def compoundDF2atomsDF(df_, sumtotal=100):

    df = copy(df_)
    allatoms = set([])
    compounds = df.columns.values

    for compound in compounds:

        atoms = composition2atoms(compound)
        values = df[compound]

        for key in atoms:
            allatoms = allatoms | set([key])

            if key not in df.columns:
                df[key] = np.zeros(len(df))
            df[key] = df[key].values + atoms[key] * df[compound].values

    df = df.reindex(list(sorted(allatoms)), axis='columns')

    # Sum of components must be = somtotal
    soma = df.sum(axis=1)
    df = df.divide(soma, axis=0)
    df = df.multiply(sumtotal, axis=0)

    return df


def loadmodel(
    DATA_LOAD_PATH,
    SEARCH_SPACE_PATH,
    MODEL_PATH,
    TUNNING_PATH,
    test=False,
):

    from tensorflow.keras.models import load_model

    ### Config

    ID = ''
    for path in [DATA_LOAD_PATH, SEARCH_SPACE_PATH, MODEL_PATH, TUNNING_PATH]:
        base = os.path.basename(path)
        ID += os.path.splitext(base)[0] + '-'
    ID = ID[:-1]
    if test:
        FINAL_MODEL_PATH = rf"./results_models/{ID}_model_test.h5"
        SUPPORT_MODEL_PATH = rf"./results_models/{ID}_support_test.p"

    else:
        FINAL_MODEL_PATH = rf"./results_models/{ID}_model.h5"
        SUPPORT_MODEL_PATH = rf"./results_models/{ID}_support.p"

    ### Final model

    X, y, X_features, y_features, X_scaler, y_scaler, best_space = pickle.load(
        open(SUPPORT_MODEL_PATH, 'rb'))
    model = load_model(FINAL_MODEL_PATH)

    ### Function

    def evalfun_x(X):
        X_scaled = X_scaler.transform(X)
        y_scaled = model.predict(X_scaled)
        y = y_scaler.inverse_transform(y_scaled)
        return y

    def evalfun_atomdf(df):

        atoms = set(df.columns)
        trained_atoms = set(X_features)

        if atoms.issubset(trained_atoms):
            x = df.reindex(X_features, axis='columns', fill_value=0).values
            y = evalfun_x(x)
            return y

        else:
            raise ValueError('Some chemicals are not in the training domain')

    def evalfun_compounddf(df):
        atomdf = compoundDF2atomsDF(df, sumtotal=1)
        y = evalfun_atomdf(atomdf)
        return y

    def evalfun_dic(dic):
        try:
            compdf = pd.DataFrame(dic)
        except ValueError:
            dic_ = {a: [b] for a, b in zip(dic.keys(), dic.values())}
            compdf = pd.DataFrame(dic_)

        atomdf = compoundDF2atomsDF(compdf, sumtotal=1)
        y = evalfun_atomdf(atomdf)
        return y

    model_dic = {
        'model': model,
        'evalfun_x': evalfun_x,
        'evalfun_atomdf': evalfun_atomdf,
        'evalfun_compounddf': evalfun_compounddf,
        'evalfun_dic': evalfun_dic,
        'X': X,
        'y': y,
        'X_features': X_features,
        'y_features': y_features,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'best_space': best_space,
    }

    return model_dic
