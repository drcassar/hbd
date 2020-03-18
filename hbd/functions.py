import re
import os
import pickle
import pandas as pd
import numpy as np
import random
from pprint import pprint
from deap import base, creator, tools
import copy


def createNewDic(dic, multiplyby):
    values = list(dic.values())
    keys = dic.keys()
    newValues = np.array(values) * multiplyby
    newDic = dict(zip(keys, newValues))
    return newDic


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


def compound2atoms(compounds):
    dic = {}
    for key in compounds.keys():
        baseValue = compounds[key]
        atoms = composition2atoms(key)
        for a in atoms.keys():
            dic[a] = dic.get(a, 0) + atoms[a] * baseValue
    return dic


def atoms2atomsF(atoms):
    multiplyby = 1 / np.sum(list(atoms.values()))
    atomsF = createNewDic(atoms, multiplyby)
    return atomsF


def individual2atomF(individual, compoundList):
    compoundDic = dict(zip(compoundList, individual))
    atomsDic = compound2atoms(compoundDic)
    atomsFractionDic = atoms2atomsF(atomsDic)
    return atomsFractionDic


def compoundDF2atomsDF(df_, sumtotal=100):

    df = copy.deepcopy(df_)
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


### Load model


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
        FINAL_MODEL_PATH = rf"{ID}_model_test.h5"
        SUPPORT_MODEL_PATH = rf"{ID}_support_test.p"

    else:
        FINAL_MODEL_PATH = rf"{ID}_model.h5"
        SUPPORT_MODEL_PATH = rf"{ID}_support.p"

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
            return np.nan

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


### GA final


def getAtoms(*data):
    data = list(data)
    commonAtoms = set(data.pop(0).columns.values)
    allAtoms = copy.deepcopy(commonAtoms)
    for dat in data:
        commonAtoms = commonAtoms & set(dat.columns.values)
        allAtoms = allAtoms | set(dat.columns.values)
    return commonAtoms, allAtoms


def getMinMaxAtom(atom, *datas):
    minAtom = np.inf
    maxAtom = -np.inf
    for data in datas:
        minAtom = min(minAtom, min(data[atom]))
        maxAtom = max(maxAtom, max(data[atom]))
    return minAtom, maxAtom


def minmaxdic(force_atom_range={}, *datas):
    commonAtoms, _ = getAtoms(*datas)
    result = {}
    for atom in sorted(commonAtoms):
        if atom in force_atom_range:
            minAtom, maxAtom = force_atom_range[atom]
        else:
            minAtom, maxAtom = getMinMaxAtom(atom, *datas)
        result[atom] = (minAtom, maxAtom)
    return result


def getCompounds(atomsLst, referenceAtom, forbiddenAtoms=[], forceAtomAdd=[]):

    # global forbidden_compounds

    compounds = set([
        'SiO2', 'Al2O3', 'B2O3', 'BaO', 'BeO', 'Bi2O3', 'CaO', 'CdO', 'CeO2',
        'Ce2O3', 'Cs2O', 'GeO2', 'HfO2', 'K2O', 'La2O3', 'Li2O', 'MgO', 'Na2O',
        'Nb2O5', 'Nd2O3', 'P2O5', 'PbO', 'Rb2O', 'Sm2O3', 'SnO', 'SrO',
        'Ta2O5', 'TiO2', 'Tl2O', 'V2O5', 'ZnO', 'Ag2O', 'Dy2O3', 'Er2O3',
        'Eu2O3', 'Gd2O3', 'Ga2O3', 'H2O', 'Fe2O3', 'R2O', 'RO', 'R2O3', 'RO2',
        'R2O5', 'Sc2O3', 'Y2O3', 'In2O3', 'SO3', 'Cu2O', 'WO3', 'NiO', 'MoO3',
        'Cr2O3', 'VO2', 'ThO2', 'Mn2O3', 'MnO2', 'As2O5', 'CoO', 'Sb2O3',
        'CuO', 'TeO2', 'SnO2', 'SO2', 'UO2', 'UO3', 'Fe3O4', 'CrO3', 'FeO',
        'MnO', 'Pr2O3', 'Tb2O3', 'Ho2O3', 'Yb2O3', 'Sb2O5', 'Co3O4', 'RO3',
        'As2O3', 'Mn3O4', 'Ni2O3', 'Mo2O3', 'Co2O3', 'NH4NO3', 'U2O5', 'OH',
        'TeO3', 'Pr6O11', 'Lu2O3', 'PrO2', 'TbO2', 'SeO2', 'Mn2O7', 'MoO',
        'P2O3', 'SrF2', 'CrF3', 'U3O8', 'Tl2O3', 'NO2', 'SbO2', 'Pb3O4',
        'PbO2', 'SiO', 'Sn2O3', 'GeO', 'N2O5', 'Ti2O3', 'Tm2O3', 'HgO',
        'Nb2O3', 'Au2O', 'PtO2', 'Re2O7', 'RuO2', 'EuO', 'Ta2O3', 'V2O3',
        'Tb3O7', 'Tb4O7', 'SeO3', 'Re2O3', 'VO6', 'MoO2', 'Mo2O5', 'Cr3O4',
        'Hg2O', 'AmO2', 'NpO2', 'PuO2', 'ZrO2', 'PdO', 'TiO', 'PtO'
    ])

    forceadd = set(forceAtomAdd)

    finalcomps = []

    for comp in compounds - forbidden_compounds:
        atoms = composition2atoms(comp)
        if referenceAtom in atoms:
            isbad = False
            for a in atoms:
                if (a not in atomsLst) or (a in forbiddenAtoms):
                    isbad = True

            if not isbad:
                finalcomps.append(comp)

    atomset = set(atomsLst) - set(forbiddenAtoms)

    for comp in finalcomps:
        atoms = composition2atoms(comp)
        a = set(atoms.keys())
        atomset = atomset - a

    finallist = sorted(list(set(finalcomps) | atomset | forceadd))

    return finallist


def score(value1, value2, desiredValue1, desiredValue2, weight1, weight2):
    '''Quanto menor este score, melhor'''

    x0 = value1
    y0 = value2
    x1 = desiredValue1
    y1 = desiredValue2
    distance = (weight1 * (x0 - x1)**2 + weight2 * (y0 - y1)**2)**(1 / 2)
    return distance


def individualInsideDomain(individual):
    atomsF = individual2atomF(individual, compoundList)

    for a in atomsF:
        constrain = minmaxdictionary.get(a, (0, 0))

        min_fraction = constrain[0] * ((100 - relax) / 100)
        max_fraction = constrain[1] * ((100 + relax) / 100)

        if not min_fraction <= atomsF[a] <= max_fraction:
            return False

    return True


def distance_individualInsideDomain(individual):
    atomsF = individual2atomF(individual, compoundList)

    distance = 0

    for a in atomsF:
        constrain = minmaxdictionary.get(a, (0, 0))

        min_fraction = constrain[0] * ((100 - relax) / 100)
        max_fraction = constrain[1] * ((100 + relax) / 100)

        if not min_fraction <= atomsF[a] <= max_fraction:
            if atomsF[a] < min_fraction:
                distance += abs(atomsF[a] - min_fraction)

            else:
                distance += abs(atomsF[a] - max_fraction)

    if distance < 1:
        return (1 / distance)**2
    else:
        return distance**2


def sufficientNumberOfFormers(individual):

    # global formers
    # global MINFRACTIONOFFORMERS

    sumOfFormers = 0
    compDic = dict(zip(compoundList, individual))
    sumOfCompounds = sum(compDic.values())

    if sumOfCompounds > 0:
        for f in formers:
            sumOfFormers += compDic.get(f, 0)

        former_fraction = sumOfFormers / sumOfCompounds

        if former_fraction >= MINFRACTIONOFFORMERS:
            return True
        else:
            return False

    else:
        return False


def distance_sufficientNumberOfFormers(individual):

    # global formers
    # global MINFRACTIONOFFORMERS
    # global compoundList

    sumOfFormers = 0
    compDic = dict(zip(compoundList, individual))
    sumOfCompounds = sum(compDic.values())

    for f in formers:
        sumOfFormers += compDic.get(f, 0)

    if sumOfCompounds > 0:
        former_percentage = sumOfFormers / sumOfCompounds * 100
        distance = abs(former_percentage - MINFRACTIONOFFORMERS * 100)
        return distance**2

    else:
        return 1000


def evaluateIndividual(individual):

    atomsDic = individual2atomF(individual, compoundList)

    value1 = model_results[prop1]['evalfun_dic'](atomsDic)[0][0]
    value2 = model_results[prop2]['evalfun_dic'](atomsDic)[0][0]

    scoreValue = score(value1, value2, desiredValue1, desiredValue2, weight1,
                       weight2)

    return scoreValue,


def createToolbox(MINCOMP, MAXCOMP, GENSIZE, CONSTRAINTPENALTY, TOURNMENTSIZE,
                  GENECROSSOVERPROB, GENEMUTPROB):

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    toolbox.register("attr_int", random.randint, MINCOMP, MAXCOMP)
    toolbox.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_int,
                     n=GENSIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluateIndividual)
    toolbox.decorate(
        "evaluate",
        tools.DeltaPenalty(sufficientNumberOfFormers, CONSTRAINTPENALTY,
                           distance_sufficientNumberOfFormers))
    toolbox.decorate(
        "evaluate",
        tools.DeltaPenalty(individualInsideDomain, CONSTRAINTPENALTY,
                           distance_individualInsideDomain))

    toolbox.register("mutate",
                     tools.mutUniformInt,
                     low=MINCOMP,
                     up=MAXCOMP,
                     indpb=GENEMUTPROB)
    toolbox.register("select", tools.selTournament, tournsize=TOURNMENTSIZE)
    toolbox.register("mate", tools.cxUniform, indpb=GENECROSSOVERPROB)

    return toolbox


def main(POPULATION, MINCOMP, MAXCOMP, GENSIZE, CONSTRAINTPENALTY,
         TOURNMENTSIZE, GENECROSSOVERPROB, GENEMUTPROB, GENERATIONS,
         CROSSOVERPROB, MUTPROB, model_results, prop1, prop2):

    toolbox = createToolbox(MINCOMP, MAXCOMP, GENSIZE, CONSTRAINTPENALTY,
                            TOURNMENTSIZE, GENECROSSOVERPROB, GENEMUTPROB)

    popresult = []
    pop = toolbox.population(n=POPULATION)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(GENERATIONS):

        best_ind = tools.selBest(pop, 1)[0]
        atomsDic = individual2atomF(best_ind, compoundList)

        value1 = model_results[prop1]['evalfun_dic'](atomsDic)[0][0]
        value2 = model_results[prop2]['evalfun_dic'](atomsDic)[0][0]

        print(
            'Starting generation {}. Current best is {:.3f}. value1 = {:.3f}, value2 = {:.3f}'
            .format(g, best_ind.fitness.values[0], value1, value2))

        if g % 100 == 0:
            pprint(atomsDic)
            compDic = dict(zip(compoundList, best_ind))
            print()
            pprint(compDic)
            print()

            for prop in possible_properties:
                pred = model_results[prop]['evalfun_dic'](atomsDic)[0][0]
                print(f'{prop} = {pred}')
            print()

        # Select the next generation individuals
        offspring = toolbox.select(pop, k=len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CROSSOVERPROB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPROB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        popresult.append(pop)

        # if g % 100 == 0:
        #     pickle.dump(popresult, open(resultpath, "wb"), protocol=-1)

    return pop
