import re
import os
import pickle
import pandas as pd
import numpy as np
import random
from pprint import pprint
from deap import base, creator, tools
import copy
from mendeleev import element


### Test

def master_atom_lst(compound_lst):
    atom_lst = []
    for key in compounds_lst:
        atoms = composition2atoms(key)
        atom_lst.append(atoms)
    return atom_lst


def individual2atomF(individual, compoundList):

    atom_dic = {}
    for baseValue, dic in zip(individual, atom_lst):
        for at in dic:
            atom_dic[at] = atom_dic.get(at, 0) + dic[at] * baseValue

    multiplyby = 1 / sum(atom_dic.values())
    atomsDic = {k:v*multiplyby for k,v in atom_dic.items()}
    return atomsDic


### Chemistry

def molarMass(atoms):
    '''Compute the molar mass of a atom dictionary. Returns in kg/mol'''

    molmass = 0
    for a in atoms:
        el = element(a)
        molmass += el.atomic_weight*atoms[a]
    return molmass/1000 # kg/mol


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


def individual2atomF_OLD(individual, compoundList):
    compoundDic = dict(zip(compoundList, individual))
    atomsDic = compound2atoms(compoundDic)
    multiplyby = 1 / sum(atomsDic.values())
    atomsDic = {k:v*multiplyby for k,v in atomsDic.items() if v > 0}
    return atomsDic


def compound2weighF(compound):
    dic = {}
    for composition in compound.keys():
        atoms = composition2atoms(composition)
        molmass = molarMass(atoms)
        dic[composition] = compound[composition] * molmass
    multiplyby = 1 / np.sum(list(dic.values()))
    weightP = {k:v*multiplyby for k,v in dic.items() if v > 0}
    return weightP



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

    model_dic = {
        'model': model,
        'evalfun_x': evalfun_x,
        'X': X,
        'y': y,
        'X_features': X_features,
        'y_features': y_features,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'best_space': best_space,
    }

    return model_dic


def predictIndividual(nonzeroAtomsDic, evalfun_x, X_features):
    X = []
    if set(nonzeroAtomsDic.keys()).issubset(set(X_features)):
        for feat in X_features:
            X.append(nonzeroAtomsDic.get(feat,0))

        return evalfun_x([X])[0][0]
    else:
        return np.nan

 

### Comosition

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


def getCompounds(atomsLst, referenceAtom, compounds, forbiddenAtoms=[],
                 forceAtomAdd=[]):

    # global forbidden_compounds

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



### Score

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

    if len(force_compound_range) > 0:
        multiplication_factor = 1 / sum(individual)
        compDic = {c:v*multiplication_factor for c,v in zip(compoundList, individual)}
        for c in force_compound_range:
            min_fraction = force_compound_range[c][0]
            max_fraction = force_compound_range[c][1]
            if not min_fraction <= compDic[c] <= max_fraction:
                return False

    return True


def distance_individualInsideDomain(individual):
    distance = 0

    atomsF = individual2atomF(individual, compoundList)
    for a in atomsF:
        constrain = minmaxdictionary.get(a, (0, 0))
        min_fraction = constrain[0] * ((100 - relax) / 100)
        max_fraction = constrain[1] * ((100 + relax) / 100)
        if not min_fraction <= atomsF[a] <= max_fraction:
            if atomsF[a] < min_fraction:
                distance += abs(atomsF[a] - min_fraction)
            else:
                distance += abs(atomsF[a] - max_fraction)

    if len(force_compound_range) > 0:
        multiplication_factor = 1 / sum(individual)
        compDic = {c:v*multiplication_factor for c,v in zip(compoundList, individual)}
        for c in force_compound_range:
            min_fraction = force_compound_range[c][0]
            max_fraction = force_compound_range[c][1]
            if not min_fraction <= compDic[c] <= max_fraction:
                if compDic[c] < min_fraction:
                    distance += abs(compDic[c] - min_fraction)
                else:
                    distance += abs(compDic[c] - max_fraction)

    if distance < 1:
        return (1 / distance)**2 + (1/distance)*100
    else:
        return distance**2 + distance*100


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

        return distance*10000 + distance**2
        # if distance < 1:
        #     return (1 / distance)**2 + (1/distance)*100
        # else:
        #     return distance**2 + distance*100

    else:
        return 1000000000


def evaluateIndividual(individual):

    atomsDic = individual2atomF(individual, compoundList)

    value1 = predictIndividual(
        atomsDic,
        model_results[prop1]['evalfun_x'],
        model_results[prop1]['X_features'],
    )
    value2 = predictIndividual(
        atomsDic,
        model_results[prop2]['evalfun_x'],
        model_results[prop2]['X_features'],
    )

    scoreValue = score(value1, value2, desiredValue1, desiredValue2, weight1,
                       weight2)

    return scoreValue,



### Final

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
         CROSSOVERPROB, MUTPROB, model_results, prop1, prop2, costdic1kg={}):

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

        value1 = predictIndividual(
            atomsDic,
            model_results[prop1]['evalfun_x'],
            model_results[prop1]['X_features'],
        )
        value2 = predictIndividual(
            atomsDic,
            model_results[prop2]['evalfun_x'],
            model_results[prop2]['X_features'],
        )

        print(
            'Starting generation {}. Current best is {:.3f}. value1 = {:.3f}, value2 = {:.3f}'
            .format(g, best_ind.fitness.values[0], value1, value2))

        if g % print_results_every_n_generations == 0:
            compDic = {c:v for c,v in zip(compoundList, best_ind) if v > 0}
            weightDic = compound2weighF(compDic)

            multi_by = 100 / sum(best_ind)
            compDic_norm = {c:round(v*multi_by,3) for c,v in zip(compoundList, best_ind) if v > 0}

            price1kg = sum([weightDic[c]*costdic1kg.get(c,np.nan) for c in
                            weightDic.keys()])

            weightDic_grams = {c:round(v*grams_of_glass_to_make,3) for c,v in weightDic.items()}

            # pprint(atomsDic)
            print()
            print('Compounds in mol (using only integers)')
            pprint(compDic)
            print()
            print('Compounds in %mol (total = 100%)')
            pprint(compDic_norm)
            print()
            print(f'Compounds in weight (in grams, to make {grams_of_glass_to_make}g of glass) [BETA: please check!]')
            pprint(weightDic_grams)
            print()
            print(f'Price of 1 kg of glass: {price1kg:.2f}$')
            print()
            for prop in possible_properties:
                pred = predictIndividual(
                    atomsDic,
                    model_results[prop]['evalfun_x'],
                    model_results[prop]['X_features'],
                )
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




