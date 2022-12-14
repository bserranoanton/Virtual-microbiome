
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:31:04 2021
@author: Belen Serrano Antón
"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import random as random
from scipy.optimize import curve_fit
#from scipy.stats import linregress, gamma, lognorm, kstest
import scipy.stats as stats

import skbio.diversity as skbio_div


class ecologyGeneration_Family:
    
    
    #number of families, genus, species and communities
    _nFamilies = 5
    _nGenus = 20
    _nSpecies = 300
    _nCommunities = 50
    
    # mean and standard deviation for lognormal MAD
    _muMAD = -18.5#3.1
    _sigmaMAD = 4.1 #1#1.
    
    #shape and scale for gamma AFD
    _shapeAFD = 1.3#2. # mean=4
    _scaleAFD = 1. 
    
    
    #Threshold for the correlation coefficient when checking Taylor's Law
    _threshold = 0.95
    
         
    def visualizeAbundances(abundances, taxa):
        """Plots a heatmap corresponding to the abundances matrix.
        
        :arg abundances: matrix of abundances. Rows: communities, Cols: taxa
        :arg taxa: could be genus, species, family..."""
        
        
        f = plt.figure()
        ax = sns.heatmap(abundances, cmap="rocket_r", linewidth=0.5)
        plt.title('Abundances')
        
        plt.xlabel(taxa)
            
        plt.ylabel('Communities')
        plt.show()
        
    
    def generateMAD(mu, sigma, nTaxa): 
        """ Generates Mean Abundance Distribution based on a lognormal distribution.
        
        :arg mu: mean of the lognormal distribution.
        :arg sigma: standard deviation  of the lognormal distribution.
        :arg nTaxa: number of species, genus, families...
        
        :return: mean abundance vector."""
        
        
        MADvector = np.random.lognormal(mu, sigma, nTaxa)
        MADvectorSum = sum(MADvector)
        MADvector = MADvector/MADvectorSum
        return MADvector
    
    def generateAFD(shape, scale, nCommunities):
        """ Generates Abundance Fluctuations Distribution based on a gamma distribution.
        
        :arg shape: shape of the gamma distribution.
        :arg scale: scale of the gamma distribution.
        :arg nCommunities: number of communities.
        
        :return: vector of abundance fluctuations."""
        
        
        AFDvector = np.random.gamma(shape, scale, nCommunities)
        AFDvectorSum = sum(AFDvector)
        AFDvector = AFDvector/AFDvectorSum
        
        return AFDvector
    
    def generateAbundanceDistribution(mu, sigma, shape, scale, nCommunities, nTaxa):
        """Generates taxa abundances per community (rows: communities, columns: taxa).
        
        :arg mu: mean of the lognormal distribution (MAD).
        :arg sigma: standard deviation  of the lognormal distribution (MAD).
        :arg shape: shape of the gamma distribution (AFD).
        :arg scale: scale of the gamma distribution (AFD).
        :arg nCommunities: number of communities.
        :arg nTaxa: number of species, genus, families,....
        
        :return: MAD vector and abundances matrix.
        """
        
        
        MADvector = ecologyGeneration_Family.generateMAD(mu, sigma, nTaxa)
        abundances = np.zeros((nCommunities, nTaxa))
        
        for species_i in range(nTaxa):
            AFDvector = ecologyGeneration_Family.generateAFD(shape, scale, nCommunities)
            abundances[:,species_i] = MADvector[species_i] * AFDvector * nCommunities
        
        return [MADvector, abundances]
    
    
    def cuadratic(x, a):
        """Objetive function for fitting Taylor's Law.
        
        :arg x: variable.
        :arg a: coefficient.
        
        :return: a*x**2.
        """
        
        
        return a*x**2
    
    
    def plotTaylorsLaw(coeff, MADvector, varianceVector, taxa):
        """Plots mean vs variance and y = coef*x^2 in logaritmic scale.
        
        :arg coef: adjusted coefficient a for equation variance = a*mean^2.
        :arg MADvector: mean abundance vector.
        :arg varianceVector: variance abundance vector.
        :arg taxa: could be family, genus, species,... (usually species)
        """
        
        
        f = plt.figure()
        xMax = np.amax(MADvector)
        xMin = np.amin(MADvector)
        t = np.arange(xMin, xMax, 0.01)    
        
        #plots with axis in logaritmic scale
        plt.loglog(t, coeff*t**2, label="y =" + str(coeff)+"*x^2")
        plt.loglog(MADvector, varianceVector, 'ro')
        
        plt.title('Taylor\'s Law at' + taxa + 'level')    
        plt.xlabel('Mean abundances per community')
        plt.ylabel('Variance of abundances per community')
        plt.legend()
        plt.show()
        
    
    def testTaylorsLaw(MADvector, abundances):
        """ Tests correlation between cuadratic mean and variance abundances.
        
        :arg MADvector: mean abundances vector.
        :arg abundances: matrix of abundances (rows: communities, cols: taxa (genus,
        species, ...)).
        
        :return: correlation coefficient, variance vector of abundances and adjusted 
        coefficient (a). variance = a*mean**2.
        """ 
        
        nSpecies = len(MADvector)
        
        varianceVector = np.zeros(nSpecies)
        
        for species_i in range(nSpecies):
            varianceVector[species_i] = np.var(abundances[:,species_i])
            
        #curve fit: varianceVector = a*MADvector^2
        a, _ = curve_fit(ecologyGeneration_Family.cuadratic, MADvector, varianceVector)
        
        #check correlation coeffcient between varianceVector and MADvector
        slope, intercept, r_value, p_value, std_err = stats.linregress(varianceVector,
                                                                       MADvector)
    
        return [r_value, varianceVector, a]
    
    
    def computeMAD(abundances):
        """
        Computes the Mean Abundance Vector.
        
        :arg abundances: matrix of abundances (rows: communities, cols: taxa (species,
        genus,...))
        
        :return: mean abundance vector.
        """
        
        rows = abundances.shape[0]
        cols = abundances.shape[1]
        
        MAD = np.zeros(cols)
        
        for col_i in range(cols):
            MAD[col_i] = sum(abundances[:, col_i])/rows
            
        return MAD
        
    
    def testDistribution(distribution, data):
        """
        Performs the (one sample or two samples) Kolmogorov-Smirnov test for goodness of fit.
        
        :arg distribution: distribution to test. Could be lognormal or gamma.
        :arg data: vector of data
        
        :return: statistic, pvalue or (0, -1) if not valid distribution.
        """
        
        stat = 0
        pvalue = -1
            
        if (distribution == "lognormal"):
            stat, pvalue = stats.kstest(data, 'lognorm', stats.lognorm.fit(data, floc=0))
            
        elif (distribution == "gamma"):
            stat, pvalue = stats.kstest(data, 'gamma', stats.gamma.fit(data, floc=0))
            
        else: 
            print("not supported distribution")
            
        
        return stat, pvalue
    
    
    def testDistributionsAbundances(abundances):
        """
        Test if rows follow a lognormal distribution and if columns follow
        a gamma distribution, by means of Kolmogorov-Smirnov test.
        
        :arg abundances: matrix of abundances (rows: communities, cols: taxa (species,
        genus, ...))
        
        :return: 1 if the abundances matrix follows the distributions above, 0 if not.
        """
        
        size = abundances.shape
        nCom = size[0]
        nSpe = size[1]
        
        valid = 1
        
        for com_i in range(nCom):
            _, pvalue = ecologyGeneration_Family.testDistribution("lognormal", abundances[com_i, :])
            if (pvalue < 0.05):
                valid = 0
                #print("Community ", com_i, " is not lognormal distributed ", pvalue)
                break
            
        spe_i = 0
        while (spe_i < nSpe and valid == 1):
            _, pvalue = ecologyGeneration_Family.testDistribution("gamma", abundances[:, spe_i])
            if (pvalue < 0.05):
                valid = 0
                #print("Species ", spe_i, " is not gamma distributed", pvalue)
                break
            spe_i += 1
        
     
        return valid
    
    def generateScenario(minNCommunities, maxNCommunities, minNFamilies, maxNFamilies,
                         minNGenus, maxNGenus, minNSpecies, maxNSpecies, minMu, maxMu, 
                         minSigma, maxSigma, minShape, maxShape):
        """
        Generates random values between the given ranges having in count ecology
        restrictions: #Families < #Genus < #Species.
        
        :arg minNCommunities: minimum number of communities.
        :arg maxNCommunities: maximum number of communities.
        :arg minNFamilies: minimum number of families.
        :arg maxNFamilies: maximum number of families.
        :arg minNGenus: minimum number of genus.
        :arg maxNGenus: maximum number of genus.
        :arg minNSpecies: minimum number of species.
        :arg maxNSpecies: maximum number of species.
        :arg minMu: minimum value of mu parameter.
        :arg maxMu: maximum value of mu parameter.
        :arg minSigma: minimum value of sigma parameter.
        :arg maxSigma: maximum value of sigma parameter.
        :arg minShape: minimum value of shape parameter.
        :arg maxShape: maximum value of shape parameter.
            
        :return: [#Communities, #Families, #Genus, #Species, mu, sigma, shape]
        
        """
        
        nCommunities = random.randint(minNCommunities, maxNCommunities)
        nSpecies = random.randint(minNSpecies, maxNSpecies)
        nGenus = random.randint(minNGenus, min(nSpecies, maxNGenus))
        nFamilies = random.randint(minNFamilies, min(nGenus, maxNFamilies))
        
        # nCommunities = 100
        # nSpecies = 1000
        # nGenus = 20
        # nFamilies = 5
        
        mu = random.uniform(minMu, maxMu)
        sigma = random.uniform(minSigma, maxSigma)
        shape = random.uniform(minShape, maxShape)
        
        return [nCommunities, nFamilies, nGenus, nSpecies, mu, sigma, shape]
    
    
    def generateGenusAbundances(mu, sigma, nGenus, nSpecies):
        """
        Groups species in genus following a lognormal distribution (the same function
        can be used to group genus in families and so on).
        
        :arg mu: mu parameter for lognormal distribution
        :arg shape: shape parameter for lognormal distribution
        :arg nGenus: number of genus
        :arg nSpecies: number of species
        
        :return: vector with the number of species per genus (v[i] = 
        number of species of genus_i) and number of genus with
        at least one species.
        """
        
    
        nEffectiveGenus = 0
        
        while(nEffectiveGenus < nGenus):
        
            #Generate lognormal distribution for the number of species in each genus
            nSpeciesPerGenus = ecologyGeneration_Family.generateMAD(mu, sigma, nGenus*4)*nSpecies
            #nSpeciesPerGenus = [int(nSpecies/nGenus) for i in range(nGenus)]
            
            #Asign the corresponding number of species
            nSpeciesPerGenus = sorted(nSpeciesPerGenus.astype(int), reverse=True)
            totalAsignedSpecies = sum(nSpeciesPerGenus)
            
            #print("Total", totalAsignedSpecies)
            #print("nSpecies", nSpecies)
            
            if(totalAsignedSpecies < nSpecies):
                for genus_i in range(nGenus*4):
                    if(nSpeciesPerGenus[genus_i] == 0):
                        nSpeciesPerGenus[genus_i] = nSpecies - totalAsignedSpecies
                        break
            
            nEffectiveGenus = sum(x > 0 for x in nSpeciesPerGenus)
        
        if(nEffectiveGenus > nGenus):
            nSpeciesPerGenus[nGenus-1] += sum(nSpeciesPerGenus[nGenus:nEffectiveGenus])
            nEffectiveGenus = nGenus
        
        totalAsignedSpecies = sum(nSpeciesPerGenus[0:nGenus])

        
        
        return [nSpeciesPerGenus, nEffectiveGenus]
    
    
    def getAbundancesGenus(nSpeciesPerGenus, nEffectiveGenus, abundances):
        """
        Computes the matrix of abundances of genus given the matrix of abundances of
        species (the same function can be used to compute the matrix of abundances of
        families given the genus and so on). 
        
        :arg nSpeciesPerGenus: vector with the number of species per genus (v[i] = 
            number of species of genus_i).
        :arg nEffectiveGenus: number of genus with at least 1 species.
        :arg abundances: matrix of abundances (rows: communities, cols: species (taxa)).
        
        :return: matrix of abundances per genus (sum of the abundances of its species)
        and list of correspondences (list[i] = species of genus_i).
        """
        
        size = abundances.shape
        nCommunities = size[0]
        nSpecies = size[1]
        abundancesGenus = np.zeros((nCommunities, nEffectiveGenus))
        
        listSpecies = list(range(nSpecies))
        indexesListSpeciesPerGenus = list(range(nEffectiveGenus))
        
        #correspondence[i] = list of species of genus_i
        correspondence = list()
        
        for genus_i in range(nEffectiveGenus):
            #select randomly the number of species
            index_nSpecies = np.random.choice(indexesListSpeciesPerGenus,
                                                            1, replace=False)
            nSpecies_genus_i = nSpeciesPerGenus[index_nSpecies[0]]
            
            #select randomly these species
            species_genus_i = np.random.choice(listSpecies, nSpecies_genus_i, replace=False)
            subMatrixAbundances = abundances[:, species_genus_i]
            correspondence.append(sorted(species_genus_i))
            
            #remove these species
            for ele in sorted(species_genus_i, reverse = True):  
                listSpecies.remove(ele)
            
            #sum abundances per community
            abundance_genus_i = np.sum(subMatrixAbundances, axis=1)
            abundancesGenus[:, genus_i] = abundance_genus_i
            
            #remove this option
            indexesListSpeciesPerGenus.remove(index_nSpecies[0]) 
            
            
        return [abundancesGenus, correspondence]
        
        
    def getAlphaDivShannon(abundances):
        """
        Computes alpha diversity by means of Shannon's index.
        
        :arg abundances: matrix of abundances (rows: communities, cols: species).
        
        :return: vector of alpha diversities per community.
        """
        
        nCommunities = abundances.shape[0]
        
        alphaDiv = np.zeros(nCommunities)
        
        for com_i in range(nCommunities):
            s = sum(abundances[com_i,:])
            p_i = abundances[com_i, :]/s
            alphaDiv[com_i] = -sum(p_i * np.log2(p_i))
        
        return alphaDiv
        
    
    def getBetaDivBrayCurtis(abundances):
        """
        Computes beta diversity by means of Bray Curtis' index.
        
        :arg abundances: matrix of abundances (rows: communities, cols: species).
        
        :return: matrix of beta diversities. Element ij compares community i with 
        community j.
        """
        
        nCommunities = abundances.shape[0]
        nSpecies = abundances.shape[1]
        
        betaDiv = np.zeros((nCommunities, nCommunities))
        
        for com_i in range(nCommunities):
            sum_com_i = sum(abundances[com_i,:])
            
            for com_j in range(com_i+1, nCommunities):
                sum_com_j = sum(abundances[com_j,:])
                C_ij = 0 #sum of the lesser values
                
                for species_i in range(nSpecies):
                    abundance_i = abundances[com_i, species_i]
                    abundance_j = abundances[com_j, species_i]
                    
                    if(abundance_i > 0 and abundance_j > 0):
                        C_ij += min(abundance_i, abundance_j)
                    
                betaDiv[com_i, com_j] = 1 - ((2*C_ij)/(sum_com_i + sum_com_j))
                betaDiv[com_j, com_i] = betaDiv[com_i, com_j]
                
        return betaDiv
    
    def runSimulation(nCommunities, nSpecies, muMAD, sigmaMAD, shapeAFD, scaleAFD):
        
        """
        Generates matrix of abundances verifying that correlation coefficient between mean
        and variance (Taylor's Law) >= _threshold and that rows and columns follows
        a lognormal and gamma distribution, respectively. 
        
        :arg nCommunities: number of communities. 
        :arg nSpecies: number of species.
        :arg muMAD: mu parameter for MAD lognormal distribution.
        :arg sigmaMAD: sigma parameter for MAD lognormal distribution.
        :arg shapeAFD: shape parameter for AFD gamma distribution.
        :arg scaleAFD: scale parameter for AFD gamma distribution.
        
        :return: the MAD vector, the matrix of abundances and the correlation 
        coefficient between mean and variance abundances (-1 if the simulation is not
        valid)
        
        """
        
        print("Running ecologyGenerator with parameters:", 
              "\nnSpecies: ", nSpecies,
              "\nnCommunities: ", nCommunities,
              "\nParameters for MAD (lognormal):",
              "\nmu: ", muMAD, "\nsigma:", sigmaMAD,
              "\nParameters for AFD (gamma):",
              "\nshape: ", shapeAFD, 
              "\nscale", scaleAFD)
           
        maxIter = 50
        i = 0
        while (i < maxIter):
            #print("\nIteration: ", i)
            [MADvector, abundances] = ecologyGeneration_Family.generateAbundanceDistribution(muMAD, sigmaMAD,
            shapeAFD, ecologyGeneration_Family._scaleAFD, nCommunities, nSpecies)
            
            #Check Taylor's Law
            [r_value, varianceVector, coeff] = ecologyGeneration_Family.testTaylorsLaw(MADvector, abundances)
             
            
            if(r_value >= ecologyGeneration_Family._threshold):
                #print("Taylor\'s Law correlation coefficient at species level: ", r_value)
                #Check lognormal distribution for communities abundance
                #Check gamma distribution for species abundances
                valid = ecologyGeneration_Family.testDistributionsAbundances(abundances)
                
                if(valid == 1):
                    #Visualize results
                    #visualizeAbundances(abundances, 'species')
                    #plotTaylorsLaw(coeff, MADvector, varianceVector, 'species')
                    break   
                
            i += 1
            
        if(i < maxIter):
            print("Valid simulation")
            
        else: #not a valid simulation
            print("Not valid simulation")
            r_value = -1
            
        return [MADvector, abundances, r_value]
            
            
    def plotHeatmapsTaylorsLaw(nCommunities, nGenus, nGenusMin, nSpecies, nSpeciesMin,
                               granularityGenus, granularitySpecies, parameters):
        """
        Runs a simulation going through the number of genus, species and
        the macroecological parameters. The number of communities is fixed. Plots 
        heatmaps showing the correlation coefficient between mean and variance abundances. 
        
        :arg nCommunities: number of communities.
        :arg nGenus: maximum number of genus.
        :arg nGenusMin: minimum number of genus.
        :arg nSpecies: maximum number of species.
        :arg nSpecesMin: minimum number of species.
        :arg granularityGenus: increase in the number of genus per iteration.
        :arg granularitySpecies: increase in the number of species per iteration.
        :arg parameters: nx3 (n>1) array with macroecological parameters (mu, sigma, shape).
        
        """
        
        #Heatmaps of species correlation coefficient for Taylor's Law
        
        #index
        numSimGenus = 0
        numSimSpecies = 0 
    
        
        sizeGenus = int((nGenus-nGenusMin)/granularityGenus)
        sizeSpecies = int((nSpecies-nSpeciesMin)/granularitySpecies)
        
        print(sizeGenus, sizeSpecies)
        
        rCoeffsMatrix = np.zeros((sizeGenus, sizeSpecies))
        
        print(rCoeffsMatrix.shape)
        maxScenarios = len(parameters) 
        
        for sce_i in range(maxScenarios):
            numSimGenus = 0
            for genus_i in range(nGenusMin,nGenus,granularityGenus): #30
                numSimSpecies = 0
                for species_i in range(nSpeciesMin,nSpecies,granularitySpecies): #500
                    rCoeffsMatrix[numSimGenus, numSimSpecies],_ = (ecologyGeneration_Family.runSimulation(nCommunities, genus_i, species_i, parameters[sce_i, 0], 
                                          parameters[sce_i, 1],  parameters[sce_i, 2]))
                    numSimSpecies += 1
                numSimGenus += 1
            
            f = plt.figure()
            ax = plt.axes()
            sns.heatmap(rCoeffsMatrix*10, annot=True, cmap="rocket_r", ax=ax, linewidths=.5, fmt=".1f", vmin=6, vmax=10,
                        xticklabels = range(nSpeciesMin,nSpecies,granularitySpecies), 
                        yticklabels = range(nGenusMin,nGenus,granularityGenus) )
            plt.title("Scenario " +str(sce_i) + " Parameters [mu, sigma, shape]: " + str(parameters[0]))
            plt.xlabel("Species")
            plt.ylabel("Genus")
            
            
    def saveEcologyDiversity(header, index = 0, nCommunities=0, nFamilies=0, nGenus=0,
                             nSpecies=0, parameters=0, 
                             fileName = "saveEcologyDiversityOutput.txt",
                             meanAlphaDiv=0, meanBetaDiv=0):
        
        """
        Saves a file with the results of the simulations. 
        
        :arg nCommunities: number of communities.
        :arg nGenus: number of genus.
        :arg nSpecies: number of species.
        :arg parameters: array with macroecological parameters (mu, sigma, shape).
        :arg fileName: name of the file where to save results.
        """
        
        #maxScenarios = len(parameters) 
        
        #write header
        if(header):
            file = open(fileName, "w")
            file.write("index" + "\t" + "nCommunities" + "\t" + "nFamilies" + "\t" + 
                       "nGenus" + "\t" + "nSpecies" + "\t" + 
                   "mu" + "\t" + "sigma" + "\t" + "shape" + "\t" + "meanAlphaDiv" + 
                   "\t" + "meanBetaDiv\n")
        else:
            file = open(fileName, "a")
            file.write(str(index) + "\t" + str(nCommunities) + "\t" + str(nFamilies) +
            "\t" + str(nGenus) + "\t" + str(nSpecies) + "\t" + 
            str(round(parameters[0],3))+ "\t" + str(round(parameters[1],3)) + "\t" + 
                          str(round(parameters[2], 3)) + "\t"
                          + str(round(meanAlphaDiv,3)) + "\t" + str(round(meanBetaDiv,3)) + "\n")        
        
        file.close()
        
    def saveCorrespondences(header, correspondence=list(), fileName="outputCorrespondence.txt",
                            index=0, mu=0, sigma=0, taxa="genus"):
        
        """
        Saves a file with the correspondence between a level of taxonomy and its 
        superior level (species and genus, e.g).
        
        :arg header: boolean that indicates if we want to open the file in "w" mode
        and write the header line. (False to open the file in "a" mode and write 
        data). 
        :arg correspondence: list of correspondences (eg, species corresponding to 
        each genus-> list[i] = species of genus_i)
        :arg fileName: name of the file
        :arg index: index of the simulation 
        :arg taxa: 
        """
        
        txtHeader = "species per genus"
        txtBody1 = "Genus "
        txtBody2 = "Species "
        
        if(taxa == "family"):
            txtHeader = "genus per family"
            txtBody1 = "Family "
            txtBody2 = "Genus "
        
        if(header):
            f = open(fileName, "w")
            f.write("Simulation  index" + "\t" + txtHeader + "\n")
            
        else:
            f = open(fileName, "a")
            f.write("Simulation " + str(index) + " mu= " + str(mu) + "sigma= " + str(sigma)
            + "\n")
            size = len(correspondence)
            for genus_i in range(size):
                f.write("\t" + txtBody1 + str(genus_i) + "\t" + txtBody2 + 
                         str(len(correspondence[genus_i])) + "\t" + 
                         str(correspondence[genus_i]) + "\n")
        
        f.close()
    
    def saveDiversity(alphaDivMeanSpecies, betaDivMeanSpecies,
                                alphaDivMeanGenus, betaDivMeanGenus,
                                alphaDivMeanFamilies, betaDivMeanFamilies,
                                 fileName):
        f = open(fileName, "w")
        f.write("Mean alpha diversity species " + str(alphaDivMeanSpecies) + 
                "\nMean alpha diversity genus " + str(alphaDivMeanGenus) + 
                "\nMean alpha diversity families " + str(alphaDivMeanFamilies) + 
                "\nMean beta diversity species " + str(betaDivMeanSpecies) + 
                "\nMean beta diversity genus " + str(betaDivMeanGenus) + 
                "\nMean beta diversity families " + str(betaDivMeanFamilies) +"\n")
        f.close()
        
        
        
        
    def saveScenarioInfo(nCommunities, nFamilies, nGenus, nSpecies, sce_i,
                     abundances, alphaDivMeanSpecies, betaDivMeanSpecies,
                     correspondence, abundancesGenus, alphaDivMeanGenus,
                     betaDivMeanGenus,  muGenus, sigmaGenus, 
                     correspondenceFamilies, abundancesFamilies,
                     alphaDivMeanFamilies, betaDivMeanFamilies,
                     muFamilies, sigmaFamilies, fileName):
        
        
        #save species info
        ecologyGeneration_Family.saveEcologyDiversity(True, fileName = fileName + "Ecology.txt")
        ecologyGeneration_Family.saveEcologyDiversity(False, index = 0, nCommunities=nCommunities, 
                             nFamilies = nFamilies, nGenus=nGenus,
                             nSpecies=nSpecies, parameters=sce_i, 
                             fileName = fileName + "Ecology.txt",
                             meanAlphaDiv=alphaDivMeanSpecies,
                             meanBetaDiv=betaDivMeanSpecies)
        
        #save genus correspondences
        ecologyGeneration_Family.saveCorrespondences(True, fileName=fileName + "Genus.txt")
        ecologyGeneration_Family.saveCorrespondences(False, correspondence=correspondence, mu= muGenus,
                            sigma= sigmaGenus,
                            fileName=fileName + "Genus.txt")
        
        #save families correspondences
        ecologyGeneration_Family.saveCorrespondences(True, fileName=fileName + "Families.txt", taxa="family")
        ecologyGeneration_Family.saveCorrespondences(False, correspondence=correspondenceFamilies, mu= muFamilies,
                            sigma= sigmaFamilies,
                            fileName=fileName + "Families.txt", taxa="family")
        
        #save diversities 
        ecologyGeneration_Family.saveDiversity(alphaDivMeanSpecies, betaDivMeanSpecies,
                                alphaDivMeanGenus, betaDivMeanGenus,
                                alphaDivMeanFamilies, betaDivMeanFamilies,
                                 fileName + "Diversities.txt")
        #save abundances
        np.savetxt(fileName+"AbundancesSpecies.txt", abundances, delimiter=',')
        np.savetxt(fileName+"AbundancesGenus.txt", abundancesGenus, delimiter=',')
        np.savetxt(fileName+"AbundancesFamilies.txt", abundancesFamilies, delimiter=',')
        
        
    def getMeanDiversities(abundances):
        """
        Computes the mean alpha and beta diversities.
        
        :arg abundances: matrix of abundances (rows: communities, cols: species (taxa))
        
        :return: mean of alpha diversity, mean of beta diversity
        """
        
        nCommunities = abundances.shape[0]
        
        adivShannonSpecies = skbio_div.alpha_diversity('shannon', abundances)
        meanAlphaDiv = np.mean(adivShannonSpecies)
        
        bdivBrayCurtisSpecies = skbio_div.beta_diversity("braycurtis", abundances)
        upperTriangular = bdivBrayCurtisSpecies[np.triu_indices(nCommunities, k = 1)]
        meanBetaDiv = np.mean(upperTriangular)
        
        return [meanAlphaDiv, meanBetaDiv]
    
    
        
    def getScenariosVariation(parameters, percentage):
        
        """
        Makes a variation of +-percentage on the parameteres. 
        
        :arg parameters: nx3 (n>1) array with macroecological parameters (mu, sigma, shaape).
        :arg percentage: percentage of the variation.
        
        :return: 2nx3 array with the original parameters and the result of the variation.
            row i: original parameters i, row i+1: result of the variation of parameters i.
        """
        
        shape = parameters.shape
        maxScenarios = shape[0]
        sizeScenario = shape[1]
        
        newParameters = np.zeros((maxScenarios*2, sizeScenario))
        
        newIndex = 0
        
        for sce_i in range(maxScenarios):
            newParameters[newIndex] = parameters[sce_i]
            for value in range(sizeScenario):
                newParameters[newIndex+1, value] = (parameters[sce_i, value] + 
                            parameters[sce_i, value] * percentage * np.random.choice([-1, 1]))
            newIndex += 2
        
        return newParameters
    
    
    def runDiverseSimulations():
        
        maxSimulations = 50
        fileName = "simulation11.txt"
        fileNameGenus = "simulation11Genus.txt"
        fileNameFamilies = "simulation11Families.txt"
        
        fig, axs = plt.subplots(2, 2)
        
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111, projection='3d')
        
        fig3 = plt.figure(3)
        ax3 = fig3.add_subplot(111, projection='3d')
        
        ecologyGeneration_Family.saveEcologyDiversity(header=True, fileName=fileName)
        ecologyGeneration_Family.saveCorrespondences(header=True, fileName=fileNameGenus)
        ecologyGeneration_Family.saveCorrespondences(header=True, fileName=fileNameFamilies, taxa="family")
        
        maxXaxis1 = 0
        maxXaxis2 = 0
        
        
        for sim_i in range(maxSimulations):
            #generate scenario
            scenario_i = ecologyGeneration_Family.generateScenario(3,30,3,10,10,30,100,500,-30,-10,1,4,0.3,1.5)
            #[nCommunities, nFamilies, nGenus, nSpecies, mu, sigma, shape]
            MADvector, abundances, r_value = ecologyGeneration_Family.runSimulation(scenario_i[0], scenario_i[3],
                                                           scenario_i[4], scenario_i[5],
                                                           scenario_i[6], ecologyGeneration_Family._scaleAFD)
            if(r_value == -1):
                sim_i -= 1
                continue
            else:
                alphaDivMean, betaDivMean = ecologyGeneration_Family.getMeanDiversities(abundances)
                ecologyGeneration_Family.saveEcologyDiversity(False, sim_i, scenario_i[0], scenario_i[1], 
                                     scenario_i[2], scenario_i[3],
                                     np.asarray(scenario_i[4:7]), fileName, 
                                     alphaDivMean, betaDivMean)
                
                #Genus level:
                nSpeciesPerGenus, nEffectiveGenus = ecologyGeneration_Family.generateGenusAbundances(scenario_i[4],
                                                        scenario_i[5], scenario_i[2], scenario_i[3])
                abundancesGenus, correspondence = ecologyGeneration_Family.getAbundancesGenus(nSpeciesPerGenus, nEffectiveGenus,
                                                 abundances)
                alphaDivMeanGenus, betaDivMeanGenus = ecologyGeneration_Family.getMeanDiversities(abundancesGenus)
                
                ecologyGeneration_Family.saveCorrespondences(False, correspondence=correspondence, fileName=fileNameGenus,
                            index=sim_i)
                
                #Family level:
                #                                    #(mu, sigma, nFamilies, nGenus):
                nGenusPerFamilies, nEffectiveFamilies = ecologyGeneration_Family.generateGenusAbundances(scenario_i[4],
                                                        scenario_i[5], scenario_i[1] ,nEffectiveGenus) 
                abundancesFamilies, correspondenceFamilies = ecologyGeneration_Family.getAbundancesGenus(nGenusPerFamilies,
                                                        nEffectiveFamilies, abundancesGenus)
                
                alphaDivMeanFamilies, betaDivMeanFamilies = ecologyGeneration_Family.getMeanDiversities(abundancesFamilies)
                
                ecologyGeneration_Family.saveCorrespondences(False, correspondence=correspondenceFamilies,
                                    fileName=fileNameFamilies, index=sim_i, taxa="family")
                
                #color for simulation i
                color_i = (random.random(),random.random(),random.random())
                
                #plot alpha diversity
                axs[0, 0].scatter(alphaDivMean, alphaDivMeanGenus, marker='o', color=color_i)
                axs[0, 0].annotate(str(sim_i), (alphaDivMean, alphaDivMeanGenus))
                
                #plot beta diversity
                axs[0, 1].scatter(betaDivMean, betaDivMeanGenus, marker='o', color=color_i)
                axs[0, 1].annotate(str(sim_i), (betaDivMean, betaDivMeanGenus))
                
                #Famly level
                #plot alpha diversity
                axs[1, 0].scatter(alphaDivMeanGenus, alphaDivMeanFamilies, marker='o', color=color_i)
                axs[1, 0].annotate(str(sim_i), (alphaDivMeanGenus, alphaDivMeanFamilies))
                
                #plot beta diversity
                axs[1, 1].scatter(betaDivMeanGenus, betaDivMeanFamilies, marker='o', color=color_i)
                axs[1, 1].annotate(str(sim_i), (betaDivMeanGenus, betaDivMeanFamilies))
                
                #3D plot alpha diversity
                plt.figure(2)
                ax2.scatter(alphaDivMean, alphaDivMeanGenus, alphaDivMeanFamilies,
                           c=color_i)
                ax2.text(alphaDivMean, alphaDivMeanGenus, alphaDivMeanFamilies,
                        str(sim_i))     
                
                #3D plot beta diversity
                plt.figure(3)
                ax3.scatter(betaDivMean, betaDivMeanGenus, betaDivMeanFamilies,
                           c=color_i)
                ax3.text(betaDivMean, betaDivMeanGenus, betaDivMeanFamilies,
                        str(sim_i)) 
                
                #adjust axis
                if(alphaDivMean > maxXaxis1):
                    maxXaxis1 = alphaDivMean
                    
                if(betaDivMean > maxXaxis2):
                    maxXaxis2 = betaDivMean
                
        maxXaxis1 += 0.2
        maxXaxis2 += 0.05
    
        
        #Alpha diversity (species, genus)
        axs[0, 0].set_title('Mean of alpha diversities (species, genus)')
        axs[0, 0].set(xlabel='Mean alpha diversity at species level', 
           ylabel='Mean alpha diversity at genus level', xlim=[0, maxXaxis1],
           ylim=[0, maxXaxis1])
    
        #Beta diversity (species, genus)
        axs[0, 1].set_title('Mean of beta diversities (species, genus)')
        axs[0, 1].set(xlabel='Mean beta diversity at species level', 
           ylabel='Mean beta diversity at genus level', xlim=[0, maxXaxis2],
           ylim=[0, maxXaxis2])
    
        #Alpha diversity (genus, family)
        axs[1, 0].set_title('Mean of alpha diversities (genus, family)')
        axs[1, 0].set(xlabel='Mean alpha diversity at genus level', 
           ylabel='Mean alpha diversity at family level', xlim=[0, maxXaxis1],
           ylim=[0, maxXaxis1])
    
        #Beta diversity (genus, family)
        axs[1, 1].set_title('Mean of beta diversities (genus, family)')
        axs[1, 1].set(xlabel='Mean beta diversity at genus level', 
           ylabel='Mean beta diversity at family level', xlim=[0, maxXaxis2],
           ylim=[0, maxXaxis2])
        
        #3D plot alpha diversity
        plt.figure(2)
        ax2.set_title("Mean alpha diversity")
        ax2.set(xlabel="Species level", ylabel="Genus level", zlabel="Families level",
                xlim=[0, maxXaxis1],
                ylim=[0, maxXaxis1])
      
        #3D plot beta diversity
        plt.figure(3)
        ax3.set_title("Mean beta diversity")
        ax3.set(xlabel="Species level", ylabel="Genus level", zlabel="Families level",
                xlim=[0, maxXaxis2],
                ylim=[0, maxXaxis2])
      
        plt.show()
        
    
    
    
if __name__ == "__main__":
    
    #run simulation with selected params 
    
    # Use params in a while with:
    # parameters = np.array([[-18.5, 4.1, 1.3], #values for mu, sigma and shape
    #                        [-16.1, 3.5, 0.4],
    #                        [-17.2, 3.8, 0.3],
    #                        [-19.8, 4.4, 0.4],
    #                        [-17.3, 4.1, 0.4],
    #                        [-14.2, 2.7, 0.3],
    #                        [-16.2, 4.6, 0.2],
    #                        [-17.5, 3.7, 0.6],
    #                        [-17.1, 3.8, 0.38],
    #                        [-22.1, 7.0, 3.2],
    #                        [-28, 8.5, 1.2],
    #                        [-14.7, 4.1, 0.7],
    #                        [-14.8, 4, 0.3],
    #                        [-13.7, 3.8, 0.9],
    #                        [-18.4, 5.2, 0.3],
    #                        [-22.6, 7.3, 1.5],
    #                        [-23.9, 7.4, 1.4]
    #                        ])

    # or define params manually (we do that in this example)
    mu = -14 #-14 low div
    sigma = 8 #8 low div, 1.9 high
    shape = 0.8 #0.8
    sce_i = [mu, sigma, shape]
    
    # New scenarios
    alphaDivIndicator = 0   # 1 if alphaDivMeanSpecies > 5,
                            # 2 if alphaDivMeanSpecies < 3
                            # 0 in other case

    #Scenarios with genus and families
    while(alphaDivIndicator != 2):
        # get abundances and diversities at species level
        r_value = -1
        while(r_value == -1):
            MADvector, abundances, r_value = ecologyGeneration_Family.runSimulation(
                    ecologyGeneration_Family._nCommunities, ecologyGeneration_Family._nSpecies, 
                    sce_i[0], sce_i[1], sce_i[2], ecologyGeneration_Family._scaleAFD)
    
        alphaDivMeanSpecies, betaDivMeanSpecies = ecologyGeneration_Family.getMeanDiversities(abundances)
        
        if(alphaDivMeanSpecies > 5 and betaDivMeanSpecies > 0.5):
            alphaDivIndicator = 1
      
        elif(alphaDivMeanSpecies < 3 and betaDivMeanSpecies > 0.5):
            alphaDivIndicator = 2
        
        print(alphaDivIndicator)
        print(alphaDivMeanSpecies)
        print(betaDivMeanSpecies)
#        for sce_n in range(1,2):
        fileName = "db3_bajaDiv"
        #set the number of genus
        nGenus = 16
#            if(sce_n == 1 or sce_n == 2 or sce_n == 5 or sce_n == 6):
#                nGenus = 10
#            else:
#                nGenus = 30
            
        #set the number of families
        nFamilies = 4
#            if(sce_n % 2 == 1):
#                nFamilies = 2
#            else:
#                nFamilies = 5
        
        #set the file name of the scenario
        fileName += str("_16Genus_")
#            if(alphaDivIndicator == 1):
#                fileName += str(sce_n)
#            elif(alphaDivIndicator == 2):
#                fileName += str(sce_n+4) 
        
        #build abundance matrices for genus and families
        alphaDivMeanSpecies, betaDivMeanSpecies = ecologyGeneration_Family.getMeanDiversities(abundances)

        #get abundances and diversities at genus level
        #mu1 = random.uniform(-30, -14)
        #sigma1 = random.uniform(2,3)
        muGenus = -17
        sigmaGenus = 0.9
        
        nSpeciesPerGenus, nEffectiveGenus = ecologyGeneration_Family.generateGenusAbundances(muGenus,sigmaGenus,
                                                                    nGenus, ecologyGeneration_Family._nSpecies)
        abundancesGenus, correspondence = ecologyGeneration_Family.getAbundancesGenus(nSpeciesPerGenus, nEffectiveGenus,
                                             abundances)
        alphaDivMeanGenus, betaDivMeanGenus = ecologyGeneration_Family.getMeanDiversities(abundancesGenus)
        
        #get abundances and diversities at families level
        muFamilies = -17
        sigmaFamilies = 0.9
        
        nGenusPerFamily, nEffectiveFamilies = ecologyGeneration_Family.generateGenusAbundances(
                muFamilies,sigmaFamilies,
                nFamilies, nGenus)
        abundancesFamilies, correspondenceFamilies = ecologyGeneration_Family.getAbundancesGenus(
                nGenusPerFamily, nEffectiveFamilies, abundancesGenus)
        alphaDivMeanFamilies, betaDivMeanFamilies = ecologyGeneration_Family.getMeanDiversities(abundancesFamilies)
        
        print("-------RESULTS--------")
        print(alphaDivMeanSpecies)
        print(alphaDivMeanGenus)
        print(betaDivMeanSpecies)
        print(betaDivMeanGenus)
        
        if (alphaDivIndicator == 2):
            ecologyGeneration_Family.saveScenarioInfo(ecologyGeneration_Family._nCommunities, 
                            nFamilies, nGenus, ecologyGeneration_Family._nSpecies, sce_i,
                             abundances, alphaDivMeanSpecies, betaDivMeanSpecies,
                             correspondence, abundancesGenus, alphaDivMeanGenus,
                             betaDivMeanGenus, muGenus, sigmaGenus, 
                             correspondenceFamilies, abundancesFamilies,
                             alphaDivMeanFamilies, betaDivMeanFamilies,
                             muFamilies, sigmaFamilies, fileName)
        

