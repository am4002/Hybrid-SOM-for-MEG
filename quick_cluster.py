# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 12:32:51 2016

@author: a
"""
import som_cluster_lib
from som_cluster_lib import *


"""
ind_pro = textIn('industrial_production',1)
indSOM = somWrap(ind_pro,h=100,w=100)
descView(indSOM,'Industrial Prodution')
"""
#### ordering of tasks is always:
#visual
#auditory
#visual distractor
#auditory distractor
#paired
#between interleaved

#subject 1
vis3773 = MEGReadin('r3773_Broad_T1_vis.hdf5')
aud3773 = MEGReadin('r3773_Broad_T4_aud.hdf5')
vdist3773 = MEGReadin('r3773_Broad_T3_vdist.hdf5')
adist3773 = MEGReadin('r3773_Broad_T6_adist.hdf5')
pair3773 = MEGReadin('r3773_Broad_T5_pair.hdf5')
bint3773 = MEGReadin('r3773_Broad_T2_bint.hdf5')

#subject 2
vis3918 = MEGReadin('r3918_Broad_T4_vis.hdf5')
aud3918 = MEGReadin('r3918_Broad_T1_aud.hdf5')
vdist3918 = MEGReadin('r3918_Broad_T2_vdist.hdf5')
adist3918 = MEGReadin('r3918_Broad_T3_adist.hdf5')
pair3918 = MEGReadin('r3918_Broad_T5_pair.hdf5')
bint3918 = MEGReadin('r3918_Broad_T6_bint.hdf5')

#subject3
vis4045 = MEGReadin('r4045_Broad_T4_vis.hdf5')
aud4045 = MEGReadin('r4045_Broad_T5_aud.hdf5')
vdist4045 = MEGReadin('r4045_Broad_T3_vdist.hdf5')
adist4045 = MEGReadin('r4045_Broad_T2_adist.hdf5')
pair4045 = MEGReadin('r4045_Broad_T6_pair.hdf5')
bint4045 = MEGReadin('r4045_Broad_T1_bint.hdf5')


#t1 = MEGReadin('r3918_Active_Broad_T1.hdf5')
#t2 = MEGReadin('r3918_Active_Broad_T2.hdf5')
#t3 = MEGReadin('r3918_Active_Broad_T3.hdf5')
#t4 = MEGReadin('r3918_Active_Broad_T4.hdf5')
#t5 = MEGReadin('r3918_Active_Broad_T5.hdf5')
#t6 = MEGReadin('r3918_Active_Broad_T6.hdf5')

dat3773 = np.vstack((vis3773,aud3773,vdist3773,adist3773,pair3773,bint3773))
dat3918 = np.vstack((vis3918,aud3918,vdist3918,adist3918,pair3918,bint3918))
dat4045 = np.vstack((vis4045,aud4045,vdist4045,adist4045,pair4045,bint4045))

som3773 = somWrap(dat3773, h = 50, w = 50, init = 'pca')

descView(som3773)

#cbook = som3918.codebook

one = [1]*150
two = [2]*150
three = [3]*150
four = [4]*150
five = [5]*150
six = [6]*150
predictor = np.concatenate((one,two,three,four,five,six))


#proper text/task labels
vis=['vis']*150
aud = ['aud']*150
vDist=['vDist']*150
aDist=['aDist']*150
pair=['pair']*150
bInt=['bInt']*150

#and it stacked together
tPredictor = np.concatenate((vis,aud,vDist,aDist,pair,bInt))

tomr3773 = tomeGen(som3773,dat3773)
#tomr3918 = tomeGen(som3773,dat3918)
#tomr4045 = tomeGen(som3773,dat4045)

flat3773 = tomr3773.reshape((900,2500))
#flat3918 = tomr3918.reshape((900,2500))
#flat4045 = tomr4045.reshape((900,2500))


svm3773 = svmWrap(flat3773,tPredictor,svmtype=1,ker='rbf',title='Subject 1 Confusion Matrix')