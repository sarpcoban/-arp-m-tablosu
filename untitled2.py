# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:38:19 2019

@author: Sarp
"""

import math

agirlik_gizli = [[0.36,0.44,0.86,0.22,0.11],[0.3,0.25,0.49,0.71,0.67]]
agirlik_gizli_degisim = [[0,0,0,0,0],[0,0,0,0,0]]
esik_gizli = [0.47,0.26,0.5,0.64,0.33]
esik_gizli_degisim = [0,0,0,0,0]
agirlik_cikis = [0.59,0.96,0.55,0.14,0.77]
agirlik_cikis_degisim = [0,0,0,0,0]
esik_cikis = 0.65
esik_cikis_degisim = 0
ogrenme = 0.1
momentum = 0.3

egitim_x = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],[1,9],[1,10],
            [3,1],[3,2],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[3,10],
            [5,1],[5,2],[5,3],[5,4],[5,5],[5,6],[5,7],[5,8],[5,9],[5,10],
            [7,1],[7,2],[7,3],[7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],
            [9,1],[9,2],[9,3],[9,4],[9,5],[9,6],[9,7],[9,8],[9,9],[9,10]]

test_x = [[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],[2,7],[2,8],[2,9],[2,10],
          [4,1],[4,2],[4,3],[4,4],[4,5],[4,6],[4,7],[4,8],[4,9],[4,10],
          [6,1],[6,2],[6,3],[6,4],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],
          [8,1],[8,2],[8,3],[8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[8,10],
          [10,1],[10,2],[10,3],[10,4],[10,5],[10,6],[10,7],[10,8],[10,9],[10,10]]

egitim_y = [1*1,1*2,1*3,1*4,1*5,1*6,1*7,1*8,1*9,1*10,
            3*1,3*2,3*3,3*4,3*5,3*6,3*7,3*8,3*9,3*10,
            5*1,5*2,5*3,5*4,5*5,5*6,5*7,5*8,5*9,5*10,
            7*1,7*2,7*3,7*4,7*5,7*6,7*7,7*8,7*9,7*10,
            9*1,9*2,9*3,9*4,9*5,9*6,9*7,9*8,9*9,9*10]

test_y = [2*1,2*2,2*3,2*4,2*5,2*6,2*7,2*8,2*9,2*10,
          4*1,4*2,4*3,4*4,4*5,4*6,4*7,4*8,4*9,4*10,
          6*1,6*2,6*3,6*4,6*5,6*6,6*7,6*8,6*9,6*10,
          8*1,8*2,8*3,8*4,8*5,8*6,8*7,8*8,8*9,8*10,
          10*1,10*2,10*3,10*4,10*5,10*6,10*7,10*8,10*9,10*10]

def normallestir_x(x):
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = x[i][j]/10
    return x

def normallestir_y(y):
    for i in range(len(y)):
        y[i] = y[i]/100
    return y

def sigmoid(net):
    return 1/(1+pow(math.e,-1*net))

def toplam(agirlik1,agirlik2,esik,girdi):
    return agirlik1*girdi[0]+agirlik2*girdi[1]+esik

def hata(beklenen,bulunan):
    return abs(beklenen-bulunan)


normal_egitim_x = normallestir_x(egitim_x)
normal_test_x = normallestir_x(test_x)
normal_egitim_y = normallestir_y(egitim_y)
normal_test_y = normallestir_y(test_y)
epoch = 0

while(1==1):
    ort_hata = 0
    for i in range(0,len(normal_egitim_x)):
        dugum_1 = toplam(agirlik_gizli[0][0],agirlik_gizli[1][0],esik_gizli[0],normal_egitim_x[i])
        dugum_1_net = sigmoid(dugum_1)
        dugum_2 = toplam(agirlik_gizli[0][1],agirlik_gizli[1][1],esik_gizli[1],normal_egitim_x[i])
        dugum_2_net = sigmoid(dugum_2)
        dugum_3 = toplam(agirlik_gizli[0][2],agirlik_gizli[1][2],esik_gizli[2],normal_egitim_x[i])
        dugum_3_net = sigmoid(dugum_3)
        dugum_4 = toplam(agirlik_gizli[0][3],agirlik_gizli[1][3],esik_gizli[3],normal_egitim_x[i])
        dugum_4_net = sigmoid(dugum_4)
        dugum_5 = toplam(agirlik_gizli[0][4],agirlik_gizli[1][4],esik_gizli[4],normal_egitim_x[i])
        dugum_5_net = sigmoid(dugum_5)
        dugum_cikis = dugum_1_net*agirlik_cikis[0]+dugum_2_net*agirlik_cikis[1]+dugum_3_net*agirlik_cikis[2]+dugum_4_net*agirlik_cikis[3]+dugum_5_net*agirlik_cikis[4]+esik_cikis
        dugum_cikis_net = sigmoid(dugum_cikis)
        hata = normal_egitim_y[i] - dugum_cikis_net
        ort_hata += pow(hata,2)/2
        hata_katsayi = dugum_cikis_net*(1 - dugum_cikis_net)*hata
        hata_katsayi_1 = dugum_1_net*(1-dugum_1_net)*(agirlik_cikis[0])*hata_katsayi
        hata_katsayi_2 = dugum_2_net*(1-dugum_2_net)*(agirlik_cikis[1])*hata_katsayi
        hata_katsayi_3 = dugum_3_net*(1-dugum_3_net)*(agirlik_cikis[2])*hata_katsayi
        hata_katsayi_4 = dugum_4_net*(1-dugum_4_net)*(agirlik_cikis[3])*hata_katsayi
        hata_katsayi_5 = dugum_5_net*(1-dugum_5_net)*(agirlik_cikis[4])*hata_katsayi
        agirlik_cikis_degisim[0] = ogrenme*hata_katsayi*dugum_1_net + momentum*agirlik_cikis_degisim[0]
        agirlik_cikis[0] = agirlik_cikis[0] + agirlik_cikis_degisim[0]
        agirlik_cikis_degisim[1] = ogrenme*hata_katsayi*dugum_2_net + momentum*agirlik_cikis_degisim[1]
        agirlik_cikis[1] = agirlik_cikis[1] + agirlik_cikis_degisim[1]
        agirlik_cikis_degisim[2] = ogrenme*hata_katsayi*dugum_3_net + momentum*agirlik_cikis_degisim[2]
        agirlik_cikis[2] = agirlik_cikis[2] + agirlik_cikis_degisim[2]
        agirlik_cikis_degisim[3] = ogrenme*hata_katsayi*dugum_4_net + momentum*agirlik_cikis_degisim[3]
        agirlik_cikis[3] = agirlik_cikis[3] + agirlik_cikis_degisim[3]
        agirlik_cikis_degisim[4] = ogrenme*hata_katsayi*dugum_5_net + momentum*agirlik_cikis_degisim[4]
        agirlik_cikis[4] = agirlik_cikis[4] + agirlik_cikis_degisim[4]
        esik_cikis_degisim = ogrenme*hata_katsayi + momentum*esik_cikis_degisim
        esik_cikis = esik_cikis + esik_cikis_degisim
        agirlik_gizli_degisim[0][0] = ogrenme*hata_katsayi_1*normal_egitim_x[i][0]+momentum*agirlik_gizli_degisim[0][0]
        agirlik_gizli[0][0] = agirlik_gizli[0][0] + agirlik_gizli_degisim[0][0]
        agirlik_gizli_degisim[0][1] = ogrenme*hata_katsayi_2*normal_egitim_x[i][0]+momentum*agirlik_gizli_degisim[0][1]
        agirlik_gizli[0][1] = agirlik_gizli[0][1] + agirlik_gizli_degisim[0][1]
        agirlik_gizli_degisim[0][2] = ogrenme*hata_katsayi_3*normal_egitim_x[i][0]+momentum*agirlik_gizli_degisim[0][2]
        agirlik_gizli[0][2] = agirlik_gizli[0][2] + agirlik_gizli_degisim[0][2]
        agirlik_gizli_degisim[0][3] = ogrenme*hata_katsayi_4*normal_egitim_x[i][0]+momentum*agirlik_gizli_degisim[0][3]
        agirlik_gizli[0][3] = agirlik_gizli[0][3] + agirlik_gizli_degisim[0][3]
        agirlik_gizli_degisim[0][4] = ogrenme*hata_katsayi_5*normal_egitim_x[i][0]+momentum*agirlik_gizli_degisim[0][4]
        agirlik_gizli[0][4] = agirlik_gizli[0][4] + agirlik_gizli_degisim[0][4]
        agirlik_gizli_degisim[1][0] = ogrenme*hata_katsayi_1*normal_egitim_x[i][1]+momentum*agirlik_gizli_degisim[1][0]
        agirlik_gizli[1][0] = agirlik_gizli[1][0] + agirlik_gizli_degisim[1][0]
        agirlik_gizli_degisim[1][1] = ogrenme*hata_katsayi_2*normal_egitim_x[i][1]+momentum*agirlik_gizli_degisim[1][1]
        agirlik_gizli[1][1] = agirlik_gizli[1][1] + agirlik_gizli_degisim[1][1]
        agirlik_gizli_degisim[1][2] = ogrenme*hata_katsayi_3*normal_egitim_x[i][1]+momentum*agirlik_gizli_degisim[1][2]
        agirlik_gizli[1][2] = agirlik_gizli[1][2] + agirlik_gizli_degisim[1][2]
        agirlik_gizli_degisim[1][3] = ogrenme*hata_katsayi_4*normal_egitim_x[i][1]+momentum*agirlik_gizli_degisim[1][3]
        agirlik_gizli[1][3] = agirlik_gizli[1][3] + agirlik_gizli_degisim[1][3]
        agirlik_gizli_degisim[1][4] = ogrenme*hata_katsayi_5*normal_egitim_x[i][1]+momentum*agirlik_gizli_degisim[1][4]
        agirlik_gizli[1][4] = agirlik_gizli[1][4] + agirlik_gizli_degisim[1][4]
        esik_gizli_degisim[0] = ogrenme*hata_katsayi_1 + momentum*esik_gizli_degisim[0]
        esik_gizli[0] = esik_gizli[0] + esik_gizli_degisim[0]
        esik_gizli_degisim[1] = ogrenme*hata_katsayi_2 + momentum*esik_gizli_degisim[1]
        esik_gizli[1] = esik_gizli[1] + esik_gizli_degisim[1]
        esik_gizli_degisim[2] = ogrenme*hata_katsayi_3 + momentum*esik_gizli_degisim[2]
        esik_gizli[2] = esik_gizli[2] + esik_gizli_degisim[2]
        esik_gizli_degisim[3] = ogrenme*hata_katsayi_4 + momentum*esik_gizli_degisim[3]
        esik_gizli[3] = esik_gizli[3] + esik_gizli_degisim[3]
        esik_gizli_degisim[4] = ogrenme*hata_katsayi_5 + momentum*esik_gizli_degisim[4]
        esik_gizli[4] = esik_gizli[4] + esik_gizli_degisim[4]
    epoch += 1
    if ort_hata <= 0.01:
        for j in range(0,len(normal_test_x)):
            dugum_1_test = toplam(agirlik_gizli[0][0],agirlik_gizli[1][0],esik_gizli[0],normal_test_x[j])
            dugum_1_test_net = sigmoid(dugum_1_test)
            dugum_2_test = toplam(agirlik_gizli[0][1],agirlik_gizli[1][1],esik_gizli[1],normal_test_x[j])
            dugum_2_test_net = sigmoid(dugum_2_test)
            dugum_3_test = toplam(agirlik_gizli[0][2],agirlik_gizli[1][2],esik_gizli[2],normal_test_x[j])
            dugum_3_test_net = sigmoid(dugum_3_test)
            dugum_4_test = toplam(agirlik_gizli[0][3],agirlik_gizli[1][3],esik_gizli[3],normal_test_x[j])
            dugum_4_test_net = sigmoid(dugum_4_test)
            dugum_5_test = toplam(agirlik_gizli[0][4],agirlik_gizli[1][4],esik_gizli[4],normal_test_x[j])
            dugum_5_test_net = sigmoid(dugum_5_test)
            dugum_cikis_test = dugum_1_test_net*agirlik_cikis[0]+dugum_2_test_net*agirlik_cikis[1] + dugum_3_test_net*agirlik_cikis[2] + dugum_4_test_net*agirlik_cikis[3]+ dugum_5_test_net*agirlik_cikis[4] + esik_cikis
            dugum_cikis_test_net = sigmoid(dugum_cikis_test)
            print(str(normal_test_x[j][0]*10),"x",str(normal_test_x[j][1]*10),"="+str(dugum_cikis_test_net*100))
        basari_orani = (sum(normal_test_y) - 50*ort_hata)/sum(normal_test_y)
        break
        