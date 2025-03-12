#!/bin/bash

#Download matrices
wget https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell10.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell2.tar.gz 
wget https://suitesparse-collection-website.herokuapp.com/MM/Bourchtein/atmosmodd.tar.gz 
wget https://suitesparse-collection-website.herokuapp.com/MM/Bourchtein/atmosmodl.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/vanHeukelum/cage13.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Bodendiek/CurlCurl_2.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Dziekonski/dielFilterV2real.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/Geo_1438.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/Hook_1498.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/ML_Laplace.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Schenk/nlpkkt80.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/Serena.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/PARSEC/Si87H76.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/StocF-1465.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Janna/Transport.tar.gz


#Unzip the files
tar -xvf af_shell10.tar.gz
mv af_shell10/af_shell10.mtx ./
rm -rf af_shell10 af_shell10.tar.gz

tar -xvf af_shell2.tar.gz
mv  af_shell2/af_shell2.mtx  ./
rm -rf af_shell2 af_shell2.tar.gz

tar -xvf atmosmodd.tar.gz
mv  atmosmodd/atmosmodd.mtx  ./
rm -rf atmosmodd atmosmodd.tar.gz

tar -xvf atmosmodl.tar.gz
mv  atmosmodl/atmosmodl.mtx  ./
rm -rf atmosmodl atmosmodl.tar.gz

tar -xvf cage13.tar.gz
mv  cage13/cage13.mtx  ./
rm -rf cage13 cage13.tar.gz

tar -xvf CurlCurl_2.tar.gz
mv  CurlCurl_2/CurlCurl_2.mtx  ./
rm -rf CurlCurl_2 CurlCurl_2.tar.gz

tar -xvf dielFilterV2real.tar.gz
mv  dielFilterV2real/dielFilterV2real.mtx  ./
rm -rf dielFilterV2real dielFilterV2real.tar.gz

tar -xvf Geo_1438.tar.gz
mv  Geo_1438/Geo_1438.mtx  ./
rm -rf Geo_1438 Geo_1438.tar.gz


tar -xvf Hook_1498.tar.gz
mv  Hook_1498/Hook_1498.mtx  ./
rm -rf Hook_1498 Hook_1498.tar.gz

tar -xvf ML_Laplace.tar.gz
mv  ML_Laplace/ML_Laplace.mtx  ./
rm -rf ML_Laplace ML_Laplace.tar.gz

tar -xvf  nlpkkt80.tar.gz
mv   nlpkkt80/nlpkkt80.mtx  ./
rm -rf  nlpkkt80  nlpkkt80.tar.gz

tar -xvf  Serena.tar.gz
mv   Serena/Serena.mtx  ./
rm -rf  Serena  Serena.tar.gz

tar -xvf Si87H76.tar.gz
mv  Si87H76/Si87H76.mtx  ./
rm -rf Si87H76 Si87H76.tar.gz

tar -xvf StocF-1465.tar.gz
mv  StocF-1465/StocF-1465.mtx  ./
rm -rf StocF-1465 StocF-1465.tar.gz

tar -xvf Transport.tar.gz
mv  Transport/Transport.mtx  ./
rm -rf Transport Transport.tar.gz

