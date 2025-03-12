#!/bin/bash

#Download matrices
wget https://suitesparse-collection-website.herokuapp.com/MM/Um/2cubes_sphere.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Sandia/ASIC_320ks.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Watson/Baumann.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Rothberg/cfd2.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/QLi/crashbasis.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Boeing/ct20stif.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/IBM_EDA/dc1.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/UTEP/Dubcova3.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/McRae/ecology2.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/FEM_3D_thermal2.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/AMD/G2_circuit.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Goodwin/Goodwin_095.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Schenk_IBMSDS/matrix-new_3.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Um/offshore.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Schenk_ISEI/para-10.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Wissgott/parabolic_fem.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/VLSI/ss1.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Norris/stomach.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Botonakis/thermomech_TK.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/CEMW/tmt_unsym.tar.gz
wget https://suitesparse-collection-website.herokuapp.com/MM/Ronis/xenon2.tar.gz

#Unzip the files
tar -xvf 2cubes_sphere.tar.gz
mv 2cubes_sphere/2cubes_sphere.mtx ./
rm -rf 2cubes_sphere.tar.gz 2cubes_sphere

tar -xvf ASIC_320ks.tar.gz
mv ASIC_320ks/ASIC_320ks.mtx ./
rm -rf ASIC_320ks.tar.gz ASIC_320ks

tar -xvf Baumann.tar.gz
mv Baumann/Baumann.mtx ./
rm -rf  Baumann Baumann.tar.gz

tar -xvf cfd2.tar.gz
mv cfd2/cfd2.mtx ./
rm  -rf cfd2 cfd2.tar.gz

tar -xvf crashbasis.tar.gz
mv crashbasis/crashbasis.mtx ./
rm -rf crashbasis crashbasis.tar.gz

tar -xvf ct20stif.tar.gz
mv ct20stif/ct20stif.mtx ./
rm -rf ct20stif ct20stif.tar.gz

tar -xvf dc1.tar.gz
mv dc1/dc1.mtx ./
rm -rf dc1 dc1.tar.gz

tar -xvf Dubcova3.tar.gz
mv Dubcova3/Dubcova3.mtx ./
rm  -rf Dubcova3 Dubcova3.tar.gz

tar -xvf ecology2.tar.gz
mv ecology2/ecology2.mtx ./
rm -rf ecology2 ecology2.tar.gz

tar -xvf FEM_3D_thermal2.tar.gz
mv FEM_3D_thermal2/FEM_3D_thermal2.mtx ./
rm -rf FEM_3D_thermal2 FEM_3D_thermal2.tar.gz

tar -xvf G2_circuit.tar.gz
mv G2_circuit/G2_circuit.mtx ./
rm -rf G2_circuit G2_circuit.tar.gz

tar -xvf Goodwin_095.tar.gz
mv Goodwin_095/Goodwin_095.mtx ./
rm -rf Goodwin_095 Goodwin_095.tar.gz

tar -xvf matrix-new_3.tar.gz
mv matrix-new_3/matrix-new_3.mtx ./
rm -rf matrix-new_3 matrix-new_3.tar.gz


tar -xvf offshore.tar.gz
mv offshore/offshore.mtx ./
rm -rf offshore offshore.tar.gz

tar -xvf para-10.tar.gz
mv para-10/para-10.mtx ./
rm -rf para-10 para-10.tar.gz

tar -xvf parabolic_fem.tar.gz
mv parabolic_fem/parabolic_fem.mtx ./
rm -rf parabolic_fem parabolic_fem.tar.gz

tar -xvf ss1.tar.gz
mv ss1/ss1.mtx ./
rm -rf ss1 ss1.tar.gz

tar -xvf stomach.tar.gz
mv stomach/stomach.mtx ./
rm -rf stomach stomach.tar.gz

tar -xvf thermomech_TK.tar.gz
mv thermomech_TK/thermomech_TK.mtx ./
rm -rf thermomech_TK thermomech_TK.tar.gz

tar -xvf tmt_unsym.tar.gz
mv tmt_unsym/tmt_unsym.mtx ./
rm -rf tmt_unsym tmt_unsym.tar.gz

tar -xvf xenon2.tar.gz
mv xenon2/xenon2.mtx ./
rm -rf xenon2 xenon2.tar.gz
