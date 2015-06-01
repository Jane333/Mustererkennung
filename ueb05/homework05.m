% Clean up
clear all
close all
clc

%%%%%%%%%%%%%%%%%  Datenaufbereitung  %%%%%%%%%%%%%%%%

A = load('fisher.txt');
Koordinaten = A(:,1:2)
Klassen = A(:,3)

%%%%%%%%%%%%%%%%%%%%%  Aufgabe 2  %%%%%%%%%%%%%%%%%%%%

% Create a default (linear) discriminant analysis classifier:
linclass = fitcdiscr(Koordinaten, Klassen);

