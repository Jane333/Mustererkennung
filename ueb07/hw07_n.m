% Clean up
clear all
close all
clc

% Datenaufbereitung
Data  = load('fieldgoal.txt');
Goal0 = Data((Data(:,2)==0),:)
Goal1 = Data((Data(:,2)==1),:)
x     = linspace(0,1);

%%% Aufgabe 1 - Logistische Regression %%%