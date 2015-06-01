% Homework 05

% Clean up
clear all
close all
clc

% Datenaufbreitung
Data = load('fisher.txt');
Data0 = Data((Data(:,3)==0),:)
Data1 = Data((Data(:,3)==1),:);

% Dimensionen
Data_n   = size(Data,2);
Data_m   = size(Data,1);
Data0_n   = size(Data0,2);
Data0_m   = size(Data0,1);
Data1_n   = size(Data1,2);
Data1_m   = size(Data1,1);

x0 = min(Data0):max(Data0);
plot(x0, Data0(:,1:2));
% Aufgabe 1

% Aufgabe 2

