% clean up
clear all
close all
clc

% load data
Training  = load('pendigits-training.txt');
Testing   = load('pendigits-testing.txt');

% Labels
labelsTraining = Training(:,17);
labelsTesting  = Testing(:,17);

% 0.5786 klassGuete bei teilen durch 100 und 40 iter
% 0.4405 klassGuete bei teilen durch max und 40 iter
% center training data
for index1 = 1:length(Training)
   for index2 = 1:16
      cTraining(index1,index2) = Training(index1,index2) / max(Training(index1,1:16));
%        cTraining(index1,index2) = Training(index1,index2) / 100;
   end
end

% center test data
for index1 = 1:length(Testing)
   for index2 = 1:16
      cTesting(index1,index2) = Testing(index1,index2) / max(Testing(index1,1:16));
%          cTesting(index1,index2) = Testing(index1,index2) / 100;
   end
end


%%% Aufgabe 2 - Handgeschriebene Zahlen klasssifizieren %%%

LTD        = horzcat(cTraining,labelsTraining);  % labeled training data
ATD        = horzcat(cTraining,ones(7494,1));    % augmented data without labels

LTDtest    = horzcat(cTesting,labelsTesting);  % labeled test data
ATDtest    = horzcat(cTesting,ones(3498,1));   % augmented data without labels
predictedClassk2 = [];
predictedClassk4 = [];
predictedClassk8 = [];
predictedClassk10 = [];



% k = 8, Training
E8         = [];           % error history for plot
e8         = 1;            % just for e != 0

% set initial values
W1         = rand(17,8);   % random weights 17x2 from layer 0 to layer 1
W2         = rand(9,10);   % random weights 3x10 from layer 1 to layer 2
alpha      = 0.01;         % learning rate

for samanthasvorschlag = 1:40
    errorSum = 0;
    for runs = 1:length(ATD)
        
        L0     = ATD(runs,:);
        label  = labelsTraining(runs);
            
        % forward pass
        % layer 1
        t                   = L0 * W1;
        perceptron01_layer1 = 1 / (1 + exp(-t(:,1)));
        perceptron02_layer1 = 1 / (1 + exp(-t(:,2)));
        perceptron03_layer1 = 1 / (1 + exp(-t(:,3)));
        perceptron04_layer1 = 1 / (1 + exp(-t(:,4)));
        perceptron05_layer1 = 1 / (1 + exp(-t(:,5)));
        perceptron06_layer1 = 1 / (1 + exp(-t(:,6)));
        perceptron07_layer1 = 1 / (1 + exp(-t(:,7)));
        perceptron08_layer1 = 1 / (1 + exp(-t(:,8)));
        
        out_layer1         = [perceptron01_layer1,perceptron02_layer1,perceptron03_layer1,perceptron04_layer1,perceptron05_layer1,perceptron06_layer1,perceptron07_layer1,perceptron08_layer1];
        
        % layer 2
        t                  = [out_layer1, 1]*W2;
        perceptron01_layer2 = 1 / (1 + exp(-t(1)));
        perceptron02_layer2 = 1 / (1 + exp(-t(2)));
        perceptron03_layer2 = 1 / (1 + exp(-t(3)));
        perceptron04_layer2 = 1 / (1 + exp(-t(4)));
        perceptron05_layer2 = 1 / (1 + exp(-t(5)));
        perceptron06_layer2 = 1 / (1 + exp(-t(6)));
        perceptron07_layer2 = 1 / (1 + exp(-t(7)));
        perceptron08_layer2 = 1 / (1 + exp(-t(8)));
        perceptron09_layer2 = 1 / (1 + exp(-t(9)));
        perceptron10_layer2 = 1 / (1 + exp(-t(10)));
        
        out_layer2         = [perceptron01_layer2,perceptron02_layer2,perceptron03_layer2,perceptron04_layer2,perceptron05_layer2,perceptron06_layer2,perceptron07_layer2,perceptron08_layer2,perceptron09_layer2,perceptron10_layer2];
        
        % error calculation
        labelVector = zeros(1,10);
        for labelIndex = 1:10
            if label == labelIndex
                labelVector(:,labelIndex+1) = 1;
            end
        end

        e8                 = out_layer2 - labelVector;
        errorSum           = errorSum + sum(e8);
        E8                 = horzcat(E8,sum(e8));    
        
        % backward pass
        t1                 = L0 * W1;
        s11                = (1 / (1+exp(-t1(:,1))))*(1-(1 / (1+exp(-t1(:,1)))));
        s12                = (1 / (1+exp(-t1(:,2))))*(1-(1 / (1+exp(-t1(:,2)))));
        s13                = (1 / (1+exp(-t1(:,3))))*(1-(1 / (1+exp(-t1(:,3)))));
        s14                = (1 / (1+exp(-t1(:,4))))*(1-(1 / (1+exp(-t1(:,4)))));
        s15                = (1 / (1+exp(-t1(:,5))))*(1-(1 / (1+exp(-t1(:,5)))));
        s16                = (1 / (1+exp(-t1(:,6))))*(1-(1 / (1+exp(-t1(:,6)))));
        s17                = (1 / (1+exp(-t1(:,7))))*(1-(1 / (1+exp(-t1(:,7)))));
        s18                = (1 / (1+exp(-t1(:,8))))*(1-(1 / (1+exp(-t1(:,8)))));
           
        D1                 = [s11,0,0,0,0,0,0,0;
                              0,s12,0,0,0,0,0,0;
                              0,0,s13,0,0,0,0,0;
                              0,0,0,s14,0,0,0,0;
                              0,0,0,0,s15,0,0,0;
                              0,0,0,0,0,s16,0,0;
                              0,0,0,0,0,0,s17,0;
                              0,0,0,0,0,0,0,s18];
        
        t2                 = [out_layer1, 1] * W2;
        s201               = (1 / (1+exp(-t2(1))))*(1-(1 / (1+exp(-t2(1)))));
        s202               = (1 / (1+exp(-t2(2))))*(1-(1 / (1+exp(-t2(2)))));
        s203               = (1 / (1+exp(-t2(3))))*(1-(1 / (1+exp(-t2(3)))));
        s204               = (1 / (1+exp(-t2(4))))*(1-(1 / (1+exp(-t2(4)))));
        s205               = (1 / (1+exp(-t2(5))))*(1-(1 / (1+exp(-t2(5)))));
        s206               = (1 / (1+exp(-t2(6))))*(1-(1 / (1+exp(-t2(6)))));
        s207               = (1 / (1+exp(-t2(7))))*(1-(1 / (1+exp(-t2(7)))));
        s208               = (1 / (1+exp(-t2(8))))*(1-(1 / (1+exp(-t2(8)))));
        s209               = (1 / (1+exp(-t2(9))))*(1-(1 / (1+exp(-t2(9)))));
        s210               = (1 / (1+exp(-t2(10))))*(1-(1 / (1+exp(-t2(10)))));
        
        D2                 = [s201,0,0,0,0,0,0,0,0,0;
                              0,s202,0,0,0,0,0,0,0,0;
                              0,0,s203,0,0,0,0,0,0,0;
                              0,0,0,s204,0,0,0,0,0,0;
                              0,0,0,0,s205,0,0,0,0,0;
                              0,0,0,0,0,s206,0,0,0,0;
                              0,0,0,0,0,0,s207,0,0,0;
                              0,0,0,0,0,0,0,s208,0,0;
                              0,0,0,0,0,0,0,0,s209,0;
                              0,0,0,0,0,0,0,0,0,s210];
        
        W2_                = W2(1:8,:);
        dW1                = -alpha*D1*W2_*D2*e8'*L0;
        dW2                = -alpha*D2*e8'*[out_layer1, 1];
        W1                 = W1 + dW1';
        W2                 = W2 + dW2';

    end
    disp('durschn. Fehler fuer den ges. Datensatz: ');
    errorSum / length(ATD)
    samanthasvorschlag
end

needed_iterations = length(E8);

% plot
figure('NumberTitle','off','Name','Aufgabe 2, k=8');
plot(E8, '.')
title('Fehlerkurven');
xlabel('Iterationen');
ylabel('Fehlerwert');
axis([-0.1 needed_iterations -10 10]);
legend('Fehlerwerte');


% k = 8, Testing

correctly_predicted = 0;
for runs = 1:length(ATDtest)
    
    L0     = ATDtest(runs,:);
    label  = labelsTesting(runs);
        
    % forward pass
    % layer 1
    t                   = L0 * W1;
    perceptron01_layer1 = 1 / (1 + exp(-t(:,1)));
    perceptron02_layer1 = 1 / (1 + exp(-t(:,2)));
    perceptron03_layer1 = 1 / (1 + exp(-t(:,3)));
    perceptron04_layer1 = 1 / (1 + exp(-t(:,4)));
    perceptron05_layer1 = 1 / (1 + exp(-t(:,5)));
    perceptron06_layer1 = 1 / (1 + exp(-t(:,6)));
    perceptron07_layer1 = 1 / (1 + exp(-t(:,7)));
    perceptron08_layer1 = 1 / (1 + exp(-t(:,8)));
    
    out_layer1         = [perceptron01_layer1,perceptron02_layer1,perceptron03_layer1,perceptron04_layer1,perceptron05_layer1,perceptron06_layer1,perceptron07_layer1,perceptron08_layer1];
    
    % layer 2
    t                  = [out_layer1, 1]*W2;
    perceptron01_layer2 = 1 / (1 + exp(-t(1)));
    perceptron02_layer2 = 1 / (1 + exp(-t(2)));
    perceptron03_layer2 = 1 / (1 + exp(-t(3)));
    perceptron04_layer2 = 1 / (1 + exp(-t(4)));
    perceptron05_layer2 = 1 / (1 + exp(-t(5)));
    perceptron06_layer2 = 1 / (1 + exp(-t(6)));
    perceptron07_layer2 = 1 / (1 + exp(-t(7)));
    perceptron08_layer2 = 1 / (1 + exp(-t(8)));
    perceptron09_layer2 = 1 / (1 + exp(-t(9)));
    perceptron10_layer2 = 1 / (1 + exp(-t(10)));
    
    out_layer2         = [perceptron01_layer2,perceptron02_layer2,perceptron03_layer2,perceptron04_layer2,perceptron05_layer2,perceptron06_layer2,perceptron07_layer2,perceptron08_layer2,perceptron09_layer2,perceptron10_layer2];
    
    % prediction calculation
    prediction = 999;  % initial value
    predictionVal = max(out_layer2);
    for index = 1:length(out_layer2)
        if out_layer2(1, index) == predictionVal
            prediction = index - 1;
        end
    end
    predictedClassk8 = vertcat(predictedClassk8, prediction);
    if prediction == label
        correctly_predicted = correctly_predicted + 1;
    end

end

% confusionMatrix_k8 = confusionmat(labelsTesting, predictedClassk8) % wegen Lizenzerrors lieber nicht verwenden

confusionMatrix_k8 = zeros(10,10);
for i = 1:length(labelsTesting)
    confusionMatrix_k8(labelsTesting(i)+1, predictedClassk8(i)+1) = confusionMatrix_k8(labelsTesting(i)+1, predictedClassk8(i)+1) + 1;
end
confusionMatrix_k8

klass_guete = correctly_predicted / size(ATDtest, 1)

W1
W2