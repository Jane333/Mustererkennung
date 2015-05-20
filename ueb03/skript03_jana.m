% A 3.1: k-means auf die Daten clusters.txt anwenden,
%        k-means soll selbst implementiert werden!

k = 3;
numIterations = 5;

mean1 = C(1,:); % mean1, selected randomly
mean2 = C(2,:); % mean2, selected randomly
mean3 = C(3,:); % mean3, selected randomly
mean1_elems = []; % elements belonging to mean1
mean2_elems = []; % elements belonging to mean2
mean3_elems = []; % elements belonging to mean3
plotArray = [];

for iter=1:numIterations
    mean1_elems = [];
    mean2_elems = [];
    mean3_elems = [];
    for elem=1:size(C,1) % iterate over all elements
        dist = sqrt(abs(C(elem,1) - mean1(:,1))^2  + abs(C(elem,2) - mean1(:,2))^2);
        closest = mean1;
        dist2 = sqrt(abs(C(elem,1) - mean2(:,1))^2  + abs(C(elem,2) - mean2(:,2))^2);
        if dist > dist2
            closest = mean2;
            dist = dist2;
        end
        dist3 = sqrt(abs(C(elem,1) - mean3(:,1))^2  + abs(C(elem,2) - mean3(:,2))^2);
        if dist > dist3
            closest = mean3;
            dist = dist3;
        end
        if closest == mean1
            mean1_elems = vertcat(mean1_elems, C(elem, :));
        elseif closest == mean2
            mean2_elems = vertcat(mean2_elems, C(elem, :));
        else
            mean3_elems = vertcat(mean3_elems, C(elem, :));
        end
    end
    mean1_elems;
    mean2_elems;
    mean3_elems;
    
    % Visualisierung der Clusterzentren
    plotOfIteration = 1; % which iteration do we want to see a plot for?
    if iter == plotOfIteration
        % x = min(mean1_elems):max(mean1_elems)
        mean1_elems_x = mean1_elems(:,1); % x coordinates of all elements belonging to mean1
        mean1_elems_y = mean1_elems(:,2); % y coordinates of all elements belonging to mean1
        mean2_elems_x = mean2_elems(:,1);
        mean2_elems_y = mean2_elems(:,2);
        mean3_elems_x = mean3_elems(:,1);
        mean3_elems_y = mean3_elems(:,2);
        scatter(mean1_elems_x, mean1_elems_y, 40, [1 0 0])
        hold on
        scatter(mean1(:,1), mean1(:,2), 60, [.3 0 0], 'filled')
        hold on
        scatter(mean2_elems_x, mean2_elems_y, 40, [0 1 0])
        hold on
        scatter(mean2(:,1), mean2(:,2), 60, [0 .3 0], 'filled')
        hold on
        scatter(mean3_elems_x, mean3_elems_y, 40, [0 0 1])
        hold on
        scatter(mean3(:,1), mean3(:,2), 60, [0 0 .3], 'filled')
    end
    
    mean1 = [mean(mean1_elems(:,1)), mean(mean1_elems(:,2))];
    mean2 = [mean(mean2_elems(:,1)), mean(mean2_elems(:,2))];
    mean3 = [mean(mean3_elems(:,1)), mean(mean3_elems(:,2))];
end
