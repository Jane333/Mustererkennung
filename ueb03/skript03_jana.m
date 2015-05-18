%  Aufgabe 3, k-means

C = load('clusters.txt');
k = 3;
numIterations = 5

mean1 = C(1,:)
mean2 = C(2,:)
mean3 = C(3,:)
mean1_elems = []
mean2_elems = []
mean3_elems = []

for iter=1:numIterations
    mean1_elems = []
    mean2_elems = []
    mean3_elems = []
    for elem=1:size(C,1) % iterate over all elements
        dist = sqrt(abs(C(elem,1) - mean1(:,1))^2  + abs(C(elem,2) - mean1(:,2))^2);
        closest = mean1;
        dist2 = sqrt(abs(C(elem,1) - mean2(:,1))^2  + abs(C(elem,2) - mean2(:,2))^2);
        if dist > dist2
            closest = mean2;
            dist = dist2
        end
        dist3 = sqrt(abs(C(elem,1) - mean3(:,1))^2  + abs(C(elem,2) - mean3(:,2))^2);
        if dist > dist3
            closest = mean3;
            dist = dist3
        end
        if closest == mean1
            mean1_elems = vertcat(mean1_elems, C(elem, :))
        elseif closest == mean2
            mean2_elems = vertcat(mean2_elems, C(elem, :))
        else
            mean3_elems = vertcat(mean3_elems, C(elem, :))
        end
    end
    mean1_elems
    mean2_elems
    mean3_elems
    mean1 = [mean(mean1_elems(:,1)), mean(mean1_elems(:,2))]
    mean2 = [mean(mean2_elems(:,1)), mean(mean2_elems(:,2))]
    mean3 = [mean(mean3_elems(:,1)), mean(mean3_elems(:,2))]
end