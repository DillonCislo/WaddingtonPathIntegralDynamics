%% Test Speed/Accuracy For Single Polygon Comparison ========================
clear; close all; clc;

% Generate a random polygon
numSidesP = randi([3 50]);
[P(:,1), P(:,2)] = simple_polygon(numSidesP);
[P(:,1), P(:,2)] = poly2ccw(P(:,1), P(:,2));

a = [-100 100];
P = (a(2)-a(1)) .* P + a(1);

% Generate a random polygon
numSidesQ = randi([3 50]);
[Q(:,1), Q(:,2)] = simple_polygon(numSidesQ);
[Q(:,1), Q(:,2)] = poly2ccw(Q(:,1), Q(:,2));

b = [-100 100];
Q = (b(2)-b(1)) .* Q + b(1);

% Calculate area of intersection (MATLAB)
tic
pshape = polyshape(P(:,1), P(:,2));
qshape = polyshape(Q(:,1), Q(:,2));
rshape = intersect(pshape,qshape);
if isempty(rshape)
    intArea1 = 0;
else
    intArea1 = rshape.area;
end
toc

fprintf('Intersection Area (MATLAB) = %0.5f\n', intArea1);

% Calculate area of intersection (C++)
tic
intArea2 = polygon_intersection_area(P, Q(:,1), Q(:,2));
toc

fprintf('Intersection Area (C++) = %0.5f\n', intArea2);

if (intArea1 == 0)
    fprintf('Absolute Error = %0.5e\n', ...
        abs(intArea1-intArea2));
else
    fprintf('Relative Error = %0.5e\n', ...
        abs(intArea1-intArea2) ./ abs(intArea1));
end

% View Polygons -------------------------------------------------------------

hold on
plot(pshape)
plot(qshape)
plot(rshape)
hold off

axis equal

%% Test Speed/Accuracy For Multiple Polygon Comparison ======================
clear; close all; clc;

% Generate a random polygon
numSidesP = randi([3 50]);
[P(:,1), P(:,2)] = simple_polygon(numSidesP);
[P(:,1), P(:,2)] = poly2ccw(P(:,1), P(:,2));

a = [-100 100];
P = (a(2)-a(1)) .* P + a(1);

% Generate a random polygon
numSidesQ = randi([3 50]);
numPolyQ = 500;

maxIter = 20;
b = [-100 100];
QX = zeros(numSidesQ+1, numPolyQ);
QY = zeros(numSidesQ+1, numPolyQ);
for i = 1:numPolyQ

    Q = [];
    iter = 0;
    while isempty(Q)

        try
            [Q(:,1), Q(:,2)] = simple_polygon(numSidesQ);
            assert(isequal(size(Q), [numSidesQ+1, 2]), ...
                'Invalid polygon size');
        catch
            Q = [];
            iter = iter + 1;
        end

        assert(iter <= maxIter, ['Failed to construct polygon ' ...
            'after %d iterations'], maxIter);

    end

    [Q(:,1), Q(:,2)] = poly2ccw(Q(:,1), Q(:,2));
    Q = (b(2)-b(1)) .* Q + b(1);
    

    QX(:,i) = Q(:,1);
    QY(:,i) = Q(:,2);

end

clear Q

% Calculate area of intersection (MATLAB)
tic
pshape = polyshape(P(:,1), P(:,2));
intAreas1 = zeros(numPolyQ, 1);
for i = 1:numPolyQ
    qshape = polyshape(QX(:,i), QY(:,i));
    intAreas1(i) = area(intersect(pshape,qshape));
end
toc

% Calculate area of intersection (C++)
tic
intAreas2 = polygon_intersection_area(P, QX, QY);
toc

fprintf('Max Relative Error = %0.5e\n', ...
    max(abs(intAreas1-intAreas2)./abs(intAreas1)));