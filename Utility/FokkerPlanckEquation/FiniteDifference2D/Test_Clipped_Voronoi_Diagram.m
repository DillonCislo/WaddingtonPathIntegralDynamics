clear; close all; clc;

numPoints = 50;
xLim = [-5 5];
yLim = [-5 5];

% Generate boundary polygon
bdyPoly = [ diff(xLim) .* rand(numPoints, 1) + xLim(1), ...
    diff(yLim) .* rand(numPoints, 1) + yLim(1) ];
bdyPoly = bdyPoly(convhull(bdyPoly), :);

% Generate points inside the bounding polygon
X = nan(numPoints, 2);
count = 1;
while count < (numPoints+1)

    curX = [ diff(xLim) .* rand(1) + xLim(1), ...
    diff(yLim) .* rand(1) + yLim(1) ];
    if inpolygon(curX(1), curX(2), bdyPoly(:,1), bdyPoly(:,2))
        X(count, :) = curX;
        count = count+1;
    end

end

DT = delaunayTriangulation(X);
[v, c] = clippedVoronoiDiagram(DT, bdyPoly);

f = max(cellfun(@numel, c, 'Uni', true));
f = cell2mat(cellfun(@(x) [x, nan(1, f-numel(x)+1)], c, 'Uni', false));


plot(polyshape(bdyPoly), 'LineWidth', 2, 'EdgeColor', 'none');
hold on
patch('Faces', f, 'Vertices', v, 'EdgeColor', 'b', 'FaceColor', 'none');
scatter(X(:,1), X(:,2), 'filled', 'r');
hold off
axis equal
xlim(xLim);
ylim(yLim);

clear count curX