function [v, c] = clippedVoronoiDiagram(DT, bdyPoly)
%CLIPPEDVORONOIDIAGRAM Generates a Voronoi decomposition of a polygonal
%region relative to a set of points contained within that region
%
%   INPUT PARAMETERS:
%
%       - DT:           Delaunay triangulation, specified as a scalar
%                       'delaunayTriangulation' object with #V points
%
%       - bdyPoly:      #Px2 coordinates of the bounding polygon
%
%   OUTPUT PARAMETERS:
%
%       - v:        #VVx2 coordinates of the Voronoi vertices
%
%       - c:        #Vx1 cell array containing the indices into v defining
%                   the clipped Voronoi polygon of each point in DT
%
%   by Dillon Cislo 2024/10/03

% Process triangulation input
assert(isa(DT, 'delaunayTriangulation'), ['Please supply input seed ' ...
    'points as ''delaunayTriangulation'' object']);
% numPoints = size(DT.Points, 1);
E = DT.edges;

% Process bounding polygon input
validateattributes(bdyPoly, {'numeric'}, {'2d', 'ncols', 2, ...
    'finite', 'real'});
% numPolyPoints = size(bdyPoly, 1);
bdyPoly = polyshape(bdyPoly);
[bboxX, bboxY] = boundingbox(bdyPoly);
scale = 2 * max(diff(bboxX), diff(bboxY));

assert(all(inpolygon(DT.Points(:,1), DT.Points(:,2), ...
    bdyPoly.Vertices(:,1), bdyPoly.Vertices(:,2))), ...
    ['Some of the input points are not contained within ' ...
    'the bounding polygon']);

% Generate Voronoi diagram of input points
[v, c] = voronoiDiagram(DT);
cv = cellfun(@(x) v(x, :), c, 'Uni', false);

% Identify which points have Voronoi vertices not contained inside the
% bounding polygon/points on the convex hull
voutIDx = ~inpolygon(v(:,1), v(:,2), ...
    bdyPoly.Vertices(:,1), bdyPoly.Vertices(:,2));
coutIDx = cellfun(@(x) any(voutIDx(x)), c, 'Uni', true);
cinfIDx = cellfun(@(x) any(x == 1), c, 'Uni', true);

for i = find(cinfIDx).'

    % The current CCW polygon (including inf point)
    curC = c{i};
    curCV = cv{i};
    infID = find(curC == 1);
    assert(numel(infID) == 1, 'More than one inf point?');

    % Shift the points so that the inf point is first
    curC = circshift(curC, [0, (-infID+1)]);
    curCV = circshift(curCV, [(-infID+1), 0]);

    % IDs of all vertex neighbors to the current point
    vnnIDx = setdiff(unique(E(any(E == i, 2), :)), i);

    if (numel(vnnIDx) > 2)

        % Find the unit vector in the direction of the Delaunay triangulation
        % edge normal to the Voronoi edge running TOWARDS infinity (CCW)
        e1_ID = vnnIDx(cellfun(@(x) all(ismember([1 curC(end)], x)), ...
            c(vnnIDx), 'Uni', true));
        if (numel(e1_ID) > 1)
            badNeighbor = cellfun(@(x) any(ismember(curC(2:(end-1)), x)), ...
                c(e1_ID), 'Uni', true);
            e1_ID(badNeighbor) = [];
        end
        assert(numel(e1_ID) == 1, 'e1 has %d neighbors', numel(e1_ID));

        % Find the unit vector in the direction of the Delaunay triangulation
        % edge normal to the Voronoi edge running AWAY FROM infinity (CCW)
        e2_ID = vnnIDx(cellfun(@(x) all(ismember([1 curC(2)], x)), ...
            c(vnnIDx), 'Uni', true));
        if (numel(e2_ID) > 1)
            badNeighbor = cellfun(@(x) any(ismember(curC(3:end), x)), ...
                c(e2_ID), 'Uni', true);
            e2_ID(badNeighbor) = [];
        end
        assert(numel(e2_ID) == 1, 'e2 has %d neighbors', numel(e2_ID));
        

    else
        
        [tx, ty] = poly2ccw(DT.Points([i; vnnIDx], 1), ...
            DT.Points([i; vnnIDx], 2));
        tIDx = knnsearch(DT.Points, [tx, ty]);
        tIDx = circshift(tIDx, [-find(tIDx == i)+1, 0]);

        e1_ID = tIDx(3);
        e2_ID = tIDx(2);

    end

    e1 = DT.Points(i,:) - DT.Points(e1_ID,:);
    e1 = e1 ./ sqrt(sum(e1.^2, 2));

    % The direction of the Voronoi edge running TOWARDS inifinity
    n1 = [e1(2), -e1(1)];

    e2 = DT.Points(e2_ID,:) - DT.Points(i,:);
    e2 = e2 ./ sqrt(sum(e2.^2, 2));

    % The direction of the Voronoi edge running AWAY FROM infinity
    n2 = [e2(2), -e2(1)];

    cv{i} = [ curCV(end, :) + scale .* n1; ...
        curCV(2, :) + scale .* n2; curCV(2:end, :) ];

end

% Compute the intersection of all offending polygons with the bounding
% polygon
cout = cellfun(@(x) intersect(polyshape(x), bdyPoly), ...
    cv(coutIDx), 'Uni', false);
cout = cellfun(@(x) [x.Vertices(:,1), x.Vertices(:,2)], ...
    cout, 'Uni', false);
cv(coutIDx) = cout;

% Sort polygons to be CCW
for i = 1:numel(cv)
    [px, py] = poly2ccw(cv{i}(:,1), cv{i}(:,2));
    cv{i} = [px, py];
end

% Compute the number of vertices in each polygon
numVC = cellfun(@(x) size(x,1), cv, 'Uni', true).';

% Update the vertex and polygon ID lists
[v, ~, c] = unique(cell2mat(cv), 'rows');
c = mat2cell(c, numVC, 1);
c = cellfun(@transpose, c, 'Uni', false);

end

