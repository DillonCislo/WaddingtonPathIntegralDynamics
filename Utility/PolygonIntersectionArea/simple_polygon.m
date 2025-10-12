function [x, y, dt] = simple_polygon(numSides)
%SIMPLE_POLYGON Generate a random simple polygon with a prescribed number
%of sides using repeated Delaunay refinements.
%
%   INPUT PARAMETERS:
%
%       - numSides:     Desired polygon edge count. Values < 3 return an
%                       empty polygon and a default triangulation object.
%
%   OUTPUT PARAMETERS:
%
%       - x:            (#numSides+1) x 1 vector of polygon vertex x-coords
%                       returned in counter-clockwise order. The final
%                       entry repeats the first vertex for convenience.
%
%       - y:            Matching y-coordinates for the generated polygon.
%
%       - dt:           Delaunay triangulation object used to build the
%                       polygon boundary.
%
%   by Dillon Cislo (documentation refreshed 2025/10/12)

    if numSides < 3
        x = [];
        y = [];
        dt = DelaunayTri();
        return
    end

    oldState = warning('off', 'MATLAB:TriRep:PtsNotInTriWarnId');

    fudge = ceil(numSides/10);
    x = rand(numSides+fudge, 1);
    y = rand(numSides+fudge, 1);
    dt = DelaunayTri(x, y);
    boundaryEdges = freeBoundary(dt);
    numEdges = size(boundaryEdges, 1);

    while numEdges ~= numSides
        if numEdges > numSides
            triIndex = vertexAttachments(dt, boundaryEdges(:,1));
            triIndex = triIndex(randperm(numel(triIndex)));
            keep = (cellfun('size', triIndex, 2) ~= 1);
        end
        if (numEdges < numSides) || all(keep)
            triIndex = edgeAttachments(dt, boundaryEdges);
            triIndex = triIndex(randperm(numel(triIndex)));
            triPoints = dt([triIndex{:}], :);
            keep = all(ismember(triPoints, boundaryEdges(:,1)), 2);
        end
        if all(keep)
            warning('Couldn''t achieve desired number of sides!');
            break
        end
        triPoints = dt.Triangulation;
        triPoints(triIndex{find(~keep, 1)}, :) = [];
        dt = TriRep(triPoints, x, y);
        boundaryEdges = freeBoundary(dt);
        numEdges = size(boundaryEdges, 1);
    end

    boundaryEdges = [boundaryEdges(:,1); boundaryEdges(1,1)];
    x = dt.X(boundaryEdges, 1);
    y = dt.X(boundaryEdges, 2);

    warning(oldState);

end