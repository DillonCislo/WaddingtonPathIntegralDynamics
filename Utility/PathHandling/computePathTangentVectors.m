function tangentVecs = computePathTangentVectors( ...
    X, pathIDx, smoothIters, isLoop)
%COMPUTEPATHTANGENTVECTORS Computes (smoothed) tangent vectors along a path
%through a point cloud
%
%   INPUT PARAMETERS:
%
%       - X:            #N x dim set of input points on which the dynamics
%                       are defined
%
%       - pathIDx:      #P x 1 vector of indices defining the path through
%                       the input point cloud
%
%       - smoothIters:  Number of smoothing iterations. Smoothing is done
%                       via and is not path length/geometry aware. Values
%                       <= 0 correspond to no smoothing.
%
%       - isLoop:       Whether the path is a loop.
%
%   OUTPUT PARAMETERS:
%
%       - tangentVecs:  #P x dim set of unit tangent vectors to the path
%                       defined for each point in the path
%
%   by Dillon Cislo 2026/06/17

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
if (nargin < 3), smoothIters = 0; end
if (nargin < 4), isLoop = false; end

validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'}, ...
    'computePathTangentVectors', 'X');
numPoints = size(X,1); dim = size(X,2);

validateattributes(pathIDx, {'numeric'}, {'vector', 'finite', ...
    'integer', 'real', 'positive', '<=' numPoints}, ...
    'computePathTangentVectors', 'pathIDx');
if (size(pathIDx, 2) ~= 1), pathIDx = pathIDx.'; end

validateattributes(smoothIters, {'numeric'}, {'vector', 'finite', ...
    'integer', 'real'}, 'computePathTangentVectors', 'smoothIters');
validateattributes(isLoop, {'logical'}, {'scalar'}, ...
    'computePathTangentVectors', 'isLoop')

%--------------------------------------------------------------------------
% COMPUTE TANGENT VECTORS
%--------------------------------------------------------------------------

if isLoop

    if (pathIDx(end) == pathIDx(1)), pathIDx(end) = []; end
    assert(isequal(pathIDx, unique(pathIDx, 'stable')), ...
    'Path contains self-intersections');

    pathEdges = [pathIDx, circshift(pathIDx, [-1, 0])];
    tangentVecs = X(pathEdges(:,2), :) - X(pathEdges(:,1), :);
    pathEdgeLengths = sqrt(sum(tangentVecs.^2, 2));

    % Perform a length weighted average of each adjacent edge vector onto
    % path vertices
    tangentVecs = ...
        (tangentVecs + circshift(tangentVecs, [1, 0])) ./ ...
        (pathEdgeLengths + circshift(pathEdgeLengths, [1, 0]));

    if smoothIters > 0
        for i = 1:smoothIters
            tangentVecs = [tangentVecs; tangentVecs; tangentVecs];
            tangentVecs = smoothdata(tangentVecs, 1, "movmean");
            tangentVecs = tangentVecs((numel(pathIDx)+1):(2 * numel(pathIDx)), :);
        end
    end

else

    assert(isequal(pathIDx, unique(pathIDx, 'stable')), ...
    'Path contains self-intersections');

    pathEdges = [pathIDx(1:(end-1)), pathIDx(2:end)];
    tangentVecs = X(pathEdges(:,2), :) - X(pathEdges(:,1), :);
    pathEdgeLengths = sqrt(sum(tangentVecs.^2, 2));

    % Perform a length weighted average of each adjacent edge vector onto
    % path vertices. End point vertices just receive their only adjacent
    % edge vector
    tangentVecs = [tangentVecs(1,:); ...
        (tangentVecs(1:(end-1), :) + tangentVecs(2:end, :)) ./ ...
        (pathEdgeLengths(1:(end-1), :) + pathEdgeLengths(2:end, :) ); ...
        tangentVecs(end,:)];

    if smoothIters > 0
        for i = 1:smoothIters
            tangentVecs = smoothdata(tangentVecs, 1, "movmean");
        end
    end

end




end