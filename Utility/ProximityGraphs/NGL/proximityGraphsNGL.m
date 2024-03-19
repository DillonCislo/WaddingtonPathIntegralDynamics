function [E, DE] = proximityGraphsNGL(X, graphType, kNN, ...
    betaParam, forceConnectivity, outputFormat)
%PROXIMITYGRAPHSNGL Produce on of a number of proximity graphs for a given
%input point cloud using the NGL library. Graphs can be computed
%exhaustively (generally O(N^3)) or approximately by only considering edges
%between a subset of k-nearest neighbors
%
%   INPUT PARAMETERS:
%
%       - X:            #N x #D input point cloud
%
%       - graphType:    The desired output graph type. Options
%                       include:
%                               - 'RelativeNeighborGraph' or 'rng'
%                               - 'GabrielGraph' or 'gg'
%                               - 'BetaSkeleton' or 'bs
%                               - 'RelaxedRelativeNeighborGraph' or 'rrng'
%                               - 'RelaxedGabrielGraph' or 'rgg'
%                               - 'RelaxedBetaSkeleton' or 'rbs'
%
%       - kNN:         	The number of nearest neighbors used to
%                       approximate the desired output graph. Any
%                       number <= 0 will use an exhaustive
%                       construction.
%
%       - betaParam:    The value of beta used to construct the skeleton
%
%       - forceConnectivity:    If true, the output graph is forced to be
%                               simply connected. NOTE: We cannot force the
%                               connectivity of a directed graph yet.
%
%       - outputFormat:	The initial output from the NGL
%                     	functions are the edges of a DIRECTED graph (i.e.
%                     	order of point
%                      	indices matter. You can choose 'symmetric' to
%                      	convert this to an
%                       UNDIRECTED graph including all of the edges in the
%                       original directed graph or 'mutual' to output an
%                       UNDIRECTED graph containing only those edges in the
%                       original graph that were shared by both end points
%
%   OUTPUT PARAMETERS:
%
%       - E:    #E x 2 edge connectivity list
%
%       - DE:   #E x 1 edge length list
%
%   by Dillon Cislo 02/01/2023

%--------------------------------------------------------------------------
% Input Processing
%--------------------------------------------------------------------------
if (nargin < 1), error('Please supply input point cloud'); end
if (nargin < 2), error('Please supply desired graph type'); end

validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X,1);

validateattributes(graphType, {'char'}, {'vector'});
graphType = lower(graphType);
allGraphTypes = {'relativeneighborgraph', 'rng', ...
    'gabrielgraph', 'gg', 'betaskeleton', 'bs', ...
    'relaxedrelativeneighborgraph', 'rrng', ...
    'relaxedgabrielgraph', 'rgg', 'relaxedbetaskeleton', 'rbs'};
assert(ismember(graphType, allGraphTypes), 'Invalid graph type');      
    
if (nargin < 3), kNN = -1; end
validateattributes(kNN, {'numeric'}, ...
    {'scalar', 'integer', 'finite', 'real'});

if (nargin < 4), betaParam = 1; end
validateattributes(betaParam, {'numeric'}, ...
    {'scalar', 'finite', 'real', '>=', 0, '<=', 1});

if (nargin < 5), forceConnectivity = false; end
validateattributes(forceConnectivity, {'logical'}, {'scalar'});

if (nargin < 6), outputFormat = 'mutual'; end
validateattributes(outputFormat, {'char'}, {'vector'});
outputFormat = lower(outputFormat);
assert(ismember(outputFormat, {'directed', 'mutual', 'symmetric'}), ...
    'Invalid output format');

%--------------------------------------------------------------------------
% Calculate Proximity Graph Using NGL
%--------------------------------------------------------------------------
% I don't know why, but the 'NGLPointSet' branch of the NGL appears broken
% for relaxed graph construction. Exhaustive relaxed graphs are computed
% using the 'ANNPointSet' branch

if any(strcmpi(graphType, {'relativeneighborgraph', 'rng'}))
    graphType = 1;
elseif any(strcmpi(graphType, {'gabrielgraph', 'gg'}))
    graphType = 2;
elseif any(strcmpi(graphType, {'betaskeleton', 'bs'}))
    graphType = 3;
elseif any(strcmpi(graphType, {'relaxedrelativeneighborhoodgraph', 'rrng'}))
    graphType = 4;
    if (kNN <= 0), kNN = numPoints-1; end
elseif any(strcmpi(graphType, {'relaxedgabrielgraph', 'rgg'}))
    graphType = 5;
    if (kNN <= 0), kNN = numPoints-1; end
elseif any(strcmpi(graphType, {'relaxedbetaskeleton', 'rbs'}))
    graphType = 6;
    if (kNN <= 0), kNN = numPoints-1; end
else
    error('Invalid graph type (switch)');
end

E = proximity_graphs_ngl(X, graphType, kNN, betaParam);

if strcmpi(outputFormat, 'mutual')
    
    E = sortrows(sort(E, 2));
    [~, uniqueRowIDx] = unique(E, 'rows');
    duplRowIDx = setdiff((1:size(E,1)).', uniqueRowIDx);
    E = E(duplRowIDx, :);
    
elseif strcmpi(outputFormat, 'symmetric')
    
    E = sortrows(sort(E,2));
    [E, ~] = unique(E, 'rows');
    
end

DE = [];
if (nargout >= 2)
    DE = X(E(:,2), :) - X(E(:,1), :);
    DE = sqrt(sum(DE.^2, 2));  
end

%--------------------------------------------------------------------------
% Handle Graph Connectivity
%--------------------------------------------------------------------------

if (forceConnectivity && any(strcmpi(outputFormat, {'mutual', 'symmetric'})))
    
    % Construct an #N x #N vertex adjacency matrix based on graph edges
    A = sparse( [ E(:,1); E(:,2) ], [ E(:,2), E(:,1) ], ...
        ones(2*size(E,1), 1), numPoints, numPoints );
    
    % Construct the Dulmage-Mendelsohn decomposition
    [ p, ~, r ] = dmperm( A + speye(size(A)) );
    
    % The number of connected components
    numCC = numel(r)-1;
    if (numCC == 1), return; end
    
    % A 1x#N row vector. C(i) is the connected component ID of point i
    C = cumsum( full( sparse( 1, r(1:end-1), 1, 1, size(A,1) ) ) );
    C(p) = C;
    
    % The number of points in each connected component
    sizeCC = histcounts(C, 0.5:1:(numCC+0.5));
    
    % A list of connected component IDs sorted by descending size
    [~, ccIDx] = sort(sizeCC, 'descend');
    
    while (numel(ccIDx) ~= 1)
        
        % Join the smallest component with its nearest neighbor
        
        
        smallCCID = ccIDx(end); % The ID of the smallest connected component
        smallIDx = (C == smallCCID).'; % The point IDs of the points belonging to the smallest component
        
        otherIDx = find(~smallIDx); % IDs of all other points in the original cloud
        smallIDx = find(smallIDx); % IDs of smallest component points in the original cloud
        
        smallX = X(smallIDx, :); % Point locations of points in the smallest component
        otherX = X(otherIDx, :); % All other point locations
        
        % Find the nearest neighbors/distances for the disconnected sets
        [nnIDx, nnDist] = knnsearch(otherX, smallX, 'K', 1);
        [~, shortestEdge] = min(nnDist);
        
        smallPID = smallIDx(shortestEdge); % The ID of the point in the smallest component that will be joined
        otherPID = otherIDx(nnIDx(shortestEdge)); % The ID of the neighboring point that will be joined
        nnDist = nnDist(shortestEdge); % The distance between the two points comprising the new edge
        
        % Add the new edge to the edge/distance list
        E = [E; sort([smallPID, otherPID])];
        DE = [DE; nnDist];
        
        otherCCID = C(otherPID); % The connected component of the other point in the new edge
        C(smallIDx) = otherCCID; % Update the component ID of all points in the former smallest component
        ccIDx(end) = []; % Remove the smallest connected component from the active list
        
    end
    
end


end

