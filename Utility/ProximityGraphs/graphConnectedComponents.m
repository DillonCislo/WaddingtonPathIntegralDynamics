function [numCC, CC, sizeCC] = graphConnectedComponents(E, numV)
%GRAPHCONNECTEDCOMPONENTS Determine the connected components of a graph
%
%   INPUT PARAMETERS:
%
%       - E:        #E x 2 graph edge connectivity list
%
%       - numV:     The number of vertices in the graph
%
%   OUTPUT PARAMETERS:
%
%       - numCC:    The number of connected components in the graph
%
%       - CC:        #numV x 1 vector. CC(i) is the connected component of
%                   vertex i
%
%       - sizeCC:   #numCC x 1 vector. sizeCC(i) is the number of vertices
%                   contained in the ith connectedcomponent
%
%   by Dillon Cislo 02/01/2023

% Input Processing --------------------------------------------------------

validateattributes(E, {'numeric'}, ...
    {'2d', 'ncols', 2, 'finite', 'integer', 'positive', 'real'});

if (nargin < 2), numV = max(E(:)); end
validateattributes(numV, {'numeric'}, ...
    {'scalar', 'integer', 'finite', 'positive', 'real'});

assert( max(E(:)) <= numV, ...
    'Edge connectivity list contains invalid vertex IDs');

% Calculate Connected Components ------------------------------------------

% Construct an #numV x #numV vertex adjacency matrix based on graph edges
A = sparse( [ E(:,1); E(:,2) ], [ E(:,2), E(:,1) ], ...
    ones(2*size(E,1), 1), numV, numV );

% Construct the Dulmage-Mendelsohn decomposition
[ p, ~, r ] = dmperm( A + speye(size(A)) );

% The number of connected components
numCC = numel(r)-1;

% A 1x#N row vector. CC(i) is the connected component ID of point i
CC = cumsum( full( sparse( 1, r(1:end-1), 1, 1, size(A,1) ) ) );
CC(p) = CC;

% The number of points in each connected component
sizeCC = histcounts(CC, 0.5:1:(numCC+0.5));

end

