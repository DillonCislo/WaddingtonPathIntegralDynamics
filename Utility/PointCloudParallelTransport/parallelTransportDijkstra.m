%PARALLELTRANSPORTDIJKSTRA  Parallel-transport Dijkstra for geodesic
%distances on point cloud graphs representing Riemannian submanifolds
%embedded in Euclidean space. Computes a pairwise geodesic distance matrix
%on by propagating an orthonormal frame along between tangent frames, and
%defines the geodesic distance from the norm of the transported
%displacement vector.
%
%   INPUT PARAMETERS:
%
%       - X:            #P x ambiDim set of point cloud coordinates.
%
%       - E:            #E x 2 undirected edge list defining graph
%                       connectivity
%
%       - allBases:     #P x 1 cell array. allBases{i} is a ambiDim x dim
%                       array of orthonormal tangent space basis vectors
%                       for the point at X(i).
%
%   OUTPUT PARAMETERS:
%
%       - D:            #P x #P  symmetric matrix of pairwise geodesic
%                       distances. 
%
%   by Dillon Cislo 2026/02/11