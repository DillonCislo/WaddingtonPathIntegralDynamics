function D = pointLineDistance(P, V1, V2)
%POINTLINEDISTANCE Calculate the perpendicular distance from a set of
%points to a set of lines in any dimension
%
%   INPUT PARAMETERS:
%
%       - P:    #Pxdim query point coordinate list
%       - V1:   #Lxdim line segment start point
%       - V2:   #Lxdim line segment end point
%
%   OUTPUT PARAMETERS:
%
%       - D:    #Px#L perpendicular distance matrix
%
%   by Dillon Cislo 2023/07/24

validateattributes(P, {'numeric'}, {'2d', 'finite', 'real'});
validateattributes(V1, {'numeric'}, {'2d', 'finite', 'real'});
validateattributes(V2, {'numeric'}, {'2d', 'finite', 'real'});

dim = size(P,2);
numPoints = size(P,1);
numSegments = size(V1,1);

assert(size(V2,1) == numSegments, 'Invalid line segment input');
assert((size(V1,2) == dim) && (size(V2,2) == dim), ...
    'All input must have the same spatial dimension');

L12 = sqrt(sum((V2-V1).^2, 2)); % Distance from V1-->V2 (#Lx1)
L12 = repmat(L12.', [numPoints 1]); % (#Px#L)

% Distance from V1-->P (#Px#L)
L1P = repmat(permute(P, [1 3 2]), [1 numSegments 1]) - ...
    repmat(permute(V1, [3 1 2]), [numPoints 1 1]);
L1P = sqrt(sum(L1P.^2, 3)); 

% Distance from V2-->P (#Px#L)
L2P = repmat(permute(P, [1 3 2]), [1 numSegments 1]) - ...
    repmat(permute(V2, [3 1 2]), [numPoints 1 1]);
L2P = sqrt(sum(L2P.^2, 3)); 

% Area of each triangle V1-->V2-->P
s = (L12 + L1P + L2P)/2;
A = sqrt(s .* (s-L12) .* (s-L1P) .* (s-L2P));

% The perpendicular distance to the line
D = 2 .* A ./ L12;

end