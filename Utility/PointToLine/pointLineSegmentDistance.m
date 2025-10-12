function D = pointLineSegmentDistance(P, V1, V2)
%POINTLINESEGMENTDISTANCE Calculate the perpendicular distance from a set
%of points to a set of line segments in any dimension
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

% The perpendicular distance to the line (not necessarily line segment)
% defined by V1 and V2
D = 2 .* A ./ L12;

% The cosine of the angle at V1 (#Px#L)
cosTheta1 = (L12.^2 + L1P.^2 - L2P.^2) ./ (2 .* L12 .* L1P);
isObtuse1 = cosTheta1 < 0;

% The cosine of the angle at V2 (#Px#L)
cosTheta2 = (L12.^2 + L2P.^2 - L1P.^2) ./ (2 .* L12 .* L2P);
isObtuse2 = cosTheta2 < 0;

% Wherever one of the end point angles is obtuse, the true distance is just
% the distance to the corresponding end point
D(isObtuse1) = L1P(isObtuse1);
D(isObtuse2) = L2P(isObtuse2);

end

