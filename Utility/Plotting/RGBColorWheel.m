function RGBColorWheel(ax, labels, labelPositions)
%RGBCOLORWHEEL Generates an RGB color wheel visualization with user
%specified labels.
%
%   INPUT PARAMETERS:
%
%       - ax:               The axis within which to generate the color
%                           wheel plot
%
%       - labels:           Text to display at a user specified set of
%                           label sites
%
%       - labelPositions:   The label positions associated with the
%                           specified labels. Possible label sites include 
%                           {'r', 'g', 'b', 'rg', 'rb', 'gb', 'rgb'} or a
%                           custom (x,y)-coordinate pair located within the
%                           unit disk
%
%  by Dillon Cislo 02/26/2023

%--------------------------------------------------------------------------
% Input Processing
%--------------------------------------------------------------------------
if ((nargin < 1) || isempty(ax)), ax = gca; end
if ((nargin < 2) || isempty(labels)), labels = {}; end
if ((nargin < 3) || isempty(labelPositions)), labelPositions = {}; end

assert(iscell(labels) && iscell(labelPositions), ...
    'Label data must be supplied as cell arrays');
assert(isequal(size(labels), size(labelPositions)), ...
    'Invalid label data sizes');

%--------------------------------------------------------------------------
% Generate Figure
%--------------------------------------------------------------------------

% Generate a triangulation of the unit disk
diskTri = diskTriangulation(30);
F = diskTri.ConnectivityList;
V = diskTri.Points;

% Convert vertex locations to polar coordinates
r = sqrt(sum(V.^2, 2));
phi = wrapTo2Pi(atan2(V(:,2), V(:,1))-pi/2);

% Generate vertex colors using HSV
vertexHues = phi/(2*pi);
vertexSaturation = r;
vertexColors = hsv2rgb([vertexHues, vertexSaturation, ones(size(V,1), 1)]);

patch('Faces', F, 'Vertices', V, 'FaceVertexCData', vertexColors, ...
    'FaceColor', 'interp', 'EdgeColor', 'none');

axis equal tight off

% Add Labels --------------------------------------------------------------

fixedPositionLabels = {'r', 'g', 'b', 'gr', 'br', 'bg', 'bgr'};

for i = 1:numel(labels)
   
    curLabel = labels{i};
    validateattributes(curLabel, {'char'}, {'vector'});
    
    curLabelPos = labelPositions{i};
    if isnumeric(curLabelPos)
        
        validateattributes(curLabelPos, {'numeric'}, {'vector', ...
            'finite', 'real', 'numel', 2});
        if (size(curLabelPos, 1) == 2)
            curLabelPos = curLabelPos.';
        end
        assert(sqrt(sum(curLabelPos)) <= 1, ...
            'Custom label positions must lie within the unit disk');
        
        textborder(curLabelPos(1), curLabelPos(2), curLabel, ...
            'w', 'k', ...
            'horiz', 'center', 'vert', 'top', ...
            'FontWeight', 'bold');
        
    elseif ischar(curLabelPos)
        
        curLabelPos = sort(lower(curLabelPos));
        assert(ismember(curLabelPos, fixedPositionLabels), ...
            'Invalid fixed position label supplied');
        
        switch curLabelPos
            case 'r'
                textborder(0, 1, curLabel, ...
                    'w', 'k', 'FontWeight', 'bold', ...
                    'horiz', 'center', 'vert', 'top');
            case 'gr'
                textborder(cos(5*pi/6), sin(5*pi/6), curLabel, ...
                    'w',  'k', 'FontWeight', 'bold', ...
                    'horiz', 'left', 'vert', 'top');
            case 'g'
                textborder(cos(7*pi/6), sin(7*pi/6), curLabel, ...
                    'w', 'k', 'FontWeight', 'bold', ...
                    'horiz', 'left', 'vert', 'bottom');
            case 'bg'
                textborder(0, -1, curLabel, ...
                    'w', 'k', 'FontWeight', 'bold', ...
                    'horiz', 'center', 'vert', 'bottom');
            case 'b'
                textborder(cos(11*pi/6), sin(11*pi/6), curLabel, ...
                    'w', 'k', 'FontWeight', 'bold', ...
                    'horiz', 'right', 'vert', 'bottom');
            case 'br'
                textborder(cos(pi/6), sin(pi/6), curLabel, ...
                    'w', 'k', 'FontWeight', 'bold', ...
                    'horiz', 'right', 'vert', 'top');
            case 'bgr'
                textborder(0, 0, curLabel, ...
                    'w', 'k', 'FontWeight', 'bold', ...
                    'horiz', 'center', 'vert', 'middle');
            otherwise
                error('Invalid fixed position label supplied');
        end
        
    else
        
        error('Invalid label position data type');
        
    end
    
    labelPositions{i} = curLabelPos;
    
end

end

