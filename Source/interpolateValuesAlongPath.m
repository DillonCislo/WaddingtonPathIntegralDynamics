function [interpVals, interpIDx] = ...
    interpolateValuesAlongPath(endPointVals, allPaths, varargin)
%INTERPOLATEVALUESALONGPATH Given a set of paths (defined by an ordered
%list of point IDs) and set of end point values for each supplied path,
%this function linearly interpolates those values along the path and
%concatenates the results into a single output vector. The results are
%suitable as equality constraints for various optimization procedures.
%
%   INPUT PARAMETERS:
%
%       - endPointVals:     #P x 2 list of path end point values
%
%       - allPaths:         #P x 1 cell array. allPaths{i} is an an ordered
%                           set of point IDs defining that path
%
%   OPTIONAL INPUT PARAMETERS (Name, Value)-Pairs:
%
%       - ('PathLengths', allPathLengths = {}): A #P x 1 cell array
%       containing the length of each each edge in the associated path
%       (This doesn't have to just be physical length - you can supply any
%       set of positive weights)
%
%       - ('InterpolationMethod', interpMethod = 'weighted'): Whether to
%       interpolate using the path length weights or to just interpolate
%       according to the number of points along the path. If no path
%       lengths are supplied, 'weighted' and 'unweighted' are equivalent
%
%       - ('CollisionMethod', collisionMethod = 'mean'): How to handle the
%       case where multiple paths intersect at a subset of the points.
%
%       - ('Verbose', verbose = false): Whether or not to print warnings
%       which collisions occur
%
%   OUTPUT PARAMETERS:
%
%       - interpVals:   #IV x 1 vector of interpolated values (includes the
%                       endpoint values of each path).
%
%       - interpIDx:    #IV x 1 vector of indices into the point set
%                       corresponding to the values in 'interpVals'
%
%   by Dillon Cislo 2024/03/26

%--------------------------------------------------------------------------
% INPUT PROCESSING
%--------------------------------------------------------------------------
validateattributes(endPointVals, {'numeric'}, {'2d', 'ncols', 2, ...
    'finite', 'real'});
numPaths = size(endPointVals, 1);

validateattributes(allPaths, {'cell'}, {'vector', 'numel', numPaths});
cellfun(@(x) validateattributes(x, {'numeric'}, {'vector', 'integer', ...
    'positive', 'finite', 'real'}), allPaths, 'Uni', false);
assert(all(cellfun(@(x) numel(x) > 1, allPaths, 'Uni', true)), ...
    'Paths must have at least two points');

% OPTIONAL INPUT PROCESSING -----------------------------------------------

allPathLengths = {};
interpMethod = 'weighted';
collisionMethod = 'mean';
verbose = false;

supportedOptions = {'PathLengths', 'InterpolationMethod', ...
    'CollisionMethod', 'Verbose'};
checkSupportedOptions(supportedOptions, varargin);

allInterpolationMethods = {'weighted', 'unweighted'};
allCollisionMethods = {'mean', 'none'};

for i = 1:length(varargin)

    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'logical'), continue; end

    if strcmpi(varargin{i}, 'PathLengths')
        allPathLengths = varargin{i+1};
        if ~isempty(allPathLengths)
            validateattributes(allPathLengths, {'cell'}, ...
                {'vector', 'numel', numPaths});
        end
    end

    if strcmpi(varargin{i}, 'InterpolationMethod')
        interpMethod = lower(varargin{i+1});
        validateattributes(interpMethod, {'char'}, {'vector'});
        assert(ismember(interpMethod, allInterpolationMethods), ...
            'Invalid interpolation method supplied');
    end

    if strcmpi(varargin{i}, 'CollisionMethod')
        collisionMethod = lower(varargin{i+1});
        validateattributes(collisionMethod, {'char'}, {'vector'});
        assert(ismember(collisionMethod, allCollisionMethods), ...
            'Invalid collision method supplied');
    end

    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end

end

%--------------------------------------------------------------------------
% PERFORM INTERPOLATION
%--------------------------------------------------------------------------

interpVals = cell(numPaths, 1);
interpIDx = cell(numPaths, 1);

for i = 1:numPaths

    curPath = allPaths{i};
    if (size(curPath, 2) ~= 1), curPath = curPath.'; end

    if (isempty(allPathLengths) || strcmpi(interpMethod, 'unweighted'))

        interpCoeffs = linspace(0, 1, numel(curPath)).';

    else

        curPathLengths = allPathLengths{i};
        validateattributes(curPathLengths, {'numeric'}, ...
            {'vector', 'finite', 'real'});
        if (size(curPathLengths, 2) ~= 1)
            curPathLengths = curPathLengths.';
        end

        if (numel(curPathLengths) == numel(curPath))

            goodPathLengths = (curPathLengths(1) == 0) && ...
                all(diff(curPathLengths, 1) > 0);
            assert(goodPathLengths, ['Invalid path lengths supplied. ' ...
                'If supplying cumulative weights, the first entry ' ...
                'must == 0 and the vector must monotonically increase']);

        elseif (numel(curPathLengths) == (numel(curPath)-1))

            goodPathLengths = all(curPathLengths > 0);
            assert(goodPathLengths, ['Invalid path lengths supplied. ' ...
                'If supplying raw weights, all weights must be positive']);
            curPathLengths = [0; cumsum(curPathLengths)];

        else

            error('Supplied path lengths have invalid size');

        end

        interpCoeffs = curPathLengths ./ curPathLengths(end);

    end

    interpVals{i} = (1-interpCoeffs) .* endPointVals(i,1) + ...
        interpCoeffs .* endPointVals(i,2);

    interpIDx{i} = curPath;

end

interpVals = cell2mat(interpVals);
interpIDx = cell2mat(interpIDx);

%--------------------------------------------------------------------------
% COLLISION HANDLING
%--------------------------------------------------------------------------

if strcmpi(collisionMethod, 'none'), return; end

% Detect collisions
dupl = find_duplicate_rows(interpIDx);
if isempty(dupl), return; end

if verbose, warning('Some known vertices assigned multiple values'); end

if strcmpi(collisionMethod, 'mean')

    for i = 1:length(dupl)
        interpVals(dupl(i).idx(1)) = mean(interpVals(dupl(i).idx));
        interpVals(dupl(i).idx(2:end)) = NaN;
    end

    rmIDx = isnan(interpVals);
    interpVals(rmIDx) = [];
    interpIDx(rmIDx) = [];

else

    error('Invalid collision method supplied');

end

assert((numel(interpVals) == numel(interpIDx)) && ...
    ~any(isnan(interpVals)) && ~any(isnan(interpIDx)) && ...
    isequal(sort(interpIDx), unique(interpIDx)), ...
    'Collision handling failed');

end