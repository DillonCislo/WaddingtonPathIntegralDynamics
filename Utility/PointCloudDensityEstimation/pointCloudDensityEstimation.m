function [density, XOut, XOutIDx, ENG, maxIDx, ...
    saddleIDx, EJT, branchIDx, kdsigma] = ...
    pointCloudDensityEstimation(X, varargin)
%POINTCLOUDDENSITYESTIMATION Estimate the density and density maxima of an
%ND point cloud using kernel density estimation and topological
%simiplification methods.
%
%   INPUT PARAMETERS:
%
%       - X:        #N x #D input point cloud coordinates
%
%   SUBSAMPLING OPTIONAL PARAMETERS: Options for SPARTAN density aware
%   point cloud subsampling procedure. See also documentation for
%   'SPARTAN.m'. Users can also choose to do simple random subsampling
%
%       - ('SubsamplingMethod', ssMethod = 'spartan'): User can choose
%       between SPARTAN subsampling and random subsampling
%
%       - ('NumSamples', numSamples = min(numPoints, 10000)): The number of
%       points in the subsampled cloud
%
%       - ('OMTIterations', omtIter = 500): The number of iterations to run
%       the OMT process mapping the input distribution into the uniform
%       distribution on U[0,1]^D
%
%       - ('OMTWassersteinThresh', omtWasserThresh = 0): OMT will terminate
%       if the change in the Wasserstein distance of the iterates falls
%       below this threshold. WARNING: Not a great indicator of map quality
%       for large point set. Probably should just leave this equal to zero.
%
%       - ('QRPType', qrpType = 'sobol'): The quasi-random point generation
%       process used to produce a space-filling set of design points over
%       U[0,1]^D.
%
%       - ('QRPLeap', qrpLeap = 0): The leap parameter used for
%       quasi-random design point generation. Change to produce different
%       design points.
%
%       - ('QRPSkip', qrpSkip = 0): The skip parameter used for
%       quasi-random design point generation. Change to produce different
%       design points.
%
%       - ('QRPScramble', qrpScramble = 0): Whether or not to scramble the
%       quasi-random design points. Another knob to tune to adjust points.
%
%   NEIGHBORHOOD GRAPH OPTIONAL PARAMETERS: Options for construction of the
%   proximity graph structure on which the topological density tree
%   structure is established. See documentation of
%   'proximityGraphsNGL.m', 'proximityGraphs.m', and
%   'proximityGraphsKNN.m' for more details.
%
%       - ('NeighborhoodGraphType', nGraphType = 'relaxedgabrielgraph'):
%       The type of desired neighborhood graph. 
%
%       - ('NumNeighbors', kNN = 50): The number of nearest neighbors used
%       for a fast approximation of an exhaustive neighborhood graph type.
%       Any number <= 0 will trigger O(N^3) exhaustive construction.
%
%       - ('GraphBeta', betaParam = 1): The beta parameter used for beta
%       skeleton construction.
%
%       - ('ForceConnectivity', forceConnectivity = true): If true, the
%       output graph is forced to be simply connected.
%
%       - ('GraphOutputFormat', graphOutputFormat = 'mutual'): Method for
%       symmetrizing one-sided relaxed graph construction. 'Mutual'
%       construction means that only edges shared by both vertices are
%       retained. 'Symmetric' construction means all edges are retained.
%       NOTE: Directed graph construction is NOT allowed for this
%       algorithm.
%
%   KERNEL DENSITY ESTIMATION OPTIONAL PARAMETERS: Options to tune the
%   normalized Gaussian kernel density estimation procedure
%
%       - ('KDSigma', kdsigma = -1): The variance of the unit normalized
%       normal pdf used in the kernel density procedure. Values of kdsigma
%       <= 0 will result in an automatic heuristic estimation of point
%       cloud neighborhood sizes based on nearest neighbor distances.
%
%       - ('UpsampleEdges', upsampleEdges = true): Includes a virtual
%       upsampled point on midpoints of neighborhood graph edges. Samples
%       are only included when the density of the new point is less than
%       the density of its immediate neighbors. This step can improve
%       quality of density critical point detection.
%
%       - ('IncludeNonSamples', includeNonSamples = false): If true, points
%       removed during the subsampling procedure are trivially
%       re-introduced into the neighborhood graph/density calculation by
%       connecting them to their nearest neighbors.
%
%   TOPOLOGICAL SIMPLIFICATION OPTIONAL PARAMETERS: Options to select how
%   spurious branches are pruned from the density join tree structure
%
%       - ('PersistenceThreshold', persistenceThreshold = 0.05): Branches
%       whose persistence (i.e. maximum branch density minus minimum branch
%       density) is less than this number times the maximum density will
%       be pruned.
%
%       - ('SizeThreshold', sizeThreshold = 0): Branches whose total size
%       (i.e. number of vertices) is less than this number will be pruned.
%
%       - ('StabilityThreshold', stabilityThreshold = 0): Branches whose
%       total stability (i.e. sum of branch densities) is less than this
%       number will be pruned.
%
%       - ('DistanceThreshold', distThreshold = 0): Branches whose maxima
%       locations are less than this distance apart in feature space will
%       be merged keeping only the highest maximum
%
%       - ('CollisionMergeMethod', collisionMergeMethod = 'sequential'): If
%       the collision merge method is set to 'sequential', nearest
%       neighbors are merged pairwise until no pair of points is closer
%       than the  distance threshold. If the collision merge method is set
%       to 'simultaneous', then all intersecting r-neighborhoods are merged
%       into one. Simultaneous merging will generically decimate more
%       candidate maxima.
%
%       - ('MaxThreshold', maxThreshold = 0): Branches whose max values are
%       less than this threshold will be pruned.
%
%       - ('PlotBranchQuality', plotBranchQuality = false): If true, the
%       function will produce a plot of the branch quality metrics.
%
%   GENERAL OPTIONAL PARAMETERS:
%
%       - ('Verbose', verbose = false): Whether or not to print verbose
%       progress updates.
%
%   OUTPUT PARAMETERS:
%
%       - density:      #NP x 1 list of estimated densities for each point
%                       in the processed output point cloud.
%
%       - XOut:         #NP x #D output point cloud coordinates (i.e. after
%                       subsampling, upsampling, nonsample inclusion, and
%                       shuffling).
%
%       - XOutIDx:      #NP x 1 list of output point IDs in the original
%                       input cloud (i.e. X(XOutIDx(i), :) = XOut(i, :) for
%                       any point i that was in the input cloud and
%                       XOutIDx(j) = NaN for any upsampled edge midpoint).
%
%       - ENG:          #ENG x 2 edge connectivity list defining the
%                       neighborhood graph of the output point set in terms
%                       of the processed point IDs.
%
%       - maxIDx:       #max x 1 list of processed point IDs corresponding
%                       to density maxima.
%
%       - saddleIDx:    #sad x 1 list of processed point IDs corresponding
%                       to the saddles of the (pruned) density join tree.
%
%       - EJT:          #(NP-1) x 2 edge connectivity list defining the
%                       (pruned) density join tree.
%
%       - branchIDx:    #NP x 1 branch decomposition (i.e. branchIDx(i)
%                       gives the branch of the ith point in the processed
%                       cloud).
%
%       - kdsigma:      The variance of the unit normalized normal pdf 
%                       used in the kernel density procedure. Basically
%                       this will return the estimated value if a
%                       nonpositive value was supplied as input
%
% by Dillon Cislo 02/08/2023

%==========================================================================
% INPUT PROCESSING
%==========================================================================
validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
numPoints = size(X,1); dim = size(X,2);

% Set default subsampling options
ssMethod = 'spartan';
numSamples = min(numPoints, 10000);
omtIter = 500;
omtWasserThresh = 0;
qrpType = 'sobol';
qrpLeap = 0;
qrpSkip = 0;
qrpScramble = false;

% Set default neighborhood graph construction options
nGraphType = 'relaxedgabrielgraph';
kNN = 50;
betaParam = 1;
forceConnectivity = true;
graphOutputFormat = 'mutual';

% Set default kernel density estimation options
kdsigma = -1;
upsampleEdges = true;
includeNonSamples = false;

% Set topological simplification options
persistenceThreshold = 0.05;
sizeThreshold = 0;
stabilityThreshold = 0;
distThreshold = 0;
collisionMergeMethod = 'sequential';
maxThreshold = 0;
plotBranchQuality = false;

% Set default general options
verbose = false;

%--------------------------------------------------------------------------
% Parse Optional Inputs
%--------------------------------------------------------------------------

for i = 1:length(varargin)
   
    if isa(varargin{i}, 'double'), continue; end
    if isa(varargin{i}, 'single'), continue; end
    if isa(varargin{i}, 'logical'), continue; end
    
    % Subsamping Options --------------------------------------------------
    if strcmpi(varargin{i}, 'SubsamplingMethod')
        ssMethod = lower(varargin{i+1});
        validateattributes(ssMethod, {'char'}, {'vector'});
        assert(ismember(ssMethod, {'spartan', 'random'}), ...
            'Invalid subsampling method');
    end
    if strcmpi(varargin{i}, 'NumSamples')
        numSamples = varargin{i+1};
        validateattributes(numSamples, {'numeric'}, ...
            {'scalar', 'integer', 'finite', 'real'});
    end  
    if strcmpi(varargin{i}, 'OMTIterations')
        omtIter = varargin{i+1};
        validateattributes(omtIter, {'numeric'}, ...
            {'scalar', 'integer', 'positive', 'finite', 'real'});
    end    
    if strcmpi(varargin{i}, 'OMTWassersteinThresh')
        omtWasserThresh = varargin{i+1};
        validateattributes(omtWasserThresh, {'numeric'}, ...
            {'scalar', 'finite', 'real'});
    end    
    if strcmpi(varargin{i}, 'QRPType')
        qrpType = varargin{i+1};
        validateattributes(qrpType, {'char'}, {'vector'});
        assert(ismember(qrpType, {'sobol', 'halton'}), ...
            'Invalid quasi-random point generation type');
    end    
    if strcmpi(varargin{i}, 'QRPLeap')
        qrpLeap = varargin{i+1};
        validateattributes(qrpLeap, {'numeric'}, ...
            {'scalar', 'integer', 'finite', 'nonnegative', 'real'});
    end    
    if strcmpi(varargin{i}, 'QRPSkip')
        qrpSkip = varargin{i+1};
        validateattributes(qrpSkip, {'numeric'}, ...
            {'scalar', 'integer', 'finite', 'nonnegative', 'real'});
    end
    if strcmpi(varargin{i}, 'QRPScramble')
        qrpScramble = varargin{i+1};
        validateattributes(qrpScramble, {'logical'}, {'scalar'});
    end
    
    % Neighborhood Graph Construction Options -----------------------------
    if strcmpi(varargin{i}, 'NeighborhoodGraphType')
        nGraphType = varargin{i+1};
        validateattributes(nGraphType, {'char'}, {'vector'});
    end
    if strcmpi(varargin{i}, 'NumNeighbors')
        kNN = varargin{i+1};
        validateattributes(kNN, {'numeric'}, ...
            {'scalar', 'integer', 'finite', 'real'});
    end
    if strcmpi(varargin{i}, 'GraphBeta')
        betaParam = varargin{i+1};
        validateattributes(betaParam, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'});
    end
    if strcmpi(varargin{i}, 'ForceConnectivity')
        forceConnectivity = varargin{i+1};
        validateattributes(forceConnectivity, {'logical'}, {'scalar'});
    end
    if strcmpi(varargin{i}, 'GraphOutputFormat')
        graphOutputFormat = lower(varargin{i+1});
        validateattributes(graphOutputFormat, {'char'}, {'vector'});
        assert(ismember(graphOutputFormat, {'mutual', 'symmetric'}), ...
            'Invalid neighborhood graph output format');
    end   
    
    % Kernel Density Estimation Options -----------------------------------
    if strcmpi(varargin{i}, 'KDSigma')
        kdsigma = varargin{i+1};
        validateattributes(kdsigma, {'numeric'}, ...
            {'scalar', 'finite', 'real'});
    end
    if strcmpi(varargin{i}, 'UpsampleEdges')
        upsampleEdges = varargin{i+1};
        validateattributes(upsampleEdges, {'logical'}, {'scalar'});
    end
    if strcmpi(varargin{i}, 'IncludeNonSamples')
        includeNonSamples = varargin{i+1};
        validateattributes(includeNonSamples, {'logical'}, {'scalar'});
    end
    
    % Topological Simplification Options ----------------------------------
    if strcmpi(varargin{i}, 'PersistenceThreshold')
        persistenceThreshold = varargin{i+1};
        validateattributes(persistenceThreshold, {'numeric'}, ...
            {'scalar', 'finite', 'real'});
    end
    if strcmpi(varargin{i}, 'SizeThreshold')
        sizeThreshold = varargin{i+1};
        validateattributes(sizeThreshold, {'numeric'}, ...
            {'scalar', 'finite', 'real'});
    end
    if strcmpi(varargin{i}, 'StabilityThreshold')
        stabilityThreshold = varargin{i+1};
        validateattributes(stabilityThreshold, {'numeric'}, ...
            {'scalar', 'finite', 'real'});
    end
    if strcmpi(varargin{i}, 'DistanceThreshold')
        distThreshold = varargin{i+1};
        validateattributes(distThreshold, {'numeric'}, ...
            {'scalar', 'nonnegative', 'finite', 'real'});
    end
    if strcmpi(varargin{i}, 'CollisionMergeMethod')
        collisionMergeMethod = lower(varargin{i+1});
        validateattributes(collisionMergeMethod, {'char'}, {'vector'});
        assert(ismember(collisionMergeMethod, {'sequential', ...
            'simultaneous'}), 'Invalid collision merge method');
    end
    if strcmpi(varargin{i}, 'MaxThreshold')
        maxThreshold = varargin{i+1};
        validateattributes(maxThreshold, {'numeric'}, ...
            {'scalar', 'finite', 'real'});
    end
    if strcmpi(varargin{i}, 'PlotBranchQuality')
        plotBranchQuality = varargin{i+1};
        validateattributes(plotBranchQuality, {'logical'}, {'scalar'});
    end
    
    % General Options -----------------------------------------------------
    if strcmpi(varargin{i}, 'Verbose')
        verbose = varargin{i+1};
        validateattributes(verbose, {'logical'}, {'scalar'});
    end
    
end

%==========================================================================
% SUBSAMPLE INPUT POINT CLOUD
%==========================================================================

performSubsampling = (0 < numSamples) && (numSamples < numPoints);
if performSubsampling
    
    if verbose, fprintf('Subsampling point cloud:\n'); end
    
    % Make a copy of the input cloud. This will become the list of
    % 'non-sample' points that may be re-introduced later
    XNS = X;
    
    numPoints0 = numPoints; % The number of raw input points
    
    if strcmpi(ssMethod, 'SPARTAN')
        
        % Run SPARTAN subsampling
        [X, XSSIDx] = SPARTAN( X, numSamples, ...
            'OMTIterations', omtIter, 'OMTWassersteinThresh', ...
            omtWasserThresh, 'QRPType', qrpType, 'QRPLeap', qrpLeap, ...
            'QRPSkip', qrpSkip, 'QRPScramble', qrpScramble, ...
            'Verbose', verbose );
        
    elseif strcmpi(ssMethod, 'Random')
        
        XSSIDx = randsample(numPoints, numSamples);
        X = X(XSSIDx, :);
        
    else
        
        error('Invalid subsampling method');
        
    end
    
    % The 'non-samples' removed from the point cloud
    XNSIDx = find(~ismember((1:numPoints0).', XSSIDx));
    XNS = XNS(XNSIDx, :);
    
    numPoints = numSamples; % The number of points in subsampled cloud
    numNonSamples = size(XNS,1); % The number of non-samples
    
    if verbose, fprintf('Done\n'); end
    
end

%==========================================================================
% CONSTRUCT NEIGHBORHOOD GRAPH
%==========================================================================

if verbose, fprintf('Constructing neighborhood graph... '); end

try
    
    ENG = proximityGraphsNGL(X, nGraphType, kNN, betaParam, ...
        forceConnectivity, graphOutputFormat);
    
catch
    
    fprintf('\n');
    warning(['Construction of proximity graphs using NGL failed. ' ...
        'Is the MEX code properly installed?']);
    
    if (kNN <= 0)
        
        ENG = proximityGraphs(X, nGraphType, graphOutputFormat, verbose);
        
    else
        
        ENG = proximityGraphsKNN(X, nGraphType, kNN, ...
            forceConnectivity, verbose);
        
    end
    
end

if verbose, fprintf('Done\n'); end

%==========================================================================
% KERNEL DENSITY ESTIMATION
%==========================================================================
% This section is kind of silly and includes a lot of redundant
% calculations. However, it's hard to think of how to efficiently perform
% these calculations for large point clouds. This is probably one of the
% sections with the most room for improvement.

% Estimate neighborhood size if necessary
if (kdsigma <= 0)
    [~, nnDists] = knnsearch(X, X, 'K', 2, 'SortIndices', true);
    kdsigma = sqrt((mean(nnDists(:,2))+2*std(nnDists(:,2)))/4);
    clear nnDists
end

% Set up the output identity vector
if performSubsampling
    XOutIDx = XSSIDx;
else
    XOutIDx = (1:numPoints).';
end

% Set up a parallel data queue to handle real-time progres outout
parDQ = parallel.pool.DataQueue;
afterEach(parDQ, @updateParallelProgressBar);

%--------------------------------------------------------------------------
% Calculate Point Densities On All (Subsampled) Points
%--------------------------------------------------------------------------

if verbose, fprintf('Initial density estimation... \n'); end
    
density = zeros(numPoints, 1);
updateParallelProgressBar(1, numPoints);
parfor i = 1:numPoints

    % Squared distance between the current point and all points in the
    % point set (includes self distance == 0)
    dij2 = sum((X - repmat(X(i,:), numPoints, 1)).^2, 2);
    
    density(i) = sum(exp(-dij2 / (2 * kdsigma^2)));
    
    % if verbose, progressbar(i, numPoints); end
    if verbose, send(parDQ, []); end
    
end

density = density ./ (numPoints * (kdsigma * sqrt(2*pi))^dim);

if verbose, fprintf('Done\n'); end

%--------------------------------------------------------------------------
% Upsample Points on Graph Edges
%--------------------------------------------------------------------------

if upsampleEdges
   
    if verbose, fprintf('Upsampling points on graph edges...\n'); end
    
    % Perform Abbreviated Calculation of Density on Edge Midpoints --------
    
    % Coordinates of edge midpoints
    EMP = (X(ENG(:,2), :) + X(ENG(:,1), :)) ./ 2;
    numMP = size(EMP,1);
    
    mpDensity = zeros(size(EMP,1), 1);
    updateParallelProgressBar(1, numMP);
    parfor i = 1:numMP

        % Squared distance between the current point and all points in the
        % point set (includes self distance == 0)
        dij2 = sum(([X; EMP(i,:)] - repmat(EMP(i,:), numPoints+1, 1)).^2, 2);
        
        mpDensity(i) = sum(exp(-dij2 / (2 * kdsigma^2)));
        
        % if verbose, progressbar(i, numMP); end
        if verbose, send(parDQ, []); end
        
    end
    
    mpDensity = mpDensity ./ ((numPoints+1) * (kdsigma * sqrt(2*pi))^dim);
    
    % Determine Midpoint Inclusion ----------------------------------------
    
    % Only keep midpoints with lower density than both attached vertices
    incMPIDx = (mpDensity < density(ENG(:,1))) & (mpDensity < density(ENG(:,2)));
    numIncMP = sum(incMPIDx);
    
    % Augment graph edge list with new edges
    newEdges = ENG(incMPIDx, :);
    newEdges = [newEdges(:,1), ...
        repmat((1:numIncMP).'+numPoints, 1, 2), newEdges(:,2)];
    newEdges = [ newEdges(:, [1 2]); newEdges(:, [3 4]) ];
    
    ENG(incMPIDx, :) = [];
    ENG = [ENG; newEdges];
    
    % Augment point list with new points
    X = [X; EMP(incMPIDx, :)];
    XOutIDx = [XOutIDx; nan(numIncMP, 1)];
    numPoints = size(X,1);
    
    if verbose, fprintf('Done\n'); end
    
end

%--------------------------------------------------------------------------
% Include Non-Samples
%--------------------------------------------------------------------------

if (performSubsampling && includeNonSamples)
    
    if verbose, fprintf('Re-introducing non-sample points... '); end
    
    % Determine the nearest neighbor in the sample set for each point in
    % the nonsample set
    nnIDx = knnsearch(X, XNS, 'K', 2, 'SortIndices', true);
    nnIDx = nnIDx(:,2); % Remove trivial self-edge
    
    % Augment point list
    X = [X; XNS];
    XOutIDx = [XOutIDx; XNSIDx];

    % Augment edge list 
    ENG = [ENG; nnIDx, (numPoints+(1:numNonSamples)).'];
    
    numPoints = size(X,1); % Update number of combined points
    
    if verbose, fprintf('Done\n'); end
    
end

%--------------------------------------------------------------------------
% Perform Final Density Estimation on Upsampled Point Set
%--------------------------------------------------------------------------

if ( (performSubsampling && includeNonSamples) || upsampleEdges )
    
    if verbose, fprintf('Final density estimation... \n'); end
    
    density = zeros(numPoints, 1);
    updateParallelProgressBar(1, numPoints);
    parfor i = 1:numPoints

        % Squared distance between the current point and all points in the
        % point set (includes self distance == 0)
        dij2 = sum((X - repmat(X(i,:), numPoints, 1)).^2, 2);
        
        density(i) = sum(exp(-dij2 / (2 * kdsigma^2)));
        
        % if verbose, progressbar(i, numPoints); end
        if verbose, send(parDQ, []); end
        
    end
    
    density = density ./ (numPoints * (kdsigma * sqrt(2*pi))^dim);
    
    
    if verbose, fprintf('Done\n'); end
 
end

% Sort points by density (low to high) and update edge connectivity lists
[density, sortIDx] = sort(density);
X = X(sortIDx, :);
XOutIDx = XOutIDx(sortIDx);
ENG = changem(ENG, (1:numPoints).', sortIDx);

%==========================================================================
% CONSTRUCT INITIAL JOIN TREE
%==========================================================================

if verbose, fprintf('Constructing initial join tree... '); end

EJT = nan(numPoints-1, 2);

% The representative for each component, i.e. C(i) returns the ID of the
% root of the component containing vertex i
C = (1:numPoints).';

% lowestVertex(i) returns the ID of the vertex in C(i) with the lowest
% height value
lowestVertex = C; 

% branchIDx(i) returns the branch that contains the ith vertex in the
% branch decomposition of the join tree
branchIDx = C;

curEdge = 1; % A counter for edge insertion into the connectivity list

for i = numPoints:-1:1
    
    % Find all neighbors of the current vertex
    nIDx = unique(ENG(any(ENG == i, 2), :));
    nIDx(nIDx == i) = [];
    
    for nID = 1:numel(nIDx)
        
        j = nIDx(nID); % The vertex ID of the current neighbor
        
        % Skip if j has a lower height value than i or if j and i already
        % share the same component
        if ((j < i) || (C(i) == C(j))), continue; end
        
        % Merge the components containing the two vertices ensuring that
        % the HIGHEST branch ID persists
        if (C(j) > C(i))
            C(C == C(i)) = C(j);
        else
            C(C == C(j)) = C(i);
        end
        
        % Add an edge to the join tree
        EJT(curEdge, :) = [lowestVertex(j), i];
        
        % Update the lowest vertex of the combined components
        lowestVertex(C == C(j)) = i;
        
        % Update the branch decomposition
        % Saddles will ALWAYS belong to the branch with the highest root
        % if (density(C(j)) > density(branchIDx(i)))
        %   branchIDx(i) = branchIDx(EJT(curEdge, 1));
        % end
        if (branchIDx(EJT(curEdge, 1)) > branchIDx(i))
            branchIDx(i) = ...
                branchIDx(EJT(curEdge, 1));
        end
        
        % Update the edge counter
        curEdge = curEdge + 1;
        
    end
    
end

goodJoinTree = (curEdge == numPoints) && ~any(isnan(EJT(:))) && ...
    isequal(sort(EJT(:,1)), (2:numPoints).') && ...
    ismember(1, find(branchIDx == numPoints));
assert(goodJoinTree, 'Initial join tree construction failed');

saddleIDx = [1, find(histcounts(EJT(:,2), 0.5:1:(numPoints+0.5)) > 1)];
[~, goodBranches] = ismember(unique(branchIDx), branchIDx);
goodBranches(goodBranches == 1) = [];
goodBranches = EJT(ismember(EJT(:,1), goodBranches), 2);
goodBranches = all(ismember(goodBranches, saddleIDx));
assert(goodBranches, 'Some branches do not terminate in saddles');

if verbose, fprintf('Done\n'); end

%==========================================================================
% PERFORM TOPOLGICAL SIMPLIFICATION
%==========================================================================

%--------------------------------------------------------------------------
% Calculate Measures of Branch Quality
%--------------------------------------------------------------------------

if verbose, fprintf('Calculating branch quality metrics... '); end

% A sorted list (low to high) of the unique roots for each branch
branchID = unique(branchIDx);
numBranches = numel(branchID);

% Calculate branch persistence and stability
branchStability = zeros(numBranches, 1);
maxDensity = zeros(numBranches, 1);
minDensity = zeros(numBranches, 1);
for i = 1:numBranches

    curBranchDensity = density(branchIDx == branchID(i));

    maxDensity(i) = max(curBranchDensity);
    minDensity(i) = min(curBranchDensity);

    branchStability(i) = sum(curBranchDensity);

end

branchPersistence = maxDensity - minDensity;

% Calculate branch size
branchSize = histcounts(changem(branchIDx, 1:numBranches, branchID), ...
    0.5:1:(numBranches+0.5)).';

% Determine which branches will be pruned
rmIDx = branchPersistence < (persistenceThreshold * max(density));
rmIDx = rmIDx | (branchSize < sizeThreshold);
rmIDx = rmIDx | (branchStability < stabilityThreshold);
rmIDx = rmIDx | (maxDensity < maxThreshold);

%--------------------------------------------------------------------------
% Prune Branches Based on Spatial Proximity
%--------------------------------------------------------------------------

if (numel(branchIDx(~rmIDx)) > 1)

    if strcmpi(collisionMergeMethod, 'sequential')

        activeBranchIDx = branchID(~rmIDx);
        [nnIDx, nnDists] = knnsearch( X(activeBranchIDx, :), ...
            X(activeBranchIDx, :), 'k', 2 );
        nnDists = nnDists(:,2);

        while any(nnDists < distThreshold)

            % Determine the branch IDs of the current closest pair within
            % the distance threshold
            minDistPairIDx = [0 0];
            [~, minDistPairIDx(1)] = min(nnDists);
            minDistPairIDx(2) = nnIDx(minDistPairIDx(1));
            minDistPairIDx = activeBranchIDx(minDistPairIDx);

            % Keep only the element of the pair with the maximum density
            rmIDx(branchID == min(minDistPairIDx)) = true;

            % Update the nearest neighbor distance list
            activeBranchIDx = branchID(~rmIDx);
            [nnIDx, nnDists] = knnsearch( X(activeBranchIDx, :), ...
                X(activeBranchIDx, :), 'k', 2 );
            nnDists = nnDists(:,2);

        end

        % TODO: Incorporate collision zones in to quality visualization ***
        trueBranchCollisions = [];

    elseif strcmpi(collisionMergeMethod, 'simultaneous')

        % Find the number of r-neighborhood collisions, i.e all density
        % maxima that are closer than the distance threshold to the current
        % maximum. NOTE: Includes the self-collision
        allBranchCollisions = ...
            rangesearch(X(branchID, :), X(branchID, :), distThreshold);
        allBranchCollisions = cellfun(@(x) branchID(x.'), ...
            allBranchCollisions, 'Uni', false);

        % Remove cells correspoding to branches with no collisions (i.e.
        % the cell contains only the self-collision)
        allBranchCollisions(cellfun(@(x) numel(x) == 1, ...
            allBranchCollisions, 'Uni', true)) = [];

        if ~isempty(allBranchCollisions)

            % Determine which collision neighborhoods intersect
            numCollisions = numel(allBranchCollisions);
            collisionEdges = nan(numCollisions * (numCollisions-1) / 2, 2);
            count = 1;
            for i = 1:numCollisions
                for j = (i+1):numCollisions
                    if ~isempty(intersect(allBranchCollisions{i}, ...
                            allBranchCollisions{j}))
                        collisionEdges(count, :) = [i j];
                    end
                    count = count+1;
                end
            end
            assert(count == (size(collisionEdges,1)+1), ...
                'Bad collision pair counting');
            collisionEdges(any(isnan(collisionEdges), 2), :) = [];

            % Establish connected components over intersecting regions
            [numIntRegions, intRegionIDx] = ...
                graphConnectedComponents(collisionEdges, numCollisions);

            % Merge connected components of intersecting regions
            trueBranchCollisions = cell(numIntRegions, 1);
            rmCollIDx = false(numBranches, 1);
            for i = 1:numIntRegions

                % Determine all unique points in the current merged
                % collision zone
                curCollisions = allBranchCollisions(intRegionIDx == i);
                curCollisions = unique(vertcat(curCollisions{:}));
                trueBranchCollisions{i} = curCollisions;

                % Remove all points from consideration that have already
                % been removed according to the previous topological
                % quality measures
                curCollisions(ismember(curCollisions, branchID(rmIDx))) = [];

                % Tag all points except the highest remaining maximium to
                % be removed
                if ~isempty(curCollisions)
                    curCollisions(curCollisions == max(curCollisions)) = [];
                    rmCollIDx(ismember(branchID, curCollisions)) = true;
                end

            end

            rmIDx = rmIDx | rmCollIDx;

        else

            % TODO: Incorporate collision zones in to quality visualization
            trueBranchCollisions = [];

        end

    else

        error('Invalid collision merge method');

    end

else

    % TODO: Incorporate collision zones in to quality visualization *******
    trueBranchCollisions = [];

end

%--------------------------------------------------------------------------
% Plot Branch Quality
%--------------------------------------------------------------------------

if plotBranchQuality
    
    % Find the nearest (Euclidean) nearest neighbor distance for each maximum
    % (Just used for plotting)
    [~, branchNNDists] = knnsearch(X(branchID, :), X(branchID, :), 'k', 2);
    branchNNDists = branchNNDists(:,2);
    
    % Re-format the branch collision IDs into local indices into the branch
    % list
    for i = 1:numel(trueBranchCollisions)
        [~, trueBranchCollisions{i}] = ismember( ...
            trueBranchCollisions{i}, branchID );
    end
    
    PlotBranchQuality( rmIDx, minDensity, maxDensity, ...
    branchSize, branchStability, branchNNDists, persistenceThreshold, ...
    sizeThreshold, stabilityThreshold, distThreshold, maxThreshold, ...
    trueBranchCollisions );

end

if verbose, fprintf('Done\n'); end

%--------------------------------------------------------------------------
% Merge Low Quality Branches
%--------------------------------------------------------------------------

if verbose, fprintf('Pruning low quality branches... '); end

% Re-format the removal index to be the vertex ID of the root of each
% branch
rmIDx = branchID(rmIDx); 

for i = numel(rmIDx):-1:1
    
    % The node ID of the root of the current branch being removed
    rmBranchID = rmIDx(i);
    
    % The IDs of all nodes belonging to this branch
    rmBranchNodeIDx = find(branchIDx == rmBranchID);
    
    % The node ID of the saddle into which the current branch terminates
    saddleID = EJT(EJT(:,1) == rmBranchNodeIDx(1), 2);
    assert(numel(saddleID) == 1, ...
        'No/Multiple saddles found during pruning');
    
    % Find the branch sharing this saddle with the highest root
    mergeBranchID = branchIDx(EJT(EJT(:,2) == saddleID, 1));
    mergeBranchID = max(mergeBranchID(mergeBranchID ~= rmBranchID));
    assert(branchIDx(saddleID) == mergeBranchID, ...
        'Invalid branch assignment for saddle');
    
    % The nodes belonging to the merge branch up and including
    % the saddle point
    mergeBranchNodeIDx = find(branchIDx == mergeBranchID);
    mergeBranchNodeIDx(mergeBranchNodeIDx < saddleID) = [];
    
    % Update branch assignments
    branchIDx(rmBranchNodeIDx) = mergeBranchID;
    
    % Generate the new edges for the merged branches
    newEdges = sort([mergeBranchNodeIDx; rmBranchNodeIDx], 'descend');
    newEdges = [newEdges(1:(end-1)), newEdges(2:end)];
    
    % Replace the old edges in the connectivity list
    rmEIDx = ismember(EJT(:,1), rmBranchNodeIDx) | ...
        ismember(EJT(:,1), mergeBranchNodeIDx(2:end));
    EJT(rmEIDx, :) = newEdges;
    
end

goodJoinTree = (curEdge == numPoints) && ~any(isnan(EJT(:))) && ...
    isequal(sort(EJT(:,1)), (2:numPoints).') && ...
    ismember(1, find(branchIDx == numPoints));
assert(goodJoinTree, 'Join tree pruning failed');

saddleIDx = [1, find(histcounts(EJT(:,2), 0.5:1:(numPoints+0.5)) > 1)];
[~, goodBranches] = ismember(unique(branchIDx), branchIDx);
goodBranches(goodBranches == 1) = [];
goodBranches = EJT(ismember(EJT(:,1), goodBranches), 2);
goodBranches = all(ismember(goodBranches, saddleIDx));
assert(goodBranches, 'Some pruned branches do not terminate in saddles');

if verbose, fprintf('Done\n'); end

%==========================================================================
% Format Output
%==========================================================================

% The final processed point cloud coordinate list
XOut = X; 

% The IDs of the nodes corresponding to maxima in the processed point cloud
maxIDx = find(~ismember((1:numPoints).', EJT(:,2)));

% The IDs if the nodes corresponding to saddles in the processed point
% cloud
saddleIDx = find(histcounts(EJT(:,2), 0.5:1:(numPoints+0.5)) > 1).';

assert(~any(ismember(saddleIDx, maxIDx)), 'Roots and saddles intersect');

end

function PlotBranchQuality( rmIDx, minDensity, maxDensity, ...
    branchSize, branchStability, branchNNDists, persistenceThreshold, ...
    sizeThreshold, stabilityThreshold, distThreshold, maxThreshold, ...
    trueBranchCollisions )
% PLOTBRANCHQUALITY A quick and dirty visualization of the various branch
% quality measures

% The maximum density of all vertices
trueDensityMax = max(maxDensity);

numBranches = numel(minDensity);
branchColors = repmat([0 0 1], numBranches, 1);
branchColors(rmIDx, :) = repmat([1 0 0], sum(rmIDx), 1);

locBranchID = 1:numBranches;

subplot(2,2,1);
hold on
plot( trueDensityMax*[0, 1.1], trueDensityMax*[0, 1.1], '-k', ...
    'LineWidth', 2 );
plot( trueDensityMax*[0, 1.1], ...
    trueDensityMax * ([0, 1.1] + persistenceThreshold), ...
    '--k', 'LineWidth', 2 );
plot( trueDensityMax*[0 1.1], maxThreshold * [1 1], '--k', 'LineWidth', 2 );
scatter( minDensity, maxDensity, 100, branchColors, 'filled', ...
    'LineWidth', 3 );
hold off
axis equal
xlim(trueDensityMax * [0, 1.1]);
ylim(trueDensityMax * [0, 1.1]);
title('Branch Persistence');
xlabel('Minimum Heights (Saddles)');
ylabel('Maximum Heights');

subplot(2,2,2);
hold on
if any(rmIDx)
    stem(locBranchID(rmIDx), branchSize(rmIDx), 'filled', 'r');
end
stem(locBranchID(~rmIDx), branchSize(~rmIDx), 'filled', 'b');
plot([1 numBranches], sizeThreshold * [1 1], '--k', 'LineWidth', 2);
hold off
title('Branch Size');
ylabel('Number of Vertices in Branch');
xlabel('Branch ID');

subplot(2,2,3);
hold on
if any(rmIDx)
    stem(locBranchID(rmIDx), branchStability(rmIDx), 'filled', 'r');
end
stem(locBranchID(~rmIDx), branchStability(~rmIDx), 'filled', 'b');
plot([1 numBranches], stabilityThreshold * [1 1], '--k', 'LineWidth', 2);
hold off
title('Branch Stability');
ylabel('Sum of Densities in Branch');
xlabel('Branch ID');

subplot(2,2,4);

hold on

if isempty(trueBranchCollisions)
    
    if any(rmIDx)
        stem(locBranchID(rmIDx), branchNNDists(rmIDx), 'filled', 'r');
    end
    stem(locBranchID(~rmIDx), branchNNDists(~rmIDx), 'filled', 'b');
    
else
    
    collisionColors = parula(numel(trueBranchCollisions));
    for i = 1:numel(trueBranchCollisions)
        stem( locBranchID(trueBranchCollisions{i}), ...
            branchNNDists(trueBranchCollisions{i}), 'filled', ...
            'MarkerFaceColor', collisionColors(i,:), ...
            'MarkerEdgeColor', collisionColors(i,:), ...
            'Color', collisionColors(i,:) );
    end
    
    noCollisionIDx = ~ismember(1:numel(rmIDx), ...
        vertcat(trueBranchCollisions{:}));
    stem(locBranchID(noCollisionIDx), branchNNDists(noCollisionIDx), ...
        'filled', 'k');
      
end


plot([1 numBranches], distThreshold * [1 1], '--k', 'LineWidth', 2);
hold off
title('Branch Spatial Proximity');
ylabel('Branch Nearest Neighbor Distance');
xlabel('Branch ID');


end

function updateParallelProgressBar(curIter, maxIter)
% Helper function to update the progressbar in parallel loops
    
    persistent parCurIter parMaxIter
    
    if (nargin == 2)
        
        parCurIter = curIter;
        parMaxIter = maxIter;
        
    else
        
        progressbar(parCurIter, parMaxIter)
        parCurIter = parCurIter + 1;
        
    end
    
end