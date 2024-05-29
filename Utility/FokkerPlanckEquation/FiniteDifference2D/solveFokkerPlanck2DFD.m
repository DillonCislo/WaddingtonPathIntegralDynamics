function [FPSol, FPO] = solveFokkerPlanck2DFD(X, Y, P0, varargin)
%SOLVEFOKKERPLANCK2DFD Solves various conjugations of the Fokker-Planck
%equation on a fixed 2D rectangular domain using the numerical method of
%lines with a simple finite differencing spatial discretization scheme. For
%a uniform diffusion constant D and a given potential field U(x,y) the
%solver can handle three different formulations of the FPE:
%
%   (1) Forward FPE:
%       dp/dt = D (del p) + (del U) p + (grad U) * (grad p)
%
%   (2) Backward FPE:
%       -dp/ds = D (del p) - (grad U) * (grad p)
%
%   (3) Hermitian Schrodinger:
%       dp/dt = D (del p) + ( (del U)/2 - (grad U).^2/(4 D) )
%
%where for the backwards equation we integrate backwards in time from s>t0
%to t0. Formulation (3) can be obtained from Formulation (1) by the
%substitution p = q e^{-U/(2D)). When U(x,y) = 0, these reduce to the
%forward/backward diffusion equations. The solver is capable of handling
%mixed Dirichlet/Neumann boundary conditions. By default, all vertices
%whose boundary conditions are not explicitly specified by the input
%variables are assigned zero flux boundary conditions, i.e.
%
%   (1) Forward/Backward:
%           n * ( D (grad p) + (grad U) p ) = 0
%
%   (2) Hermitian Schrodinger:
%           n * ( D (grad p) + (grad U/2) p ) = 0
%
%
%   INPUT PARAMETERS:
%
%       - X:    #Nx#N grid of x-coordinates for evaluation points. Values
%               should conform to the output format of the 'meshgrid' function
%
%       - Y:    #Nx#N grid of y-coordinates for evaluation points. Values
%               should conform to the output format of the 'meshgrid' function
%
%       - P0:   #Nx#N grid of initial density values
%
%   OPTIONAL INPUT PARAMETERS:
%
%       - ('DiffusionCoefficient' or 'D', D = 1): The uniform diffusion
%       coefficent
%
%       - ('Potential' or 'U', U = 0): The spatially dependent potential
%       field governing the drift dynamics. The potential field can be
%       supplied either as
%
%               (1) An #Nx#N matrix of nodal values
%               (2) An anonymous function, i.e. U = @(x,y) ...
%               (3) A symbolic scalar function terms of two symbolic
%               variables 'x' and 'y'
%
%       - ('TimeSpan', timeSpan = [0 1]): The times over which the equation
%       will be integrated. 
%
%       - ('IntegratorType', odeIntType = 'stiff'): Uses the 'ode15s'
%       integrator for stiff problems. If you're quite sure your system is
%       non-stiff setting odeIntType = 'non-stiff' will use the 'ode45'
%       explicit Runge-Kutta solver
%
%       - ('ODEOptions', odeOptions = odeset): The options fed to the ODE
%       integrator. See documentation for 'odeset' for more details.
%
%       - ('FokkerPlanckType', FPType = 'forward'): The formulation of the
%       FPE which is to be solved
%
%       - ('DirichletBoundaryConditions', dirBC = []): A #DBx2 list with
%       boundary vertex IDs in the first column and fixed density values
%       for those vertices in the second column, i.e. P(dirBC(i,1)) =
%       dirBC(i,2)
%
%       - ('NeumannBoundaryConditions', neuBC = []): A #NBx2 list with
%       boundary vertex IDs in the first column and normal density gradient
%       values in the second column i.e., n * (grad P(neuBC(i,1)) =
%       neuBC(i,2)
%
%       - ('Display', odeDisplay = 'off'): Set to 'iter' to display verbose
%       progress output
%
%   OUTPUT PARAMETERS:
%
%       - FPSol:    Solution output structure that can be evaluated for
%                   any time on the interval given by 'timeSpan'
%
%       - FPO:      #Nx#N Linear Fokker-Planck Operator
%
%   by Dillon Cislo 01/13/2022

%--------------------------------------------------------------------------
% Input Processing
%--------------------------------------------------------------------------
if (nargin < 1), error('Please supply evaluation point x-coordinates'); end
if (nargin < 2), error('Please supply evaluation point y-coordinates'), end
if (nargin < 3), error('Please supply initial values'); end

validateattributes(X, {'numeric'}, {'2d', 'finite', 'real'});
validateattributes(Y, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', size(X,1), 'ncols', size(X,2)});
validateattributes(P0, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', size(X,1), 'ncols', size(X,2)});

assert((unique(diff(X,1,1)) == 0) && (std(unique(diff(X,1,2))) < 1e-10), ...
    'Invalid x-coordinate format. See ''meshgrid'' documentation');
assert((unique(diff(Y,1,2)) == 0) && (std(unique(diff(Y,1,1))) < 1e-10), ...
    'Invalid y-coordinate format. See ''meshgrid'' documentation');

dx = mean(unique(diff(X,1,2))); % Grid spacing in x
dy = mean(unique(diff(Y,1,1))); % Grid spacing in y

% A set of node IDs for each grid point
gridIDx = reshape((1:numel(X)).', size(X));

% A list of boundary node IDs
bdyIDx = false(size(X));
bdyIDx(1,:) = true; bdyIDx(end,:) = true;
bdyIDx(:,1) = true; bdyIDx(:,end) = true;
bdyIDx = find(bdyIDx(:));

% Set Optional Parameters -------------------------------------------------

D = 1;
U = zeros(size(X));
timeSpan = [0 1];
odeIntType = 'stiff';
odeOptions = odeset;
FPType = 'forward';
dirBC = [];
neuBC = [];
odeDisplay = 'off';
dt = 0.1;

for i = 1:numel(varargin)
    
    if isa(varargin, 'double'), continue; end
    if isa(varargin, 'logical'), continue; end
    
    if any(strcmpi(varargin{i}, {'DiffusionCoefficient', 'D'}))
        D = varargin{i+1};
        validateattributes(D, {'numeric'}, ...
            {'scalar', 'finite', 'real', 'positive'});
    end
    
    if any(strcmpi(varargin{i}, {'Potential', 'U'}))
        U = varargin{i+1}; % Validation handled later
    end
    
    if strcmpi(varargin{i}, 'TimeSpan')
        timeSpan = varargin{i+1};
        validateattributes(timeSpan, {'numeric'}, ...
            {'vector', 'numel', 2, 'finite', 'real'});
        if (size(timeSpan,2) ~= 2), timeSpan = timeSpan.'; end
    end
    
    if strcmpi(varargin{i}, 'IntegratorType')
        odeIntType = lower(varargin{i+1});
        validateattributes(odeIntType, {'char'}, {'vector'});
        assert(ismember(odeIntType, {'stiff', 'non-stiff'}), ...
            'Invalid integrator type');
    end
    
    if strcmpi(varargin{i}, 'ODEOptions')
        odeOptions = varargin{i+1}; % Validation handled by integrator
    end
    
    if strcmpi(varargin{i}, 'FokkerPlanckType')
        FPType = lower(varargin{i+1});
        validateattributes(FPType, {'char'}, {'vector'});
        assert(ismember(FPType, {'forward', 'backward', 'hermitian'}), ...
            'Invalid Fokker-Planck formulation');
    end
    
    if strcmpi(varargin{i}, 'DirichletBoundaryConditions')
        dirBC = varargin{i+1};
        if isempty(dirBC), continue; end
        validateattributes(dirBC, {'numeric'}, ...
            {'2d', 'finite', 'real', 'ncols', 2});
    end
    
    if strcmpi(varargin{i}, 'NeumannBoundaryConditions')
        error('General Neumann boundary conditions are not yet implemented');
        neuBC = varargin{i+1};
        validateattributes(neuBC, {'numeric'}, ...
            {'2d', 'finite', 'real', 'ncols', 2});
    end

    if strcmpi(varargin{i}, 'Display')
        odeDisplay = varargin{i+1};
        validateattributes(odeDisplay, {'char'}, {'vector'});
    end
    
end

if strcmpi(odeDisplay, 'iter')
    odeOptions.OutputFunction = @odeprog;
end

%--------------------------------------------------------------------------
% Process Potential Field Input
%--------------------------------------------------------------------------

if isa(U, 'sym')
    
    syms x y
    assume(x, 'real'); assume(y, 'real');
    
    assert(all(ismember(symvar(U), {'x', 'y'})), ...
        'Invalid symbolic potential input');
    
    gradU = matlabFunction(gradient(U, [x,y]).', 'Vars', {x,y});  
    delUX = matlabFunction(gradient(gradient(U, x), x), 'Vars', {x,y});
    delUY = matlabFunction(gradient(gradient(U, y), y), 'Vars', {x,y});
    delU = matlabFunction(laplacian(U, [x,y]), 'Vars', {x,y});
    U = matlabFunction(U, 'Vars', {x,y});
    
    U = reshape(U(X(:), Y(:)), size(X));
    validateattributes(U, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', size(X,1), 'ncols', size(X,2)});
    
    gradU = gradU(X(:), Y(:));
    gradUX = reshape(gradU(:,1), size(X));
    gradUY = reshape(gradU(:,2), size(X));
    gradU = gradU(:);
    
    delU = delU(X(:), Y(:));
    
    delUX = delUX(X(:), Y(:));
    if isscalar(delUX), delUX = delUX * ones(size(X));
    else, delUX = reshape(delUX, size(X)); end
    
    delUY = delUY(X(:), Y(:));
    if isscalar(delUY), delUY = delUY * ones(size(X));
    else, delUY = reshape(delUY, size(X)); end

elseif (isa(U, 'function_handle') || ismatrix(U))
    
    if isa(U, 'function_handle')
        U = reshape(U(X(:), Y(:)), size(X));
    end
    
    validateattributes(U, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', size(X,1), 'ncols', size(X,2)});
    
    % Calculate numerical gradient
    [gradUX, gradUY] = gradient(U, dx, dy);
    gradU = [gradUX(:); gradUY(:)];
    
    % Calculate numerical second derivative along x
    delUX = zeros(size(U));
    delUX(:, 2:(end-1)) = (diff(U(:,2:end),1,2)-diff(U(:,1:(end-1)),1,2)) / dx^2;
    delUX(:,1) = 2 * delUX(:,2) - delUX(:,3);
    delUX(:,end) = -delUX(:,end-2) + 2 * delUX(:,end-1);
    
    % Calculate numerical second derivative along y
    delUY = zeros(size(U));
    delUY(2:(end-1), :) = (diff(U(2:end,:),1,1)-diff(U(1:(end-1),:),1,1)) / dy^2;
    delUY(1,:) = 2 * delUY(2,:) - delUY(3,:);
    delUY(end,:) = -delUY(end-2,:) + 2 * delUY(end-1,:);
    
    % Combine to form numerical Laplacian
    delU = delUX(:) + delUY(:);
    
else
    
    error('Invalid potential field input');
    
end

validateattributes(gradU, {'numeric'}, {'vector', ...
    'finite', 'real', 'numel', 2 * numel(X)});
validateattributes(gradUX, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', size(X,1), 'ncols', size(X,2)});
validateattributes(gradUY, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', size(X,1), 'ncols', size(X,2)});
validateattributes(delU, {'numeric'}, {'vector', ...
    'finite', 'real', 'numel', numel(X)});
validateattributes(delUX, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', size(X,1), 'ncols', size(X,2)});
validateattributes(delUY, {'numeric'}, {'2d', 'finite', 'real', ...
    'nrows', size(X,1), 'ncols', size(X,2)});

%--------------------------------------------------------------------------
% Formulate Derivative Operators on Finite Difference Grid
%--------------------------------------------------------------------------
% NOTE: These are not the typical finite difference operators!!!
% They are formulated specifically so that the final Fokker-Planck
% operators will automatically satisfy all no-flux/user specified Neumann
% boundary conditions.

if strcmpi(FPType, 'hermitian')
    BCGradUX = gradUX/2; BCGradUY = gradUY/2;
else
    BCGradUX = gradUX; BCGradUY = gradUY;
end

% Second Derivative Operator Along X --------------------------------------

% Central difference formula for interior vertices
I = gridIDx; I(:,1) = []; I(:,end) = [];
J = [I; I+size(X,1); I-size(X,1)];
V = [-2*ones(size(I)); ones(size(I)); ones(size(I))] ./ dx^2;
I = [I; I; I];

% Modified central difference formula left boundary vertices
I = [I(:); repmat(gridIDx(:,1), 2, 1)];
J = [J(:); gridIDx(:,1); gridIDx(:,2)];
V = [V(:); -ones(size(X,1),1)/dx^2; ones(size(X,1),1)/dx^2];

% Modified central difference formula for right boundary vertices
I = [I; repmat(gridIDx(:,end), 2, 1)];
J = [J; gridIDx(:,end); gridIDx(:,end-1)];
V = [V; -ones(size(X,1),1)/dx^2; ones(size(X,1),1)/dx^2];

Lx = sparse(I, J, V, numel(X), numel(X));

% Second Derivative Operator Along Y --------------------------------------

% Central difference formula for interior vertices
I = gridIDx; I(1,:) = []; I(end,:) = [];
J = [I; I+1; I-1];
V = [-2*ones(size(I)); ones(size(I)); ones(size(I))] / dy^2;
I = [I; I; I];

% Modified central difference formula for top boundary vertices
I = [I(:); repmat(gridIDx(1,:).', 2, 1)];
J = [J(:); gridIDx(1,:).'; gridIDx(2,:).'];
V = [V(:); -ones(size(X,2),1)/dy^2; ones(size(X,2),1)/dy^2];

% Modified central difference formula for bottom boundary vertices
I = [I; repmat(gridIDx(end,:).', 2, 1)];
J = [J; gridIDx(end,:).'; gridIDx(end-1,:).'];
V = [V; -ones(size(X,2),1)/dy^2; ones(size(X,2),1)/dy^2];

Ly = sparse(I, J, V, numel(X), numel(X));

% Assemble Forward Divergence Operator In X -------------------------------
% Equivalent to d/dx * ( (dUdx) p ) = (d^2Udx^2)p + (dUdx) * (dpdx)

% Central difference formula for interior values
I = gridIDx; I(:,1) = []; I(:,end) = [];
J = [I + size(X,1); I - size(X,1)];
V = [gradUX(:,3:end); -gradUX(:,1:(end-2))] ./ (2 * dx);
I = [I; I];

% Modified central difference formula for left boundary vertices
I = [I(:); repmat(gridIDx(:,1), 2, 1)];
J = [J(:); gridIDx(:,1); gridIDx(:,2)];
V = [V(:); gradUX(:,1) ./ (2 * dx); gradUX(:,2) ./ (2 * dx)];

% Modified central difference formula for right boundary vertices
I = [I; repmat(gridIDx(:,end), 2, 1)];
J = [J; gridIDx(:,end); gridIDx(:,end-1)];
V = [V; -gradUX(:,end) ./ (2 * dx); -gradUX(:,end-1) ./ (2 * dx)];

FDX = sparse(I, J, V, numel(X), numel(X));

% Assemble Forward Divergence Operator in Y -------------------------------
% Equivalent to d/dy * ( (dUdy) p ) = (d^2Udy^2)p + (dUdy) * (dpdy)

% Central difference formula for interior values
I = gridIDx; I(1,:) = []; I(end,:) = [];
J = [I + 1; I - 1];
V = [gradUY(3:end, :); -gradUY(1:(end-2),:)] ./ (2 * dy);
I = [I; I];

% Modified central difference formula for top boundary vertices
I = [I(:); repmat(gridIDx(1,:).', 2, 1)];
J = [J(:); gridIDx(1,:).'; gridIDx(2,:).'];
V = [V(:); gradUY(1,:).' ./ (2 * dy); gradUY(2,:).' ./ (2 * dy) ];

% Modified central difference formula for bottom boundary vertices
I = [I; repmat(gridIDx(end,:).', 2, 1)];
J = [J; gridIDx(end,:).'; gridIDx(end-1,:).'];
V = [V; -gradUY(end,:).' ./ (2 * dy); -gradUY(end-1,:).' ./ (2 * dy) ];

FDY = sparse(I, J, V, numel(X), numel(X));

%--------------------------------------------------------------------------
% Assemble Fokker-Planck Operator
%--------------------------------------------------------------------------

if strcmpi(FPType, 'Forward')
    
    FPO = D .* (Lx + Ly) + FDX + FDY;
    
elseif strcmpi(FPType, 'Backward')
    
     FPO = D .* (Lx + Ly) - FDX - FDY;
    
elseif strcmpi(FPType, 'Hermitian')
    
    FPO = delU./2 + sum(reshape(gradU, numel(X), 2).^2, 2) ./ (4 * D);
    FPO = D * (Lx + Ly) + spdiags(FPO, 0, numel(X), numel(X));
    
end

%--------------------------------------------------------------------------
% Handle Boundary Conditions
%--------------------------------------------------------------------------

% Handle no-flux boundary conditions - basically just set the first order
% finite differences to satisfy the boundary conditions (corners?)
P0(gridIDx(:,1)) = P0(gridIDx(:,2)) ./ (1 - dx * BCGradUX(:,1) / D);
P0(gridIDx(:,end)) = P0(gridIDx(:,end-1)) ./ (1 + dx * BCGradUX(:,end) / D);
P0(gridIDx(1,:)) = P0(gridIDx(1,:)) ./ (1 - dy * BCGradUY(1,:) / D);
P0(gridIDx(end,:)) = P0(gridIDx(end-1,:)) ./ (1 + dy * BCGradUY(end,:) / D);

% Handle Dirichlet boundary conditions
% (These overwrite no-flux boundary conditions)
if ~isempty(dirBC)
    
    assert(isequal(sort(dirBC(:,1)), unique(dirBC(:,1))), ...
        'Multiple Dirichlet conditions for a single node');
    assert(all(ismember(dirBC(:,1), bdyIDx)), ...
        'Invalid boundary vertices for Dirichlet conditions');
    
    % Ensure initial conditions match the specified values
    P0(dirBC(:,1)) = dirBC(:, 2);
    
    % Modify the Fokker-Planck operators (time derivatives of nodes with
    % Dirichlet conditions should uniformly vanish)
    FPO(dirBC(:,1), :) = sparse(size(dirBC,1), size(FPO,2));
    
end

%--------------------------------------------------------------------------
% Solve Fokker-Planck Equation
%--------------------------------------------------------------------------

if strcmpi(odeIntType, 'non-stiff')
    
    FPSol = ode45( @(t,y) FPO * y, timeSpan, P0(:), odeOptions );
  
else
    
    FPSol = ode15s( @(t,y) FPO * y, timeSpan, P0(:), odeOptions );
    
end


end

% Define the output function
function status = odeprog(t, ~, flag)
    switch flag
        case 'init'
            % Initialization: Display any necessary information
            disp('ODE solver initialization');
        case ''
            % ODE evaluation: Display progress information
            disp(['Current time: ', num2str(t)]);
        case 'done'
            % Finalization: Display any necessary final information
            disp('ODE solver finished');
    end
    status = 0; % Return 0 to continue the solver
end

