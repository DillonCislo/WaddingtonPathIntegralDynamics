function [FPSol, FPO] = solveFokkerPlanck1DFD(X, P0, varargin)
%SOLVEFOKKERPLANCK1DFD Solves the forward Fokker-Planck equation on a fixed
%1D grid useing the numerical method of lines with a simple finite
%differencing spatial discretization scheme. For a uniform diffusion
%constant D and a given potential field U(x) the Fokker-Planck equation can
%be written as
%
%   dp/dt = D (del p) + (del U) p + (grad U) * (grad p)
%
%The solver is capabile of handling mixed Dirichlet/Neumann boundary
%conditions. By default, all vertices whose boundary conditions are not
%explicitly specified by the input variables are assigned zero flux
%boundary conditions, i.e.
%
%           D (grad p) + (grad U) p = 0
%
%   INPUT PARAMETERS
%
%       - X:	#Nx1 vector of x-coordinates of evaluation points. Values
%               should conform to the output of the 'linspace' function
%
%       - P0:   #Nx1 vector of initial density values
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
%               (1) An #Nx1 matrix of nodal values
%               (2) An anonymous function, i.e. U = @(x) ...
%               (3) A symbolic scalar function terms of a symbolic
%               variable 'x' 
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
%       - ('DirichletBoundaryConditions', dirBC = [NaN NaN]): A 1x2 vector
%       corresponding to the two boundary points on the grid. Non-NaN
%       values denote Dirichlet boundary conditions for the corresponding
%       boundary vertex
%
%       - ('NeumannBoundaryConditions', neuBC = [NaN NaN]): A 1x2 vector
%       corresponding to the two boundary points on the grid. Non-NaN
%       values denote Neumann boundary conditions for the corresponding
%       boundary vertex
%
%   OUTPUT PARAMETERS:
%
%       - FPSol:    Solution output structure that can be evaluated for
%                   any time on the interval given by 'timeSpan'
%
%       - FPO:      #Nx#N Linear Fokker-Planck Operator
%
%   by Dillon Cislo 07/26/2023

%--------------------------------------------------------------------------
% Input Processing
%--------------------------------------------------------------------------
if (nargin < 1), error('Please supply evaluation point x-coordinates'); end
if (nargin < 2), error('Please supply initial density values'); end

validateattributes(X, {'numeric'}, {'vector', 'finite', 'real'});
numPoints = numel(X);
if (size(X,2) ~= 1), X = X.'; end

validateattributes(P0, {'numeric'}, {'vector', 'finite', 'real', ...
    'numel', numPoints});
if (size(P0,2) ~= 1), P0 = P0.'; end

assert((std(unique(diff(X))) < 1e-10) && all(diff(X,1) > 0), ...
    'Invalid x-coordinate format. See ''linspace'' documentation');

dx = mean(unique(diff(X))); % Grid spacing in x

% Set Optional Parameters -------------------------------------------------

D = 1;
U = zeros(size(X));
timeSpan = [0 1];
odeIntType = 'stiff';
odeOptions = odeset;
FPType = 'forward';
dirBC = nan(1,2);
neuBC = nan(1,2);

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
    
    if strcmpi(varargin{i}, 'DirichletBoundaryConditions')
        dirBC = varargin{i+1};
        validateattributes(dirBC, {'numeric'}, {'vector', 'numel', 2});
        if ~isnan(dirBC(1))
            assert(~isinf(dirBC(1)) && isreal(dirBC(1)), ...
                'Invalid Dirichlet boundary condition input');
        end
        if ~isnan(dirBC(2))
            assert(~isinf(dirBC(2)) && isreal(dirBC(2)), ...
                'Invalid Dirichlet boundary condition input');
        end
    end
    
    if strcmpi(varargin{i}, 'NeumannBoundaryConditions')
        error('General Neumann boundary conditions are not yet implemented');
        neuBC = varargin{i+1};
        validateattributes(neuBC, {'numeric'}, {'vector', 'numel', 2});
        if ~isnan(neuBC(1))
            assert(~isinf(neuBC(1)) && isreal(neuBC(1)), ...
                'Invalid Neumann boundary condition input');
        end
        if ~isnan(neuBC(2))
            assert(~isinf(neuBC(2)) && isreal(neuBC(2)), ...
                'Invalid Neumann boundary condition input');
        end
    end
    
end

%--------------------------------------------------------------------------
% Process Potential Field Input
%--------------------------------------------------------------------------

if isa(U, 'sym')
    
    syms x
    assume(x, 'real');
    
    assert(all(ismember(symvar(U), {'x'})), ...
        'Invalid symbolic potential input');
    
    gradU = matlabFunction(gradient(U, x), 'Vars', x);  
    delU = matlabFunction(gradient(gradient(U, x), x), 'Vars', x);
    % delU = matlabFunction(laplacian(U, [x,y]), 'Vars', {x,y});
    U = matlabFunction(U, 'Vars', x);
    
    U = reshape(U(X), size(X));
    validateattributes(U, {'numeric'}, {'vector', 'finite', 'real', ...
    'numel', numPoints});
    
    gradU = reshape(gradU(X), size(X)); 
    delU = reshape(delU(X), size(X));

elseif (isa(U, 'function_handle') || isvector(U))
    
    if isa(U, 'function_handle')
        U = reshape(U(X), size(X));
    end
    
    validateattributes(U, {'numeric'}, {'vector', 'finite', 'real', ...
    'numel', numPoints});
    
    % Calculate numerical gradient
    gradU = gradient(U, dx);
    
    % Calculate numerical second derivative along x
    delU = zeros(size(U));
    delU(2:(end-1)) = (diff(U(2:end),1)-diff(U(1:(end-1)),1)) / dx^2;
    delU(1) = 2 * delU(2) - delU(3);
    delU(end) = -delU(end-2) + 2 * delU(end-1);
    
else
    
    error('Invalid potential field input');
    
end

validateattributes(gradU, {'numeric'}, {'vector', ...
    'finite', 'real', 'numel', numel(X)});
validateattributes(delU, {'numeric'}, {'vector', ...
    'finite', 'real', 'numel', numel(X)});

%--------------------------------------------------------------------------
% Formulate Derivative Operators on Finite Difference Grid
%--------------------------------------------------------------------------
% NOTE: These are not the typical finite difference operators!!!
% They are formulated specifically so that the final Fokker-Planck
% operators will automatically satisfy all no-flux/user specified Neumann
% boundary conditions.

% Second Derivative Operator ----------------------------------------------

% Central difference formula for interior vertices
I = (2:(numPoints-1)).';
J = [I; I+1; I-1];
V = [-2*ones(size(I)); ones(size(I)); ones(size(I))] / dx^2;
I = [I; I; I];

% Modified central difference formula for left boundary vertex
I = [I; 1; 1];
J = [J; 1; 2];
V = [V; -1/dx^2; 1/dx^2];

% Modified central difference formula for right boundary vertex
I = [I; numPoints; numPoints];
J = [J; numPoints; numPoints-1];
V = [V; -1/dx^2; 1/dx^2];

L = sparse(I, J, V, numPoints, numPoints);

% Assemble Forward Divergence Operator ------------------------------------
% Equivalent to d/dx * ( (dUdx) p ) = (d^2Udx^2)p + (dUdx) * (dpdx)

% Central difference formula for interior vertices
I = (2:(numPoints-1)).';
J = [I+1; I-1];
V = [gradU(3:end); -gradU(1:(end-2))]/(2*dx);
I = [I; I];

% Modified central difference formula for left boundary vertex
I = [I; 1; 1];
J = [J; 1; 2];
V = [V; gradU(1)/(2*dx); gradU(2)/(2*dx)];

% Modified central difference formula for right boundary vertex
I = [I; numPoints; numPoints];
J = [J; numPoints; numPoints-1];
V = [V; -gradU(end)/(2*dx); -gradU(end-1)/(2*dx)];

FD = sparse(I, J, V, numPoints, numPoints);

% Assemble Fokker-Planck Operator -----------------------------------------
FPO = D .* L + FD;

%--------------------------------------------------------------------------
% Handle Boundary Conditions
%--------------------------------------------------------------------------

% Handle no-flux boundary conditions - basically just set the first order
% finite differences to satisfy the boundary conditions
P0(1) = P0(2) / (1 - dx * gradU(1) / D);
P0(end) = P0(end-1) / (1 + dx * gradU(end) / D);

% Handle Dirichlet boundary conditions (these overwrite the no-flux
% boundary conditions): (1) Ensure initial conditions match the specified
% values and (2) Modify the Fokker-Planck operators (time derivatives of
% nodes with Dirichlet conditions should uniformly vanish)
if ~isnan(dirBC(1))
    P0(1) = dirBC(1); FPO(1, :) = sparse(1, numPoints);  
end
if ~isnan(dirBC(2))
    P0(end) = dirBC(2); FPO(end, :) = sparse(1, numPoints);
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

