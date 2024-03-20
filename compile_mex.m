%% Run this script to compile all of the mex code

[projectDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(projectDir);

% Compile the NGL library
extDir = fullfile(projectDir, 'External');
cd(extDir);
compile_ngl(false); % <-- Set to 'true' for verbose output

% Compile NGL MATLAB bindings
nglDir = fullfile(projectDir, 'Utility/ProximityGraphs/NGL');
cd(nglDir);
compile_proximity_graphs_ngl;