function compile_parallel_transport_dijkstra
%COMPILE_PARALLEL_TRANSPORT_DIJKSTRA Build the parallel transport Dijkstra
%MEX bindings.
%
%   The script configures include and library paths pointing to the
%   Eigen, then invokes MATLAB's MEX toolchain with OpenMP
%   enabled to compile `parallelTransportDijkstra.cpp`.
%
%   by Dillon Cislo 2026/02/11

[projectDir, ~, ~] = fileparts(mfilename("fullpath"));
cd(projectDir);

CXXOPTIMFLAGS = '"-O3" ';
CXXFLAGS = '"$CXXFLAGS -march=native -fopenmp -fPIC -std=c++17" ';
LDFLAGS = '"$LDFLAGS -fopenmp -fPIC" ';

eigenDir = fullfile(projectDir, '../../External/eigen-5.0.0');
includeFlags = [ '-I' eigenDir ' '];

libFlags = [];

mexString = [ 'mex -v -O parallelTransportDijkstra.cpp ' ...
    'CXXOPTIMFLAGS=' CXXOPTIMFLAGS ' ' ...
    'CXXFLAGS=' CXXFLAGS ' ' ...
    'LDFLAGS=' LDFLAGS ' ' ...
    includeFlags libFlags ];

eval(mexString);

end

