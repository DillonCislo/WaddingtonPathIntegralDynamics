function compile_proximity_graphs_ngl

[projectDir, ~, ~] = fileparts(mfilename("fullpath"));
cd(projectDir);

CXXOPTIMFLAGS = '"-O3" ';
CXXFLAGS = '"$CXXFLAGS -march=native -fopenmp -fPIC -std=c++17" ';
LDFLAGS = '"$LDFLAGS -fopenmp -fPIC" ';

nglDir = fullfile(projectDir, '../../../External/ngl-beta/include');
annDir = fullfile(projectDir, '../../../External/ngl-beta/ann_1.1.2/include');
includeFlags = [ ...
    '-I' nglDir ' ' ...
    '-I' annDir ' ' ];

annLibDir = fullfile(projectDir, '../../../External/ngl-beta/ann_1.1.2/lib');
libFlags = [ '-L' annLibDir ' ' ];
libFlags = [ libFlags '-lANN '];

mexString = [ 'mex -v -O proximity_graphs_ngl.cpp ' ...
    'CXXOPTIMFLAGS=' CXXOPTIMFLAGS ' ' ...
    'CXXFLAGS=' CXXFLAGS ' ' ...
    'LDFLAGS=' LDFLAGS ' ' ...
    includeFlags libFlags ];

eval(mexString);

end

