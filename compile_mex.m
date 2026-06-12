%% COMPILE MEX CODE =======================================================
%
%   This is a script to help compile all of the code needed to use this
%   library. Some of the code is just straight up 'mex' code and should
%   compile very easily. Some of the code is part of various third-party
%   libraries and may require special care. Hopefully, this script manages
%   everything for you as painlessly as possible.
%
%   HOW TO USE THIS SCRIPT: Just press 'Run'! It is highly likely that you
%   will encounter errors on your first run through. When this happens,
%   navigate to the function (e.g. 'compile_ngl') that threw the error and
%   check for more detailed instructions.
%
%   COMMON ISSUES:
%       
%       (1) Missing dependencies. This code ships out-of-the-box with most
%       of its dependencies, but others you must install on your own. All
%       of these dependencies can be installed trivially through a package
%       manager (e.g. Homebrew on macOS). Please check the individual
%       functions for detailed install commands.
%
%       (2) MATLAB nonsense. MATLAB ships with pre-packaged versions of
%       C/C++ libraries that are always out of date. Since you are running
%       this script from within MATLAB, it overrides your usual PATH and
%       can lead to annoying errors. The solution is to ensure that MATLAB
%       knows to user the system libraries instead
%
%       FOR LINUX USERS: Find your system version of 'libstdc++.so.6' and
%       see where it points. Usually this is something like
%
%       >> ls -lah /usr/lib/x86-64-linux-gnu/libstdc++.so.6
%       >> lrwxrwxrwx 1 root root 19 May 13  2023 libstdc++.so.6 -> libstdc++.so.6.0.30
%
%       The safest way to proceed is to COPY the target of this soft link
%       into the proper location in your MATLAB root and update the
%       corresponding soft link in that folder
%
%       >> sudo cp /usr/lib/x86-64-linux-gnu/libstdc++.so.6.0.30 MATLABRoot/sys/os/glxna64
%       >> cd MATLABRoot/sys/os/glxna64
%       >> chmod ugo+x libstdc++.so.6.0.30
%       >> ln -sf libstdc++.so.6.0.30 libstdc++.so.6
%
%       This is a little annoying because you'll have to do it every time
%       you update, but it's safer than modifying system files. A more sane
%       solution would be to just set the "LD_PRELOAD" environmental
%       variable to point to the system libraries, but this never seems to
%       work for me. You will have to do this for libstdc++.so.6,
%       libcurl.so.4, and libuv.so.1.
%
%   WARNING: This script does not currently support Windows!!
%   
%   by Dillon Cislo 2024/03/22

[projectDir, ~, ~] = fileparts(matlab.desktop.editor.getActiveFilename);
cd(projectDir);

% Compile the NGL library
extDir = fullfile(projectDir, 'External');
cd(extDir);
compile_ngl(true); % <-- Set to 'true' for verbose output

% Compile NGL MATLAB bindings
nglDir = fullfile(projectDir, 'Utility/ProximityGraphs/NGL');
cd(nglDir);
compile_proximity_graphs_ngl;

% Compile parallel transport Dijkstra algorithm
ptdDir = fullfile(projectDir, 'Utility/PointCloudParallelTransport');
cd(ptdDir);
compile_parallel_transport_dijkstra