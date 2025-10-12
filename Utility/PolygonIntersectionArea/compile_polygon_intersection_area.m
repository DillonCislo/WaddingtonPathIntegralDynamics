function compile_polygon_intersection_area
%COMPILE_POLYGON_INTERSECTION_AREA Build the polygon intersection MEX
%binary shipped with WPID. Adjust include/lib paths as needed before
%running this helper.
%
%   This utility invokes MATLAB's MEX interface with OpenMP support to
%   compile `polygon_intersection_area.cpp`. It assumes Boost headers and
%   libraries are available in standard system locations.
%
%   by Dillon Cislo 2025/10/12

clc;

mex -v -O polygon_intersection_area.cpp ...
    CXXOPTIMFLAGS="-O3" ...
    CXXFLAGS="$CXXFLAGS -std=c++17" ...
    LDFLAGS="$LDFLAGS -fopenmp" ...
    -I/usr/include ...
    -I/usr/local/include ...
    -L/usr/lib:/usr/local/lib  -lboost_thread -lboost_system

end