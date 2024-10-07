/* =============================================================================================
 *
 *  polygon_intersection_area.cpp
 *  
 *  This function calculates the area of the intersection regions between a base polygon 
 *  and a set of additional polygons. The input polygons are represented as vertex lists, 
 *  and it is assumed that all additional polygons have the same number of vertices. The base 
 *  polygon and the additional polygons are assumed to be simple (i.e., non-self-intersecting 
 *  and simply connected) and to have vertices ordered in a counter-clockwise (CCW) direction.
 *
 *  Input Arguments:
 *    1. basePolyVertices (Mx2 double matrix) - The vertex list defining the base polygon, 
 *       where M is the number of vertices. Each row is of the form [x, y], representing the 
 *       coordinates of a vertex. 
 *    2. polyX (NxK double matrix) - The x-coordinates of the vertices of the K additional 
 *       polygons, where N is the number of vertices in each polygon.
 *    3. polyY (NxK double matrix) - The y-coordinates of the vertices of the K additional 
 *       polygons, where N is the number of vertices in each polygon.
 *
 *  Output Arguments:
 *    1. intersectionAreas (Kx1 double matrix) - A column vector containing the areas of the 
 *       intersection regions between the base polygon and each of the K additional polygons.
 *
 *  Notes:
 *    - The polygons must all be simple, meaning they should not self-intersect, and must be 
 *      simply connected.
 *    - The vertices defining the base polygon and the additional polygons must be in 
 *      counter-clockwise order (CCW).
 *    - The additional polygons must all have the same number of vertices.
 *    - Boost Geometry expects clockwise (CW) ordering of the vertices, so the vertex lists 
 *      are reversed internally to match this convention.
 *
 *  Example Usage (MATLAB):
 *    % Define the base polygon with vertices in a Mx2 matrix
 *    basePolyVertices = [x1, y1; x2, y2; ...; xM, yM];
 *    
 *    % Define K additional polygons using their x-coordinates and y-coordinates in NxK matrices
 *    polyX = [x11, x12, ..., x1K; x21, x22, ..., x2K; ...; xN1, xN2, ..., xNK];
 *    polyY = [y11, y12, ..., y1K; y21, y22, ..., y2K; ...; yN1, yN2, ..., yNK];
 *
 *    % Calculate intersection areas
 *    intersectionAreas = polygon_intersection_area(basePolyVertices, polyX, polyY);
 *
 *  Author:
 *    Dillon Cislo
 *    05/19/2022
 *
 *  This is a MEX-file for MATLAB.
 *  
 * ============================================================================================*/


#include "mex.h" // for MATLAB

#include <iostream>
#include <deque>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/foreach.hpp>

using namespace boost::geometry;

typedef model::d2::point_xy<double> point_xy;
typedef model::polygon<point_xy> polygon;

// Main function
void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[] ) {

  //-------------------------------------------------------------------------------------
  // INPUT PROCESSING
  // ------------------------------------------------------------------------------------

  // Check for proper number of arguments
  if ( nrhs != 3 ) {
    mexErrMsgIdAndTxt( "MATLAB:polygon_intersection_area:nargin",
        "POLYGON_INTERSECTION_AREA requires 3 input arguments" );
  } else if ( nlhs != 1 ) {
    mexErrMsgIdAndTxt("MATLAB:polygon_intersection_area:nargout",
        "POLYGON_INTERSECTION_AREA requries 1 output arguments" );
  }

  // The vertex list defining the face polygon
  double *basePolyVertices = mxGetPr( prhs[0] );
  int numBaseVertices = (int) mxGetM( prhs[0] );

  if ( (int) mxGetN( prhs[0] ) != 2 )
    mexErrMsgTxt("Base polygon vertex list is improperly sized");

  // The x-coordinates of the additional polygons
  double *polyX = mxGetPr( prhs[1] );
  int numAddVertices = (int) mxGetM( prhs[1] );
  int numAddPoly = (int) mxGetN( prhs[1] );

  // The y-coordinates of the additional polygons
  double *polyY = mxGetPr( prhs[2] );

  if ( ((int) mxGetM(prhs[2]) != numAddVertices) ||
      ((int) mxGetN(prhs[2]) != numAddPoly) )
    mexErrMsgTxt("Additional polygon inputs are improperly sized");

  // Construct the base polygon. Boost actually takes CW polygons
  // so we fill in the vertices in reverse order
  std::vector<point_xy> ppts(numBaseVertices);
  for( int i = 0; i < numBaseVertices; i++ )
  {
    ppts[numBaseVertices - i - 1] =
      point_xy(basePolyVertices[i], basePolyVertices[i+numBaseVertices]);
  }

  // Boost also expects CLOSED polygons (i.e. fist vertex equals the last vertex)
  if (!equals(ppts[0], ppts[numBaseVertices - 1]))
    ppts.push_back(ppts[0]);

  polygon P;
  assign_points(P, ppts);

  //-------------------------------------------------------------------------------------
  // CALCULATE INTERSECTION AREAS
  //-------------------------------------------------------------------------------------
  
  plhs[0] = mxCreateDoubleMatrix( numAddPoly, 1, mxREAL );
  double *intersectionAreas = mxGetPr( plhs[0] );

  for( int j = 0; j < numAddPoly; j++ )
  {

    // Construct the current polygon. Boost actually takes CW polygons
    // so we fill in the vertices in reverse order
    std::vector<point_xy> qpts(numAddVertices);
    for( int i = 0; i < numAddVertices; i++)
    {
      qpts[numAddVertices - i -1] =
        point_xy(polyX[i+(j*numAddVertices)], polyY[i+(j*numAddVertices)]);
    }

    // Boost also expects CLOSED polygons (i.e. first vertex equals the last vertex)
    if (!equals(qpts[0], qpts[numAddVertices - 1]))
      qpts.push_back(qpts[0]);

    polygon Q;
    assign_points(Q, qpts);

    // Compute the intersection of P and Q
    std::deque<polygon> R;
    intersection(P, Q, R);

    // Compute the total area of the intersection
    double intArea = 0.0;
    BOOST_FOREACH(polygon const &p, R)
      intArea += area(p);

    intersectionAreas[j] = intArea;

  }

  return;

};
