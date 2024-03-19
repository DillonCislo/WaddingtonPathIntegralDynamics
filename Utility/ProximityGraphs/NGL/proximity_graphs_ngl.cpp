/* =============================================================================================
 *
 *  proximity_graphs_ngl.cpp
 *  
 *  A MATLAB binding to the Neighborhood Graph Library (NGL)
 *  (https://code.google.com/archive/p/ngl/)
 *
 *  Produces a variety of neighborhood graphs for a given set of input points
 *
 *  by Dillon Cislo
 *  02/01/2023
 *
 *  This is a MEX-file for MATLAB
 *  
 * ============================================================================================*/

#include "mex.h" // for MATLAB

#include <iostream>

#include "ngl.h"

using namespace ngl;

// An enumeration of the different graph constructions
enum NGL_GRAPH {

  // Relative neighborhood graph
  NGL_RNG = 1,

  // Gabriel graph
  NGL_GG = 2,

  // Beta skeleton
  NGL_BS = 3,

  // Relaxed relative neighborhood graph
  NGL_RRNG = 4,

  // Relaxed Gabriel graph
  NGL_RGG = 5,

  // Relaxed beta skeleton
  NGL_RBS = 6

};

// Main function
void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[] ) {

  // Check for proper number of arguments
  if ( nrhs != 4 ) {
    mexErrMsgIdAndTxt( "MATLAB:proximity_graphs_ngl:nargin",
        "PROXIMITY_GRAPHS_NGL requires 4 input arguments" );
  } else if ( nlhs != 1 ) {
    mexErrMsgIdAndTxt("MATLAB:proximity_graphs_ngl:nargout",
        "PROXIMITY_GRAPHS_NGL requries 1 output arguments" );
  }

  double *pInD = mxGetPr( prhs[0] ); // The input point cloud
  // float *pIn = (float*) mxGetPr( prhs[0] ); // The input point cloud
  int numPoints = mxGetM( prhs[0] ); // The number of points
  int dim = mxGetN( prhs[0] ); // The dimensionality of the point list

  // This is really weird - I think they take the transpose?
  float *pIn = new float[(numPoints*dim)];
  for( int i = 0; i < numPoints; i++ )
    for( int j = 0; j < dim; j++ )
      pIn[(dim*i)+j] = (float) pInD[i+(j*numPoints)];

  // The desired output graph type
  int graphType = (int) *mxGetPr( prhs[1] );

  // Maximum number of nearest neighbors to use in the calculation
  int kNN = (int) *mxGetPr( prhs[2] );
  if ( kNN <= 0 ) { kNN = -1; }

  // Beta-skeleton parameter
  float beta = (float) *mxGetPr( prhs[3] );

  // Initialize problem geometry
  Geometry<float>::init(dim);

  // Initialize parameters
  NGLParams<float> params;
  params.param1 = beta;
  params.iparam0 = kNN;
  IndexType *indices;
  int numEdges;

  // Calculate neighborhood graphs
  // This is hideous, but I don't feel like properly setting up
  // the polymorphism and there shouldn't be a hit to execution time
  if ( graphType == NGL_RNG ) {

    if ( kNN <= 0 ) {

      NGLPointSet<float> P(pIn, numPoints);
      getRelativeNeighborGraph(P, &indices, numEdges, params);

    } else {

      ANNPointSet<float> P(pIn, numPoints);
      getRelativeNeighborGraph(P, &indices, numEdges, params);

    }

  } else if ( graphType == NGL_GG ) {

    if ( kNN <= 0 ) {

      NGLPointSet<float> P(pIn, numPoints);
      getGabrielGraph(P, &indices, numEdges, params);

    } else {

      ANNPointSet<float> P(pIn, numPoints);
      getGabrielGraph(P, &indices, numEdges, params);

    }

  } else if ( graphType == NGL_BS ) {

    if ( kNN <= 0 ) {

      NGLPointSet<float> P(pIn, numPoints);
      getBSkeleton(P, &indices, numEdges, params);

    } else {

      ANNPointSet<float> P(pIn, numPoints);
      getBSkeleton(P, &indices, numEdges, params);

    }

  } else if ( graphType == NGL_RRNG ) {

    if ( kNN <= 0 ) {

      NGLPointSet<float> P(pIn, numPoints);
      getRelaxedRelativeNeighborGraph(P, &indices, numEdges, params);

    } else {

      ANNPointSet<float> P(pIn, numPoints);
      getRelaxedRelativeNeighborGraph(P, &indices, numEdges, params);

    }

  } else if ( graphType == NGL_RGG ) {

    if ( kNN <= 0 ) {

      NGLPointSet<float> P(pIn, numPoints);
      getRelaxedGabrielGraph(P, &indices, numEdges, params);

    } else {

      ANNPointSet<float> P(pIn, numPoints);
      getRelaxedGabrielGraph(P, &indices, numEdges, params);

    }

  } else if ( graphType == NGL_RBS ) {

    if ( kNN <= 0 ) {

      NGLPointSet<float> P(pIn, numPoints);
      getRelaxedBSkeleton(P, &indices, numEdges, params);

    } else {

      ANNPointSet<float> P(pIn, numPoints);
      getRelaxedBSkeleton(P, &indices, numEdges, params);

    }

  } else { mexErrMsgTxt("Invalid output graph type"); }

  // Format graph edge output
  // NOTE: All of the above functions return the edges of a DIRECTED graph.
  // Presumably this is because the relaxed versions of the graph do NOT
  // need to be symmetric. We will return all DIRECTED edges and leave
  // further formatting to the associated MATLAB wrapper
  plhs[0] = mxCreateDoubleMatrix( numEdges, 2, mxREAL );
  double *edgesOut = mxGetPr( plhs[0] );
  for (unsigned int i = 0; i < numEdges; i++ ) {

    edgesOut[i] = (double) (indices[2*i] + 1.0);
    edgesOut[i+numEdges] = (double) (indices[2*i+1] + 1.0);

  }


  return;

};
