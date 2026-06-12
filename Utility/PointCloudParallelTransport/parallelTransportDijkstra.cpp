/* ============================================================================
 *
 *  parallelTransportDijkstra.cpp
 *  
 *
 *  Parallel-transport Dijkstra for geodesic distances on point cloud
 *  graphs representing Riemannian submanifolds embedded in Euclidean
 *  space. Computes a pairwise geodesic distance matrix on by propagating
 *  an orthonormal frame along between tangent frames, and defines the
 *  geodesic distance from the norm of the transported displacement vector.
 *
 *  by Dillon Cislo
 *  02/11/2026
 *
 *  This is a MEX-file for MATLAB
 *  
 * ==========================================================================*/

#include "mex.h" // for MATLAB

#include <iostream>
#include <functional>
#include <queue>
#include <vector>
#include <algorithm>
#include <utility>
#include <limits>

// #include "../../External/eigen-5.0.0/Eigen/Core"
// #include "../../External/eigen-5.0.0/Eigen/src/SVD/JacobiSVD.h"
#include <Eigen/Core>
#include <Eigen/SVD>

typedef typename Eigen::MatrixXd MatrixXd;
typedef typename Eigen::VectorXd VectorXd;
typedef typename Eigen::MatrixXi MatrixXi;
typedef typename Eigen::VectorXi VectorXi;

typedef typename Eigen::Matrix<bool, Eigen::Dynamic, 1> VectorXb;

inline void adjacency_list(
    const MatrixXi &E,
    int numPoints,
    std::vector<std::vector<int>> &A) {

  // Allocate space for the lists
  A.clear();
  if (numPoints <= 0)
    numPoints = E.maxCoeff() + 1;
  A.resize(numPoints);

  // Loop over edges
  for (int i = 0; i < E.rows(); i++) {
    A.at(E(i,0)).push_back(E(i,1));
    A.at(E(i,1)).push_back(E(i,0));
  }

  // Remove duplicates
  for (int i = 0; i < (int) A.size(); ++i) {
    std::sort(A[i].begin(), A[i].end());
    A[i].erase(std::unique(A[i].begin(), A[i].end()), A[i].end());
  }

};

// Main function
void mexFunction( int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[] ) {

  const double inf_val = std::numeric_limits<double>::infinity();

  //--------------------------------------------------------------------------
  // INPUT PROCESSING
  //--------------------------------------------------------------------------

  // Check for proper number of arguments
  if ( nrhs != 3 ) {
    mexErrMsgIdAndTxt( "MATLAB:parallel_transport_dijkstra:nargin",
        "PARALLEL_TRANSPORT_DIJKSTRA requires 3 input arguments" );
  } else if ( nlhs != 1 ) {
    mexErrMsgIdAndTxt("MATLAB:parallel_transport_dijkstra:nargout",
        "PARALLEL_TRANSPORT_DIJKSTRA requries 1 output arguments" );
  }

  // Point cloud coordinates
  double *Xin = mxGetPr( prhs[0] );
  int numPoints = (int) mxGetM( prhs[0] );
  int ambiDim = (int) mxGetN( prhs[0] );

  // Proximity graph edge list
  double *Ein = mxGetPr( prhs[1] );
  int numEdges = (int) mxGetM( prhs[1] );
  if ((int) mxGetN(prhs[1]) != 2)
    mexErrMsgTxt("Proximity graph edge list is improperly sized");

  // Tangent space list
  const mxArray *allTCell = prhs[2];
  if (!mxIsCell(allTCell))
    mexErrMsgTxt("Tangent space must be supplied as a cell array");
  if (mxGetNumberOfElements(allTCell) != numPoints) {
    mexErrMsgTxt("Elements of the tangent space array must "
        "match number of points");
  }

  // Map input arrays to Eigen-style matrices
  MatrixXd X = Eigen::Map<MatrixXd>(Xin, numPoints, ambiDim);
  if (!X.array().isFinite().all())
    mexErrMsgTxt("Some elements of X are not finite");
  if (X.array().isNaN().any())
    mexErrMsgTxt("X contains NaN");

  MatrixXd Ed = Eigen::Map<MatrixXd>(Ein, numEdges, 2);
  MatrixXi E = Ed.cast <int> ();
  E = (E.array() - 1).matrix(); // Account for MATLAB indexing
  if (!E.array().isFinite().all())
    mexErrMsgTxt("Some elements of E are not finite");
  if (E.array().isNaN().any())
    mexErrMsgTxt("E contains NaN");
  if ((E.array() < 0).any() || (E.array() >= numPoints).any())
    mexErrMsgTxt("Edge list contains invalid indices");

  int dim = -1;
  std::vector<MatrixXd> allT;
  allT.reserve(numPoints);
  for (int i = 0; i < numPoints; i++) {

    const mxArray *curTptr = mxGetCell(allTCell, i);
    double *curTin = mxGetPr(curTptr);

    if (mxGetM(curTptr) != ambiDim) {
      mexErrMsgTxt("Rows of the tangent space array must "
          "match embedding space dimension");
    }

    if (dim < 0) {
      dim = mxGetN(curTptr);
    } else if (mxGetN(curTptr) != dim) {
      mexErrMsgTxt("Mismatch in intrinsic dimension "
          "of tangent space");
    }

    MatrixXd curT = Eigen::Map<MatrixXd>(curTin, ambiDim, dim);
    if (!curT.array().isFinite().all())
      mexErrMsgTxt("Some elements of tangent space array are not finite");
    if (curT.array().isNaN().any())
      mexErrMsgTxt("Tangent space array contains NaN");

    allT.push_back(curT);

  }

  // Build vertex adjacency list from edge list
  std::vector<std::vector<int>> A;
  adjacency_list(E, numPoints, A);

  //--------------------------------------------------------------------------
  // PERFORM PARALLEL TRANSPORT DIJKSTRA
  //--------------------------------------------------------------------------
  
  MatrixXd D = MatrixXd::Zero(numPoints, numPoints);

  // Loop over vertices
  for (int i = 0; i < numPoints; i++) {

    // Create fresh minimum priority queue
    std::priority_queue<
      std::pair<double, int>,
      std::vector<std::pair<double, int>>,
      std::greater<>> pq;

    // Cumulative transport maps and transported vectors
    // relative to current base point
    std::vector<MatrixXd> R;
    std::vector<VectorXd> v;
    R.reserve(numPoints);
    v.reserve(numPoints);
    for (int j = 0; j < numPoints; j++) {
      R.push_back(MatrixXd::Identity(dim, dim));
      v.push_back(VectorXd::Zero(dim));
    }

    // Cumulative distances relative to current base point
    VectorXd dist = VectorXd::Constant(numPoints, inf_val);
    VectorXd geo_dist = VectorXd::Constant(numPoints, inf_val);
    dist(i) = 0.0;
    geo_dist(i) = 0.0;

    // Iteratively updated predecessor index
    VectorXi pred = -VectorXi::Ones(numPoints);

    // Accumulate data on the neighbors of the current base point
    VectorXd xi = X.row(i).transpose();
    for (int jj = 0; jj < A[i].size(); jj++) {
      int j = A[i][jj];
      pred(j) = i;
      VectorXd xj = X.row(j).transpose();
      dist(j) = (xj - xi).norm();
      pq.emplace(dist(j), j);
    }

    while (!pq.empty()) {

      // Unpack data about the current point and its predecessor
      double dr = pq.top().first;
      int r = pq.top().second;
      pq.pop();
      if (dr > dist(r)) { continue; }
      VectorXd xr = X.row(r).transpose();
      MatrixXd Tr = allT[r];

      int q = pred(r);
      VectorXd xq = X.row(q).transpose();
      MatrixXd Tq = allT[q];

      // Compute the parallel transport map
      Eigen::JacobiSVD<MatrixXd, Eigen::ComputeFullU | Eigen::ComputeFullV>
        svd(Tq.transpose() * Tr);

      // Update the cumulative maps and transported vectors
      R[r] = R[q] * svd.matrixU() * svd.matrixV().transpose();
      v[r] = v[q] + R[q] * Tq.transpose() * (xr - xq);
      geo_dist(r) = v[r].norm();

      // Accumulate data on the neighbors of the current point
      for (int jj = 0; jj < A[r].size(); jj++) {
        int j = A[r][jj];
        VectorXd xj = X.row(j).transpose();
        double tmp_dist = dist(r) + (xj - xr).norm();
        if (tmp_dist < dist(j)) {
          dist(j) = tmp_dist;
          pred(j) = r;
          pq.emplace(dist(j), j);
        }
      }
    }

    D.row(i) = geo_dist.transpose();

  }

  D = (0.5 * (D.array() + D.transpose().array())).matrix();

  //--------------------------------------------------------------------------
  // OUTPUT PROCESSING
  //--------------------------------------------------------------------------

  plhs[0] = mxCreateDoubleMatrix( numPoints, numPoints, mxREAL );
  Eigen::Map<MatrixXd>( mxGetPr(plhs[0]), numPoints, numPoints ) = D;

  return;

};
