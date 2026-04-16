/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2023 Tomislav Maric, TU Darmstadt
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "Pstream.H"
#include "tmopMotionSolver.H"
#include "addToRunTimeSelectionTable.H"
#include "OFstream.H"
#include "meshTools.H"
#include "syncTools.H"
#include "mapPolyMesh.H"
#include "fvPatch.H"
#include "fixedValuePointPatchFields.H"
//#include "motionInterpolation.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(tmopMotionSolver, 0);

    addToRunTimeSelectionTable
    (
        motionSolver,
        tmopMotionSolver,
        dictionary
    );

    addToRunTimeSelectionTable
    (
        displacementMotionSolver,
        tmopMotionSolver,
        displacement
    );
}

Foam::labelList Foam::tmopMotionSolver::filterValidCmpts(const Vector<label>& dims)
{
    labelList valid;

    forAll(dims, dI)
        if (dims[dI] == 1)  // Active solution dimension in OpenFOAM
            valid.push_back(dI); // Valid dimension 0 (x), 1 (y), or 2 (z)

    return valid;
}

// * * * * * * * * * * * * * Private Member Functions * * * * * * * * * * * * * * //

void Foam::tmopMotionSolver::writeSolutionDimToDatabase()
{
    client_.put_tensor("solution_dim",
                        &solutionDim_,
                        {1},
                        SRTensorTypeInt32,
                        SRMemLayoutContiguous);
}


void Foam::tmopMotionSolver::writeDistanceToBoundary()
{
    const auto& localMesh = motionSolver::mesh();
    const auto& meshBoundary = localMesh.boundaryMesh();
    const auto& meshPoints = localMesh.points();
    labelList pointsToGlobal, uniquePoints;
    const auto globalIDs = *localMesh.globalData().mergePoints(pointsToGlobal, uniquePoints);
    scalarField DistanceToBoundary(uniquePoints.size(), GREAT);

    // Loop over all patches on the boundary
    forAll(meshBoundary, patchI)
    {
        if (meshBoundary[patchI].type() == "empty"
           || meshBoundary[patchI].type() == "processor")
        {
           continue;
        }

        const polyPatch& patch   = meshBoundary[patchI];
        const labelList& patchPointIds = patch.meshPoints();

        // Loop over all points in the bulk
        forAll(uniquePoints, pointI)
        {
            const label localIdx = uniquePoints[pointI];
            // Loop over all points on the boundary
            forAll(patchPointIds, id)
            {
                scalar dist = mag(meshPoints[localIdx] - meshPoints[patchPointIds[id]]);
                DistanceToBoundary[pointI] = min(DistanceToBoundary[pointI], dist);
            }
        }
    }
    client_.put_tensor(rankMeshDistancesName_,
                       DistanceToBoundary.cdata(),
                       {size_t(uniquePoints.size()), 1},
                       SRTensorTypeDouble,
                       SRMemLayoutContiguous);

}

void Foam::tmopMotionSolver::writeMeshPointsToDatabase()
{
    const auto& localMesh = motionSolver::mesh();
    const auto& meshPoints = localMesh.points(); //fvMesh_.points();
    labelList pointsToGlobal, uniquePoints;
    const auto globalIDs = *localMesh.globalData().mergePoints(pointsToGlobal, uniquePoints);

    if (solutionDim_ == 3) // 3D case
    {
        // Send existing 3D mesh points for forward inference: nPoints,
        // dim=3. Saves time and memory in avoiding to create a 2D point buffer.
        client_.put_tensor(rankMeshPointsName_,
                           meshPoints.cdata(),
                           {size_t(meshPoints.size()), 3},
                           SRTensorTypeDouble,
                           SRMemLayoutContiguous);
    }
    else if (solutionDim_ == 2) // OpenFOAM pseudo 2D case
    {

        // Initialize local buffer for 2D points.
        std::vector<double> points2D(uniquePoints.size() * solutionDim_);
        std::vector<label> globalNodeIDs(uniquePoints.size());

        // Fill the points2D buffer with 2D mesh points
        // Only sending unique points per each processor,
        // i.e. each processor sends points it owns.
        label i = 0;
        forAll(uniquePoints, pointI)
        {
            const label localIdx = uniquePoints[pointI];
            const label globalIdx = pointsToGlobal[localIdx];

            const point& p = meshPoints[localIdx];
            // Assign components in points2D using valid solution directions and
            // 3D meshPoint data.
            points2D[2*i] = p[validCmpts_[0]];
            points2D[2*i+ 1] = p[validCmpts_[1]];
            // store global index of given point
            globalNodeIDs[i] = globalIdx;
            i++;
        }

        // Send points2D to SmartRedis
        client_.put_tensor(
            rankMeshPointsName_,
            points2D.data(),
            {size_t(uniquePoints.size()), size_t(solutionDim_)},
            SRTensorTypeDouble,
            SRMemLayoutContiguous
        );
        client_.put_tensor(
            rankMeshPointsIndicesName_,
            globalNodeIDs.data(),
            {size_t(globalNodeIDs.size())},
            SRTensorTypeInt32,
            SRMemLayoutContiguous
        );
    }
}

// do not need this for TMOP since we can just
// filter the points in python later (?)
void Foam::tmopMotionSolver::writeBoundaryPointsToDatabase()
{
    const pointField& points0 = this->points0();
    List<point> mpiRankPoints;
    const auto& meshBoundary = motionSolver::mesh().boundaryMesh();
    const auto& boundaryDisplacements = pointDisplacement().boundaryField();
    // Aggregate all points on the boundaries
    forAll(boundaryDisplacements, patchI)
    {
        if (meshBoundary[patchI].type() == "empty"
           || meshBoundary[patchI].type() == "processor")
        {
           continue;
        }

        const polyPatch& patch   = meshBoundary[patchI];
        const labelList& patchPointIds = patch.meshPoints();

        forAll(patchPointIds, id)
        {
            mpiRankPoints.append(points0[patchPointIds[id]]);
        }
    }

    List<List<point>>   globalPointListList(Pstream::nProcs());
    globalPointListList[Pstream::myProcNo()] = mpiRankPoints;
    Pstream::gatherList(globalPointListList);

    // - Send data to SmartRedis for ML model training from the main rank (0)
    if (Pstream::myProcNo() == 0)
    {
        // - Compute the global number of boundary points and displacements.
        label nGlobalBoundaryPoints = 0;
        forAll(globalPointListList, rankI)
        {
            nGlobalBoundaryPoints += globalPointListList[rankI].size();
        }
        // - Resize agglomerated point and displacement data to equal size.
        boundaryPoints_.resize(nGlobalBoundaryPoints * solutionDim_);

        // - Agglomerate the gathered boundary List<List<vector>> points and
        // displacements into boundaryPoints_ and boundaryDisplacements_ attributes.
        label globalCmptI = 0;
        forAll(globalPointListList, rankI)
        {
            // Get the list of points from each rank
            const List<point>& rankPoints = globalPointListList[rankI];
            forAll(rankPoints, pointI)
            {
                forAll(validCmpts_, dimI)
                {
                    boundaryPoints_[globalCmptI] = rankPoints[pointI][validCmpts_[dimI]];
                    ++globalCmptI;
                }
            }
        }
        // Send points to SmartRedis for ML model training.
        client_.put_tensor(
            "points",
            boundaryPoints_.data(),
            {size_t(nGlobalBoundaryPoints), size_t(solutionDim_)},
            SRTensorTypeDouble,
            SRMemLayoutContiguous
        );
    }
}

void Foam::tmopMotionSolver::writeMeshElements()
{
    /* 
    * Build a list of list (actually a std::vector<label>)
    * that stores the node labels for each element in the mesh. 
    * This only works in 2D for now, and assumes the
    * existence of empty "front" and "back" patches,
    * where faces in the "front" patch list nodes in
    * the counter-clockwise direction (positive orientation
    * for TMOP). Only "front" elements are considered.
    * Since elements might have different number of
    * points, pad the lists with -1 so that a nice
    * tensor can be sent to the database.
    */
    const auto& localMesh = motionSolver::mesh();
    const auto& patches = localMesh.boundaryMesh();
    labelList pointsToGlobal, uniquePoints;
    const auto globalIDs = *localMesh.globalData().mergePoints(pointsToGlobal, uniquePoints);

    label frontPatchID = patches.findPatchID("front");
    label emptyPatchID = frontPatchID;
    // forAll(patches, i)
    // {
    //     if (patches[i].type() == "empty")
    //     {
    //         emptyPatchID = i;
    //         break;
    //     }
    // }

    const auto& patch = patches[emptyPatchID];

    // get maximum number of nodes per element in the mesh
    label max_nodes = 0;
    forAll(patch, fI)
    {
        max_nodes = std::max(max_nodes, patch[fI].size());
    }
    // get max across MPI ranks.
    // Sometimes it might happen that rank i
    // only has triangles but rank j also
    // has quadrilaterals. Need to know if
    // this is the case.
    reduce(max_nodes, maxOp<label>(), max_nodes);

    // iterate over faces in the "front" patch, and
    // fill the elements with the global id of each point
    // in the face.
    std::vector<label> elements_padded(patch.size() * max_nodes);
    forAll(patch, fI)
    {
        const auto& face = patch[fI];
        const label& el_size = face.size();

        // labelList global_f = globalIDs.toGlobal(face);

        label row = fI * max_nodes;
        for (label j = 0; j < max_nodes; j++)
        {
            elements_padded[row + j] = j < el_size ? pointsToGlobal[face[j]] : -1;
        }
    }

    client_.put_tensor(
        rankElementsName_,
        elements_padded.data(),
        {size_t(patch.size()), size_t(max_nodes)},
        SRTensorTypeInt32,
        SRMemLayoutContiguous
    );

}

void Foam::tmopMotionSolver::matchFrontAndBack()
{
    /*
    * Since we only work with the "front" patch, build a
    * map from the "front" nodes to the corresponding "back"
    * nodes. This is used later to assign the same displacement
    * to "front" and "back", so that no twoDPointCorrector is
    * needed. (that would compute the average between front and
    * back displacement, which causes the mesh to move less 
    * than intended, and likely produces inverted elements).
    * The matching nodes are found by interating over mesh 
    * edges originating from the "front" node and getting the 
    * `otherVertex` located on the "back" patch.
    * 
    * Maybe it is possible to decrease the size of the frontToBack
    * list by only iterating over unique points (owned).
    */
    const auto& localMesh = motionSolver::mesh();
    const auto& patches = localMesh.boundaryMesh();
    labelList pointsToGlobal, uniquePoints;
    const auto globalIDs = *localMesh.globalData().mergePoints(pointsToGlobal, uniquePoints);

    label frontPatchID = patches.findPatchID("front");
    label backPatchID = patches.findPatchID("back");

    const polyPatch& frontPatch = patches[frontPatchID];
    const polyPatch& backPatch = patches[backPatchID];

    const labelList& frontPts = frontPatch.meshPoints();
    const labelList& backPts = backPatch.meshPoints();

    frontToBack.resize_fill(localMesh.nPoints(), -1);

    const labelListList& pointEdges = localMesh.pointEdges();
    const edgeList& edges = localMesh.edges();

    forAll(frontPts, fI)
    {
        const label fp = frontPts[fI];
        const labelList& pEdges = pointEdges[fp];
        forAll(pEdges, eI)
        {
            const edge& e = edges[pEdges[eI]];
            const label bp = e.otherVertex(fp);
            if (backPts.contains(bp))
            {
                frontToBack[fp] = bp;
                break;
            }
        }
    }
}
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::tmopMotionSolver::tmopMotionSolver
(
    const polyMesh& mesh,
    const IOdictionary& dict
)
:
    displacementMotionSolver(mesh, dict, typeName),
    fvMotionSolver(mesh),
    clusterMode_(this->coeffDict().get<bool>("clusterMode")),
    client_(clusterMode_),
    solutionDim_(
        std::count(
            fvMesh_.solutionD().cbegin(),
            fvMesh_.solutionD().cend(),
            1
        )
    ),
    validCmpts_(filterValidCmpts(fvMesh_.solutionD())),
    rankMeshPointsName_("points_MPI_" + std::to_string(Pstream::myProcNo())),
    rankMeshPointsIndicesName_("indices_MPI_" + std::to_string(Pstream::myProcNo())),
    rankMeshDisplacementsName_("displacements_MPI_" + std::to_string(Pstream::myProcNo())),
    rankMeshDistancesName_("distances_MPI_" + std::to_string(Pstream::myProcNo())),
    rankElementsName_("elements_MPI_" + std::to_string(Pstream::myProcNo())),
    boundaryPoints_(),
    boundaryDisplacements_()
{
    if (solutionDim_ != 2)
    {
        FatalErrorInFunction
            << "TMOP solver currently works on 2D cases only."
            << exit(FatalError);
    }
    writeSolutionDimToDatabase();
    writeMeshPointsToDatabase();
    writeDistanceToBoundary();
    writeBoundaryPointsToDatabase();
    writeMeshElements();
    matchFrontAndBack();
}

Foam::tmopMotionSolver::
tmopMotionSolver
(
    const polyMesh& mesh,
    const IOdictionary& dict,
    const pointVectorField& pointDisplacement,
    const pointIOField& points0
)
:
    displacementMotionSolver(mesh, dict, pointDisplacement, points0, typeName),
    fvMotionSolver(mesh),
    clusterMode_(dict.getOrDefault<bool>("clusterMode", true)),
    client_(clusterMode_),
    solutionDim_(
        std::count(
            fvMesh_.solutionD().cbegin(),
            fvMesh_.solutionD().cend(),
            1
        )
    ),
    validCmpts_(filterValidCmpts(fvMesh_.solutionD())),
    rankMeshPointsName_("points_MPI_" + std::to_string(Pstream::myProcNo())),
    rankMeshPointsIndicesName_("indices_MPI_" + std::to_string(Pstream::myProcNo())),
    rankMeshDisplacementsName_("displacements_MPI_" + std::to_string(Pstream::myProcNo())),
    rankMeshDistancesName_("distances_MPI_" + std::to_string(Pstream::myProcNo())),
    rankElementsName_("elements_MPI_" + std::to_string(Pstream::myProcNo())),
    boundaryPoints_(),
    boundaryDisplacements_()
{
    if (solutionDim_ != 2)
    {
        FatalErrorInFunction
            << "TMOP solver currently works on 2D cases only."
            << exit(FatalError);
    }
    writeSolutionDimToDatabase();
    writeMeshPointsToDatabase();
    writeDistanceToBoundary();
    writeBoundaryPointsToDatabase();
    writeMeshElements();
    matchFrontAndBack();
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::tmopMotionSolver::
~tmopMotionSolver() {}

// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::pointField> Foam::tmopMotionSolver::curPoints() const
{
    tmp<pointField> tcurPoints
    (
        points0() + pointDisplacement_.primitiveField()
    );
    pointField& curPoints = tcurPoints.ref();
    twoDCorrectPoints(curPoints);

    return tcurPoints;
}

void Foam::tmopMotionSolver::solve()
{
    // The points have moved so before interpolation update
    pointDisplacement_.boundaryFieldRef().evaluate();

    const auto& localMesh = motionSolver::mesh();
    const auto& meshBoundary = localMesh.boundaryMesh();
    const auto& boundaryDisplacements = pointDisplacement().boundaryField();
    labelList pointsToGlobal, uniquePoints;
    const auto globalIDs = *localMesh.globalData().mergePoints(pointsToGlobal, uniquePoints);
    // Build a map from global node label to local.
    // Needed because TMOP sends back nodes with their global ID,
    // and each processor needs to convert that to its local
    // numbering to assign the correct displacement.
    std::unordered_map<label, label> mpiRankGlobalToLocal;
    forAll(uniquePoints, pI)
    {
        const label u = uniquePoints[pI];
        mpiRankGlobalToLocal[pointsToGlobal[u]] = u;
    }

    List<vector> mpiRankDisplacements;
    List<label> mpiRankGIDs;
    forAll(boundaryDisplacements, patchI)
    {
        if (meshBoundary[patchI].type() == "empty"
           || meshBoundary[patchI].type() == "processor")
        {
           continue;
        }

        const auto& patch = meshBoundary[patchI];
        const auto& patchPoints = patch.meshPoints();

        tmp<vectorField> dispPtr = boundaryDisplacements[patchI].patchInternalField();
        const vectorField& disp  = dispPtr();

        forAll(disp, id)
        {
            const label patchId = patchPoints[id];
            if (disp[id].size() > 0) 
            {
                // track both the displacement and the global id
                // of the node that moves by that displacement
                mpiRankDisplacements.append(disp[id]);
                mpiRankGIDs.append(pointsToGlobal[patchId]);
            }
        }
    }

    // - Prepare global displacement and point lists for gather
    List<List<vector>>  globalDisplacementListList(Pstream::nProcs());
    List<List<label>>  globalGIDsListList(Pstream::nProcs());

    // - Assign data in the global lists list from this MPI rank
    globalDisplacementListList[Pstream::myProcNo()] = mpiRankDisplacements;
    globalGIDsListList[Pstream::myProcNo()] = mpiRankGIDs;

    // - Gather all data from all ranks at the main rank (0)
    Pstream::gatherList(globalDisplacementListList);
    Pstream::gatherList(globalGIDsListList);

    // - Send data to SmartRedis for ML model training from the main rank (0)
    if (Pstream::myProcNo() == 0)
    {
        // - Compute the global number of boundary points and displacements.

        label nGlobalBoundaryDisplacements= 0;
        forAll(globalDisplacementListList, rankI)
        {
            nGlobalBoundaryDisplacements += globalDisplacementListList[rankI].size();
        }
        boundaryDisplacements_.resize(nGlobalBoundaryDisplacements * solutionDim_);
        std::vector<label> gIDs(nGlobalBoundaryDisplacements);

        // - Agglomerate the gathered boundary List<List<vector>> points and
        // displacements into boundaryPoints_ and boundaryDisplacements_ attributes.
        label globalCmptI = 0;
        label globalJ = 0;
        forAll(globalDisplacementListList, rankI)
        {
            // Get the list of displacements from each rank
            const List<point>& rankDisplacements = globalDisplacementListList[rankI];
            const List<label>& rankGIDs = globalGIDsListList[rankI];

            // Assign rank points and rank displacements to boundaryPoints_ and
            // boundaryDisplacements_.
            // meshPoints [1,2,3],[4,5,6]
            // validCmpts [0,2] - xz axis is the solution plane.
            // globalPoints_ = [1,3,4,6] - viewed as [1,3], [4,6].

            // Iteration step is therefore point * solution dimension for
            // globalPoints_  and globalDisplacements_
            forAll(rankDisplacements, pointI)
            {
                gIDs[globalJ++] = rankGIDs[pointI];
                forAll(validCmpts_, dimI)
                {
                    boundaryDisplacements_[globalCmptI] = rankDisplacements[pointI][validCmpts_[dimI]];
                    ++globalCmptI;
                }
            }
        }

        client_.put_tensor(
            "displacements",
            boundaryDisplacements_.data(),
            {size_t(nGlobalBoundaryDisplacements), size_t(solutionDim_)},
            SRTensorTypeDouble,
            SRMemLayoutContiguous
        );
        client_.put_tensor(
            "displacements_gids",
            gIDs.data(),
            {size_t(nGlobalBoundaryDisplacements)},
            SRTensorTypeInt32,
            SRMemLayoutContiguous
        );

        client_.put_tensor(
            "data_ready",
            &solutionDim_,
            {1},
            SRTensorTypeInt32,
            SRMemLayoutContiguous
        );
    }

    bool displacements_ready = client_.poll_key("displacements_ready", 1, 300000);
    if (! displacements_ready)
    {
        FatalErrorInFunction
            << "Displacements not available in the SmartRedis database."
            << exit(Foam::FatalError);
    }
    else // Assign rank-displacements
    {
        // Allocate the displacements buffer.
        // const auto& meshPoints = fvMesh_.points();
        std::vector<double> rankMeshDisplacements(
            uniquePoints.size() * solutionDim_,
            0
        );

        // Allocate the buffer for global node ids
        std::vector<label> rankMeshIndices(
            uniquePoints.size(),
            0
        );

        // Unpack into the allocated displacements
        client_.unpack_tensor(
            rankMeshDisplacementsName_,
            rankMeshDisplacements.data(),
            {rankMeshDisplacements.size()},
            SRTensorTypeDouble,
            SRMemLayoutContiguous
        );

        // Unpack into the allocated global ids
        client_.unpack_tensor(
            rankMeshPointsIndicesName_,
            rankMeshIndices.data(),
            {rankMeshIndices.size()},
            SRTensorTypeInt32,
            SRMemLayoutContiguous
        );

        label globalId = 0;
        pointVectorField newDisplacement("newDisplacement", pointDisplacement_);
        // set the internalfield to zero (needed for the reduction later on)
        newDisplacement.internalFieldRef() = dimensionedVector("zero", dimensionSet(0,1,0,0,0), vector::zero);
        forAll(rankMeshIndices, gpointI)
        {
            // get local label from global label 
            label lLabel = -1;
            auto it = mpiRankGlobalToLocal.find(rankMeshIndices[gpointI]); 
            if (it != mpiRankGlobalToLocal.end())
                lLabel = it->second;
            forAll(validCmpts_, cmptI)
            {
                newDisplacement[lLabel][validCmpts_[cmptI]] = rankMeshDisplacements[globalId];
                // newDisplacement[frontToBack[lLabel]][validCmpts_[cmptI]] = rankMeshDisplacements[globalId];
                ++globalId;
            }
        }

        // Assign the same displacement as the "front" patch to the "back"
        const labelList& fPoints = meshBoundary[meshBoundary.findPatchID("front")].meshPoints();
        forAll(fPoints, fI)
        {
            const label& fL = fPoints[fI];
            forAll(validCmpts_, cI)
            {
                newDisplacement[frontToBack[fL]][validCmpts_[cI]] = newDisplacement[fL][validCmpts_[cI]];
            }
        }


        // Call to sync the value of shared nodes across processors.
        // E.g. if processor0 and processor1 share a boundary, 
        // the owner of points on that boundary might be processor0,
        // so the motion of those points is only known to processor0.
        // Here we use this call to let processor1 know that the 
        // boundary between processor1 and processor0 moves according
        // to the motion of processor0, which is the owner of the points.
        //
        // For each shared point, the function gathers all the displacements
        // from all the processors that share the point, then selects the 
        // one with largest magnitude as actual displacement.
        //
        // If we do not zero out the internal field above, this breaks.
        syncTools::syncPointList(
            localMesh,
            newDisplacement,
            maxMagSqrEqOp<vector>(),
            // eqOp<vector>(),
            vector::zero
        );


        //newDisplacement.boundaryFieldRef().evaluate();
        pointDisplacement_.internalFieldRef() = newDisplacement.internalField();
        pointDisplacement_.boundaryFieldRef().evaluate();

    }

    // Emulate MPI_Barrier() - wait for all MPI ranks to perform forward
    // inference of displacements and move the mesh with ML displacements.
    label totalRank = Pstream::myProcNo();
    reduce(totalRank, sumOp<label>(), totalRank);

    if (Pstream::myProcNo() == 0)
        client_.delete_tensor("displacements_ready");
}

// ************************************************************************* //
