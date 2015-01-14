#include "bubble/bubbleprocess.h"
#include "imageprocess/imageprocess.h"
#include "database/databasemanager.h"
#include "bdst.h"
#include "Utility.h"
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int8.h>
#include <std_msgs/Int16.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <QDir>
#include <QDebug>
#include <QFile>
#include <sys/sysinfo.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

#include <stdio.h>
#include <stdlib.h>  /* The standard C libraries */
extern "C" {
#include "cluster.h"
}

/*  TODO: place detect edildiginde o place i okuyup agaca ekleme, agac yoksa olusturma ve merging kismi    */
/* Place leri db den okuma, bdst bottom up recognition*/
/* YAPILDI bdst level connection index de maximum u bulma. Bu zaten oradaki en ust baglantiyi temsil ediyor*/
/* Cost function u yaratma. Cost Function hesaplama kismi*/
/* Previous place in agacta nerede oldugunu bulup oradan search e baslamak*/

#define MIN_NO_PLACES 3

// A typedef for sorting places based on their distance and index
typedef std::pair<float,int> mypair;

// Comparator function for sorting
bool comparator ( const mypair& l, const mypair& r)
{ return l.first < r.first; }

// Addition operator for std::vector
template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());

    std::vector<T> result;
    result.reserve(a.size());

    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}



// Calculates the distance matrix based on mean invariants of places
double** calculateDistanceMatrix(int nrows, int ncols, double** data, int** mask);

// Calculates the cost function based on the first closest, second closest as well as the SVM results
float calculateCostFunction(float firstDistance, float secondDistance, Place closestPlace, Place detected_place);

// Perform the one-Class SVM calculation
float performSVM(Mat trainingVector, Mat testVector);

void performBDSTCalculations();

// Calculate the binary bdst
Node* calculateBinaryBDST(int nrows, int ncols, double** data);

// Binary den Merged bdst'ye gecis
void calculateMergedBDST(float tau_h, int nnodes, int noplaces, Node* tree, BDST* bdst);

// Calculate the mean invariants so that the robot can perform recognition
void calculateMeanInvariantsOfBDST(BDST* aLevel);

int performTopDownBDSTRecognition(float tau_g, float tau_l, BDST* bdst, Place detected_place);

//double calculateLevelCostFunction(Place place, std::vector<double> closestMeanInvariant, std::vector<double> secondClosestMeanInvariant, float voteRate);


// OPENCV modifikasyonu
double compareHistHK( InputArray _H1, InputArray _H2, int method );


// FOR debugging
std::vector< std::vector<float> > readInvariantVectors();

void trainSVM();


/*class UninformativeFrame
{
public:
    UninformativeFrame();
    int sat_mean;
    int sat_var;
    int frame_no;


};
UninformativeFrame*/
TemporalWindow::TemporalWindow()
{
    startPoint = 0;
    endPoint = 0;
    tau_w = 0;
    tau_n = 0;
    id = -1;

}

ros::Timer timer;

std::vector<LearnedPlace> places;

int learnedPlaceCounter = 1;

std::vector< std::vector<float> > invariants;

TopologicalMap topmap;

Place currentPlace;

bool performRecognition = false;

BDST* bdst ;

DatabaseManager dbmanager;

DatabaseManager knowledgedbmanager;

int lastTopMapNodeId;

void constructInvariantsMatrix(std::vector<LearnedPlace> plcs)
{
    if(invariants.size() > 0)
        invariants.clear();
    for(int i = 0 ; i < plcs.size(); i++)
    {
        std::vector<float> invariant = plcs.at(i).meanInvariant;
        invariants.push_back(invariant);
    }
}

void updateTopologicalMap(int node1, int node2)
{
    std::pair<int,int> mapNode;


    lastTopMapNodeId = node2;
    // Just to shift the id's from 0 to 1
    mapNode.first = node1;
    mapNode.second = node2;

    topmap.connections.push_back(mapNode);


    knowledgedbmanager.insertTopologicalMapRelation(topmap.connections.size(),mapNode);

    // DB'YE ILISKI YAZILACAK

}

LearnedPlace convertPlacetoLearnedPlace(Place place)
{
    LearnedPlace aplace;

    aplace.id = learnedPlaceCounter;
    aplace.meanInvariant = place.meanInvariant;
    aplace.memberInvariants = place.memberInvariants;
    aplace.memberIds = place.memberIds;

    if(aplace.memberPlaces.empty())
    {
        aplace.memberPlaces = cv::Mat(1,1,CV_16UC1);
        aplace.memberPlaces.at<unsigned short>(0,0) = (unsigned short)place.id;
    }

    knowledgedbmanager.insertLearnedPlace(aplace);

    learnedPlaceCounter++;
    /* else
    {
        Mat tempMat = cv::Mat(1,1,CV_16UC1);
        tempMat.at<unsigned short>(0,0) = (unsigned short)place.id;
        vconcat(tempMat,aplace.memberPlaces);
    }*/
    // aplace.memberPlaces.push_back(place.id);

    // learnedPlaceCounter++;
    return aplace;
}

void placeCallback(std_msgs::Int16 placeId)
{
    currentPlace = dbmanager.getPlace((int)placeId.data);

    if(places.size() < MIN_NO_PLACES && places.size() >= 1)
    {
        updateTopologicalMap(places.back().id,currentPlace.id);
    }

    if(places.size() < MIN_NO_PLACES)
    {
        LearnedPlace alearnedPlace = convertPlacetoLearnedPlace(currentPlace);
        places.push_back(alearnedPlace);
        qDebug()<<"Places size"<<places.size();
        return;

    }


    // places.push_back(aPlace);
    qDebug()<<"Places size"<<places.size();

    performRecognition = true;

}



int main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "createBDSTISL");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    ros::Subscriber sbc = nh.subscribe<std_msgs::Int16>("placeDetectionISL/placeID",5, placeCallback);

    //  QFile file("/home/hakan/ros_workspace/createBDSTISL/invariants.txt");

    //qDebug()<<sysinfo::bufferram;


    /*  const int nrows = 6;
      const int ncols = 600;
      double** data = new double*[nrows];
     int** mask = new int*[nrows];*/

    //   invariants = readInvariantVectors();

    // performBDSTCalculations();

    //  performBDSTCalculations();


    //  std::vector<float> resultt = invariants[0]+invariants[1];

    //  std::cout<<resultt[0]<<" "<<invariants[0][0]<<" "<<invariants[1][0]<<std::endl;



    /*  for(int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < ncols; j++ )
        {
            mask[i][j] = 1;

        }

    }*/


    ros::Rate loop(50);


    if(dbmanager.openDB("/home/hakan/Development/ISL/Datasets/Own/deneme/db1.db"))
    {
        qDebug()<<"Places db opened";
    }

    if(knowledgedbmanager.openDB("/home/hakan/Development/ISL/Datasets/Own/deneme/knowledge.db","knowledge"))
    {
        qDebug()<<"Knowledge db opened";
    }


    //  trainSVM();


    /*    cv::Mat mt(600,1,CV_32FC1);
    memcpy(mt.data,invariants[1].data(),invariants[1].size()*sizeof(float));

    aPlace.meanInvariant = mt;

    qDebug()<<aPlace.meanInvariant.at<float>(599,0);

    aPlace.id = 12;*/

    //  performTopDownBDSTRecognition(1,2,bdst, aPlace);

    /*   Place aPlace;

    if(dbmanager.openDB("/home/hakan/Development/ISL/Datasets/Own/deneme/db1.db"))
    {
       aPlace = DatabaseManager::getPlace(15);

       qDebug()<<aPlace.memberIds.rows<<aPlace.memberIds.cols;

       qDebug()<<aPlace.memberInvariants.rows<<aPlace.memberInvariants.cols;

       performTopDownBDSTRecognition(1.1,2,bdst, aPlace);

    }*/


    //   calculateCostFunction(2,3,aPlace,aPlace);



    //timer.start();

    while(ros::ok())
    {


        ros::spinOnce();

        if(places.size() >= MIN_NO_PLACES && performRecognition)
        {
            performRecognition = false;

            // Do we have a bdst?
            if(bdst)
            {
                int result = performTopDownBDSTRecognition(1.25,2,bdst,currentPlace);
                qDebug()<<"Recognized state"<<result;

                // We didn't recognize
                if(result < 0)
                {

                    // Place lastPlace = places.back();
                    LearnedPlace lastlearnedplace = places.back();

                    LearnedPlace anewLearnedPlace = convertPlacetoLearnedPlace(currentPlace);

                    updateTopologicalMap(lastTopMapNodeId,currentPlace.id);

                    places.push_back(anewLearnedPlace);

                    constructInvariantsMatrix(places);
                    performBDSTCalculations();
                }
                else
                {
                    // We should just update the place that bdst belongs to
                    // The topological map will not be updated only the last node should be updated

                    LearnedPlace recognizedPlace = places[result];

                    Mat totalMemberInvariants;

                    cv::hconcat(currentPlace.memberInvariants,recognizedPlace.memberInvariants,totalMemberInvariants);

                    recognizedPlace.memberInvariants = totalMemberInvariants;

                    recognizedPlace.calculateMeanInvariant();

                    Mat totalMemberIds;

                    cv::vconcat(currentPlace.memberIds,recognizedPlace.memberIds,totalMemberIds);

                    recognizedPlace.memberIds = totalMemberIds;

                    cv::Mat temp = cv::Mat(1,1,CV_16UC1);
                    temp.at<unsigned short>(0,0) = (unsigned short)currentPlace.id;
                    //  qDebug()<<temp.at<unsigned short>(0,0);

                    Mat totalMemberPlaces;

                    cv::vconcat(temp,recognizedPlace.memberPlaces,totalMemberPlaces);

                    recognizedPlace.memberPlaces = totalMemberPlaces;

                    for(int k = 0; k < recognizedPlace.memberPlaces.rows; k++)
                    {
                        qDebug()<<"Members of recognized place"<<recognizedPlace.memberPlaces.at<unsigned short>(k,0);
                    }

                    places[result] = recognizedPlace;

                    knowledgedbmanager.insertLearnedPlace(recognizedPlace);

                    updateTopologicalMap(lastTopMapNodeId,recognizedPlace.id);

                    lastTopMapNodeId = recognizedPlace.id;



                }
            }


        }
        else if(places.size() >= MIN_NO_PLACES)
        {
            // We have enough places, we should generate the bdst
            if(!bdst)
            {
                constructInvariantsMatrix(places);
                performBDSTCalculations();
                performRecognition = false;
            }
        }

        loop.sleep();

    }



    //  timer.stop();

    dbmanager.closeDB();

    ros::shutdown();

    if(bdst)
    bdst->deleteLater();

    //  qDebug()<<mt.rows;


    return 0;



}
void performBDSTCalculations()
{

    const int nrows = invariants.size();
    const int ncols = invariants[0].size();

    double** data = new double*[nrows];

    //  int** mask = new int*[nrows];

    int i;

    //double** distmatrix;


    for (i = 0; i < nrows; i++)
    {
        data[i] = new double[ncols];

        //mask[i] = new int[ncols];
    }


    for(i = 0; i < nrows; i++)
    {
        for(int j = 0 ; j < ncols; j++)
        {
            data[i][j] = invariants[i][j];
            //   qDebug()<<data[i][j];


        }

    }

    Node* binarytree = calculateBinaryBDST(nrows,ncols,data);

    if(bdst)
        bdst->deleteLater();

    bdst =  new BDST;

    calculateMergedBDST(0.25,nrows-1,nrows,binarytree,bdst);

    free(binarytree);

    for ( i = 0; i < nrows; i++){
        //delete [] mask[i];
        delete [] data[i];
    }
    // delete [] mask;
    delete [] data;


}

double** calculateDistanceMatrix(int nrows, int ncols, double** data, int** mask)
/* Calculate the distance matrix between genes using the Euclidean distance. */
{
    int i, j;
    double** distMatrix;
    double* weight = new double[ncols];
    printf("============ Euclidean distance matrix between genes ============\n");
    for (i = 0; i < ncols; i++) weight[i] = 1.0;
    distMatrix = distancematrix(nrows, ncols, data, mask, weight, 'e', 0);

    if (!distMatrix)
    {
        printf ("Insufficient memory to store the distance matrix\n");
        delete weight;
        return NULL;
    }
    // This part is for changing the values of distMatrix to the MATLAB format. Multiply by the length of the feature vector (600) and take the sqrt
    for(i = 0; i < nrows; i++)
    {
        for(j = 0; j< i; j++)
        {


            distMatrix[i][j] = sqrt(distMatrix[i][j]*600);
        }

    }
    printf("   Place:");
    for(i=0; i<nrows-1; i++) printf("%6d", i);
    printf("\n");
    for(i=0; i<nrows; i++)
    { printf("Gene %2d:",i);
        for(j=0; j<i; j++) printf(" %5.4f",distMatrix[i][j]);
        printf("\n");
    }
    printf("\n");
    delete weight;
    return distMatrix;


}

Node* calculateBinaryBDST(int nrows, int ncols, double** data)
{
    int** mask = new int*[nrows];

    double** distmatrix;

    int i = 0;


    for (i = 0; i < nrows; i++)
    {

        mask[i] = new int[ncols];
    }

    for (i = 0; i < nrows; i++)
    {


        mask[i] = new int[ncols];
    }

    for(int i = 0; i < nrows; i++)
    {
        for(int j = 0; j < ncols; j++ )
        {
            mask[i][j] = 1;

        }
    }

    distmatrix = calculateDistanceMatrix(nrows, ncols, data, mask);

    const int nnodes = nrows-1;

    Node* tree;

    printf("\n");
    printf("================ Pairwise single linkage clustering ============\n");
    /* Since we have the distance matrix here, we may as well use it. */
    tree = treecluster(nrows, ncols, 0, 0, 0, 0, 'e', 's', distmatrix);
    /* The distance matrix was modified by treecluster, so we cannot use it any
       * more. But we still need to deallocate it here.
       * The first row of distmatrix is a single null pointer; no need to free it.
       */
    for (i = 1; i < nrows; i++) delete distmatrix[i];
    delete distmatrix;



    if (!tree)
    { // Indication that the treecluster routine failed

        qDebug()<<"treecluster routine failed due to insufficient memory";

        return NULL;


    }
    else
    {
        for(i=0; i<nnodes; i++)

            qDebug()<<-i-1<<tree[i].left<<tree[i].right<<tree[i].distance;

        //delete tree;


        return tree;

    }

    return NULL;

}
void calculateMergedBDST(float tau_h, int nnodes, int noplaces, Node* tree, BDST* bdst)
{
    //int levelcount = 0;
    //int nodecount = 0;

    std::vector<TreeLeaf> leaves;

    QString homepath = QDir::homePath();

    homepath.append("/mergedbdst.txt");

    QFile file(homepath);
    if(file.open(QFile::WriteOnly))
    {
        qDebug()<<"BDST File is opened for writing";

    }

    QTextStream txtstr(&file);
    // BDST bdst;

    //  bdst.levels.resize(1);

    // places start from 0 to

    // These are the initial leaves, we will merge them
    for(int i = 0 ; i < nnodes; i++)
    {
        TreeLeaf leaf;

        // The left child
        leaf.left = tree[i].left;

        // If left child is less than 0 that means it is an inner node. MATLAB implementation uses positive index so we switch the index to positive
        if(leaf.left < 0) leaf.left=noplaces-leaf.left;

        // The right child
        leaf.right = tree[i].right;

        // If right child is less than 0 that means it is an inner node. MATLAB implementation uses positive index so we switch the index to positive
        if(leaf.right < 0) leaf.right=noplaces-leaf.right;

        // The value of the leaf
        leaf.val = tree[i].distance;

        // While building merged BDST we check whether the leaf is used or not
        leaf.isused = false;

        // Each leaf has a connection to the parent node. It is marked by this variable
        leaf.parentConnection = noplaces+i+1;

        leaves.push_back(leaf);

    }

    // Merging starts here
    for(uint i = 0; i < leaves.size(); i++)
    {
        TreeLeaf aLeaf = leaves[i];

        qDebug()<<"i is"<<i;

        if(!aLeaf.isused)
        {

            bool shouldADDLevel = true;

            // TODO
            qDebug()<<"Lastly i am here"<<bdst->levels.size();
            if(bdst->levels.size() > 0)
            {
                for(uint j = 0 ; j< bdst->levels.size(); j++)
                {
                    qDebug()<<"j is"<<j;
                    // Level aLevel = bdst.levels[j];

                    for(uint k = 0; k < bdst->levels[j].parentNodes.size(); k++)
                    {
                        if(aLeaf.parentConnection == bdst->levels[j].parentNodes.at(k))
                        {
                            shouldADDLevel = false;

                            aLeaf.isused = true;
                            if(aLeaf.left < noplaces)
                            {
                                bdst->levels[j].members.push_back(aLeaf.left);

                            }
                            else if(aLeaf.left > noplaces)
                            {
                                bdst->levels[j].parentNodes.push_back(aLeaf.left);

                            }
                            if(aLeaf.right < noplaces)
                            {
                                bdst->levels[j].members.push_back(aLeaf.right);

                            }
                            else if(aLeaf.right > noplaces)
                            {
                                bdst->levels[j].parentNodes.push_back(aLeaf.right);

                            }

                            if(aLeaf.right < noplaces && aLeaf.left < noplaces)
                             txtstr<<aLeaf.left+1<<" "<<aLeaf.right+1<<" "<<bdst->levels[j].val<<" "<<"\n";
                            else if(aLeaf.right < noplaces)
                                txtstr<<aLeaf.left<<" "<<aLeaf.right+1<<" "<<bdst->levels[j].val<<" "<<"\n";
                            else if(aLeaf.left < noplaces)
                                txtstr<<aLeaf.left+1<<" "<<aLeaf.right<<" "<<bdst->levels[j].val<<" "<<"\n";



                            // bdst.levels[j] = aLevel;

                            leaves[i]= aLeaf;

                            break;


                        }

                    }

                    if(!shouldADDLevel)
                        break;

                }

            }

            if(shouldADDLevel)
            {
                qDebug()<<"Level should be added";
                Level aLevel;

                aLevel.members.push_back(aLeaf.left);
                aLevel.members.push_back(aLeaf.right);

                if(aLeaf.right < noplaces && aLeaf.left < noplaces)
                 txtstr<<aLeaf.left+1<<" "<<aLeaf.right+1<<" "<<aLevel.val+tau_h<<" "<<"\n";
                else if(aLeaf.right < noplaces)
                    txtstr<<aLeaf.left<<" "<<aLeaf.right+1<<" "<<aLevel.val+tau_h<<" "<<"\n";
                else if(aLeaf.left < noplaces)
                    txtstr<<aLeaf.left+1<<" "<<aLeaf.right<<" "<<aLevel.val+tau_h<<" "<<"\n";


           //     txtstr<<aLeaf.left+1<<" "<<aLeaf.right+1<<" "<<aLevel.val+tau_h<<"\n";

                aLevel.parentNodes.push_back(aLeaf.parentConnection);

                aLeaf.isused = true;
                leaves[i].isused = true;

                int currentConnection = aLeaf.parentConnection;

                aLevel.val = aLeaf.val + tau_h;

                for(uint j = 0; j < leaves.size(); j++)
                {
                    // If the leaf is not used && and left member of the leaf is equal to the current connection && the value is in the range
                    if(!leaves[j].isused && leaves.at(j).left == currentConnection && leaves.at(j).val <= aLevel.val)
                    {
                        if(leaves[j].right < noplaces)
                        {
                            aLevel.members.push_back(leaves[j].right);

                            txtstr<<leaves[j].left<<" "<<leaves[j].right+1<<" "<<aLevel.val<<"\n";

                            //   aLevel.parentNodes.push_back(leaves[j].parentConnection);


                        }
                        else
                        {
                            aLevel.parentNodes.push_back(leaves[j].right);

                            txtstr<<leaves[j].left<<" "<<leaves[j].right<<" "<<aLevel.val<<"\n";


                        }


                        leaves[j].isused = true;
                        aLevel.parentNodes.push_back(leaves[j].parentConnection);

                        currentConnection = leaves[j].parentConnection;
                        //      break;


                    }
                    // If the leaf is not used && and right member of the leaf is equal to the current connection && the value is in the range
                    else if(!leaves[j].isused && leaves.at(j).right == currentConnection && leaves.at(j).val <= aLevel.val)
                    {
                        if(leaves[j].left < noplaces)
                        {
                            aLevel.members.push_back(leaves[j].left);
                            leaves[j].isused = true;

                            txtstr<<leaves[j].left+1<<" "<<leaves[j].right<<" "<<aLevel.val<<"\n";


                        }
                        else
                        {
                            aLevel.parentNodes.push_back(leaves[j].left);

                            txtstr<<leaves[j].left<<" "<<leaves[j].right<<" "<<aLevel.val<<"\n";


                        }

                        aLevel.parentNodes.push_back(leaves[j].parentConnection);



                        leaves[j].isused = true;

                        currentConnection = leaves[j].parentConnection;

                        //    break;


                    }
                    else if(leaves.at(j).val > aLevel.val)
                    {
                        leaves.at(j).isused = false;
                        //  j = j-1;
                        break;

                    }





                }



                bdst->levels.append(aLevel);



            }



        }


    }

    for(int j = 0; j < bdst->levels.size(); j++)
    {
        bdst->levels[j].connectionIndex = bdst->levels.at(j).parentNodes.back();



        // calculateMeanInvariantForBDSTLevel( &bdst->levels[j]);

    }

    calculateMeanInvariantsOfBDST( bdst);

    file.close();

    qDebug()<<"Finished";




}

std::vector< std::vector<float>   >  readInvariantVectors()
{
    QFile file("/home/hakan/ros_workspace/createBDSTISL/invariants.txt");

    std::vector< std::vector<float> > invariants;


    /****************** THIS PART IS FOR MANUALLY READING INVARIANTS (DEBUGGING ONLY)  ********************************/

    if(file.open(QFile::ReadOnly))
    {
        QTextStream stream(&file);

        // double num1,num2,num3,num4,num5,num6;
        int j = 0;

        while(!stream.atEnd())
        {
            // stream>>num1>>num2>>num3>>num4>>num5>>num6;
            QString line = stream.readLine();
            QStringList list = line.split("\t");
            //    qDebug()<<list<<list.size();
            if(j  == 0)
            {
                invariants.resize(list.size());

            }

            if(j < 600)
            {
                for(int i = 0 ; i < list.size(); i++)
                {
                    invariants[i].push_back(list.at(i).toFloat());

                }
                /* data[0][j] = list.at(0).toDouble();
                  data[1][j] = list.at(1).toDouble();
                  data[2][j] = list.at(2).toDouble();
                  data[3][j] = list.at(3).toDouble();
                  data[4][j] = list.at(4).toDouble();
                  data[5][j] = list.at(5).toDouble();*/

                //     qDebug()<<data[0][j]<<data[1][j]<<data[2][j];

                j++;
            }
        }


        file.close();

    }

    /***********************************************************************************************************************/

    int i;
    //  const int nrows = 6;
    //  const int ncols = 600;


    return invariants;
}
void trainSVM()
{

    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Set up training data
    // float labels[4] = {1.0, 1.0, 1.0, 1.0};
    // Mat labelsMat(4, 1, CV_32FC1, labels);

    Mat labelsMat;

    float trainingData[4][2] = { {501, 10}, {502, 12}, {501, 13}, {503, 10} };
    Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::ONE_CLASS;
    params.kernel_type = CvSVM::RBF;
    params.gamma = 0.5;
    params.nu = 0.5;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);

    Mat testData(1,2,CV_32FC1);

    testData.at<float>(0,0) = 501;
    testData.at<float>(0,1) = 10;

    // qDebug()<<testData.at<float>(0,1);

    qDebug()<<SVM.predict(testData,true);







}
static inline float computeSquare (float x) { return x*x; }

int performTopDownBDSTRecognition(float tau_g, float tau_l, BDST *bdst, Place detected_place)
{
    Level currentLevel;
    // qDebug()<<detected_place.memberIds.at<uint>(1,0);
    currentLevel = bdst->levels.at(bdst->levels.size()-1);
    // We are going from top to down
    while(1)// for(int i = bdst->levels.size()-1; i >= 0; i--)
    {


        // For the first level
        //    if(i == bdst->levels.size()-1)
        //    {
        //std::pair distances;

        std::vector< mypair> distpairs;
        mypair distpair;

        // For each member of the first level
        for(uint k = 0; k < currentLevel.members.size(); k++)
        {
            // Get the member number
            int aMember = currentLevel.members.at(k);

            float sum_of_elems = 0;

            std::vector<float> invariant;

            // If the member is a terminal node
            if(aMember < invariants.size())
            {
                invariant = invariants.at(aMember);
            }
            // If it is not a terminal node, we should get the mean Invariant
            else
            {
                for(uint j = 0; j < bdst->levels.size(); j++)
                {
                    if(bdst->levels.at(j).connectionIndex == aMember)
                    {
                        invariant = bdst->levels.at(j).meanInvariant;
                        break;
                    }
                }
            }

            // We get the place's mean invariant and transform to an std::vector
            std::vector<float>  placeInvariant = detected_place.meanInvariant;

            // This is the result string
            std::vector<float> result;

            // Now we take the difference between the member and detected place invariant
            std::transform(invariant.begin(),invariant.end(), placeInvariant.begin(),
                           std::back_inserter(result),
                           std::minus<float>());

            // Now we get the square of the result to eliminate minuses
            std::transform(result.begin(), result.end(), result.begin(), computeSquare);

            // We are summing the elements of the result
            sum_of_elems =std::accumulate(result.begin(),result.end(),0.0);//#include <numeric>

            sum_of_elems = sqrt(sum_of_elems);

            // We now take the square root
            // std::transform(result.begin(), result.end(), result.begin(), (float(*)(float)) sqrt);




            // We are now collecting the difference and the indexes
            distpair.first = sum_of_elems;
            distpair.second = aMember;

            distpairs.push_back(distpair);

            qDebug()<<"result"<<sum_of_elems;

        }

        // Now we are sorting in ascending order the distance and member pairs
        std::sort(distpairs.begin(),distpairs.end(),comparator);

        // We find the closest and second closest members
        mypair firstClosestMember = distpairs.at(0);
        mypair secondClosestMember = distpairs.at(1);

        // If it is not a terminal node, then we should go one level down
        if(firstClosestMember.second >= invariants.size())
        {
            for(uint j = 0; j < bdst->levels.size(); j++)
            {
                if(bdst->levels.at(j).connectionIndex == firstClosestMember.second)
                {
                    currentLevel = bdst->levels.at(j);
                    break;
                }
            }


        }
        // We have found the closest terminal node, now we should calculate the cost function and check if it is recognized
        else
        {
            qDebug()<<"Closest terminal node"<<firstClosestMember.second;
            qDebug()<<"Second closest terminal node"<<secondClosestMember.second;

            //  if(dbmanager.openDB("/home/hakan/Development/ISL/Datasets/Own/deneme/db1.db"))
            //   {
            Place aPlace = dbmanager.getPlace((firstClosestMember.second+1));

            float costValue =  calculateCostFunction(firstClosestMember.first,secondClosestMember.first,aPlace,detected_place);

            if(costValue <= tau_g)
            {
                qDebug()<<"Recognized";
                return firstClosestMember.second;
            }

            qDebug()<<"Not Recognized!!";

            return -1;

            //   qDebug()<<aPlace.memberInvariants.rows<<aPlace.memberInvariants.cols;

            //      performSVM(aPlace.memberInvariants,aPlace.memberInvariants);

            //     DatabaseManager::closeDB();

            //  break;

            // }



        }
        //    std::vector<double> levelMemberInvariant = bdst->levels.at(j).



        //  }




    }


    return -1;
}

/*float CvSVM::predict( const float* row_sample, int row_len, bool returnDFVal ) const
{
    assert( kernel );
    assert( row_sample );

    int var_count = get_var_count();
    assert( row_len == var_count );
    (void)row_len;

    int class_count = class_labels ? class_labels->cols :
                  params.svm_type == ONE_CLASS ? 1 : 0;

    float result = 0;
    cv::AutoBuffer<float> _buffer(sv_total + (class_count+1)*2);
    float* buffer = _buffer;

    if( params.svm_type == EPS_SVR ||
        params.svm_type == NU_SVR ||
        params.svm_type == ONE_CLASS )
    {
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)decision_func;
        int i, sv_count = df->sv_count;
        double sum = -df->rho;

        kernel->calc( sv_count, var_count, (const float**)sv, row_sample, buffer );
        for( i = 0; i < sv_count; i++ )
            sum += buffer[i]*df->alpha[i];

        std::cout<<sum;
        result = params.svm_type == ONE_CLASS && !returnDFVal ? (float)(sum > 0) : (float)sum;
    }
    else if( params.svm_type == C_SVC ||
             params.svm_type == NU_SVC )
    {
        CvSVMDecisionFunc* df = (CvSVMDecisionFunc*)decision_func;
        int* vote = (int*)(buffer + sv_total);
        int i, j, k;

        memset( vote, 0, class_count*sizeof(vote[0]));
        kernel->calc( sv_total, var_count, (const float**)sv, row_sample, buffer );
        double sum = 0.;

        for( i = 0; i < class_count; i++ )
        {
            for( j = i+1; j < class_count; j++, df++ )
            {
                sum = -df->rho;
                int sv_count = df->sv_count;
                for( k = 0; k < sv_count; k++ )
                    sum += df->alpha[k]*buffer[df->sv_index[k]];

                vote[sum > 0 ? i : j]++;
            }
        }

        for( i = 1, k = 0; i < class_count; i++ )
        {
            if( vote[i] > vote[k] )
                k = i;
        }
        result = returnDFVal && class_count == 2 ? (float)sum : (float)(class_labels->data.i[k]);
    }
    else
        CV_Error( CV_StsBadArg, "INTERNAL ERROR: Unknown SVM type, "
                                "the SVM structure is probably corrupted" );

    return result;
}
float CvSVM::predict( const Mat& _sample, bool returnDFVal ) const
{
    CvMat sample = _sample;
    return predict(&sample, returnDFVal);
}
float CvSVM::predict( const CvMat* sample, bool returnDFVal ) const
{
    float result = 0;
    float* row_sample = 0;


    result = predict( row_sample, get_var_count(), returnDFVal );



    if( sample && (!CV_IS_MAT(sample) || sample->data.fl != row_sample) )
        cvFree( &row_sample );

    return result;
}*/
void calculateMeanInvariantsOfBDST(BDST *bdst)
{
    for(int j =0 ; j < bdst->levels.size()-1; j++){

        Level* aLevel = &bdst->levels[j];

        std::vector<float> sum;

        for(int i = 0 ; i < aLevel->members.size(); i++)
        {
            if(i == 0)
            {
                if(aLevel->members.at(i) < invariants.size())
                {

                    sum = invariants[aLevel->members.at(i)];
                }
                else
                {
                    for(int k = 0; k < bdst->levels.size(); k++)
                    {
                        if(bdst->levels.at(k).connectionIndex == aLevel->members.at(i) )
                        {
                            sum = bdst->levels.at(k).meanInvariant;
                            break;

                        }
                    }
                }
            }
            else
            {

                if(aLevel->members.at(i) < invariants.size())
                {

                    sum = sum + invariants[aLevel->members.at(i)];
                }
                else
                {
                    for(int k = 0; k < bdst->levels.size(); k++)
                    {
                        if(bdst->levels.at(k).connectionIndex == aLevel->members.at(i) )
                        {
                            sum = sum + bdst->levels.at(k).meanInvariant;
                            break;

                        }
                    }
                }


            }
        }
        for(int i = 0; i < sum.size(); i++)
        {
            sum[i] = sum[i]/aLevel->members.size();
        }

        aLevel->meanInvariant = sum;
    }

}

float calculateCostFunction(float firstDistance, float secondDistance, Place closestPlace, Place detected_place)
{
    float result = -1;

    float firstPart = firstDistance;
    float secondPart = firstDistance/secondDistance;
    float votePercentage = 0;

    //  if(DatabaseManager::openDB("/home/hakan/Development/ISL/Datasets/Own/deneme/db1.db"))
    //  {
    // Place aPlace = DatabaseManager::getPlace(closestPlace.id);

    votePercentage= performSVM(closestPlace.memberInvariants,detected_place.memberInvariants);
    qDebug()<<"Vote percentage"<<votePercentage;

    //  Mat trainingVector;

    //  trainingVector = aPlace.memberInvariants;

    //qDebug()<<trainingVector.rows<<trainingVector.cols;


    //  DatabaseManager::closeDB();
    // }

    result = firstPart+secondPart+(1-votePercentage);

    return result;


}
float performSVM(Mat trainingVector, Mat testVector)
{
    //  Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

    float result = 0;

    cv::transpose(trainingVector,trainingVector);

    cv::transpose(testVector,testVector);

    Mat labelsMat;

    // Set up SVM's parameters
    CvSVMParams params;
    params.svm_type    = CvSVM::ONE_CLASS;
    params.kernel_type = CvSVM::RBF;
    params.gamma = (double)1.0/trainingVector.rows;
    params.nu = 0.5;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-6);

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingVector, labelsMat, Mat(), Mat(), params);

    // Mat testData(1,2,CV_32FC1);

    // testData.at<float>(0,0) = 501;
    // testData.at<float>(0,1) = 10;

    // qDebug()<<testData.at<float>(0,1);

    //Mat resultsVector;

    float summ = 0;
    for(int i = 0; i< testVector.rows; i++){

        //   Mat singleTest =
        summ+=  SVM.predict(testVector.row(i));
    }

    ///   cv::Scalar summ = cv::sum(resultsVector);

    result = (float)summ/testVector.rows;

    return result;

}
