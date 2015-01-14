#ifndef BDST_H
#define BDST_H

#include "Utility.h"
#include <QObject>
#include <QVector>

class Level
{

public:

    Level();
    ~Level();
    std::vector<int> members;
    std::vector<int> parentNodes;
    double val;
    //  Connection index bize max parent node numarasini veriyor. Bu node bu noktadan bagli
    int connectionIndex;
    // Mean Invariant bize bu level in ortalama invariant vectorunu veriyor
    std::vector<float> meanInvariant;




};

class TreeLeaf
{
public:
    int left;
    int right;
    float val;
    int parentConnection;
    bool isused;




};



class BDST : public QObject
{
    Q_OBJECT
public:
    explicit BDST(QObject *parent = 0);
    ~BDST();
    QList<Level> levels;
    
signals:
    
public slots:
    
};

#endif // BDST_H
