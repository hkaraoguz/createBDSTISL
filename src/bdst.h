#ifndef BDST_H
#define BDST_H

#include "Utility.h"
#include <QObject>
#include <QVector>



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
