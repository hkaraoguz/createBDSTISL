#include "bdst.h"

Level::Level()
{
    val = 0.0;
}
Level::~Level()
{
}

BDST::BDST(QObject *parent) :
    QObject(parent)
{


}
BDST::~BDST()
{
    if(levels.size() > 0)
        levels.clear();

}
