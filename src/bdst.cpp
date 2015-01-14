#include "bdst.h"

Level::Level()
{
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
