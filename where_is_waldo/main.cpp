#include "gui/guicontroller.h"
#include "src/controller.h"
#include "../src/where_is_waldo.h"

int main(int argc, char *argv[])
{
    Controller c(argc, argv);
    c.run();
    //c.GetRefPoints("", "", QPoint(1010,1125));
}
