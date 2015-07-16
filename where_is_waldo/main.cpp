#include "gui/guicontroller.h"
#include "src/controller.h"
#include "../src/where_is_waldo.h"

int main(int argc, char *argv[])
{
    const char* a = "training_areas.ppm";
    const char* b = "training_areas_output.ppm";
    qDebug() << "Start Gauss";
    doGauss(a, b);
    qDebug() << "END Gauss";

    Controller c(argc, argv);
    c.run();



}
