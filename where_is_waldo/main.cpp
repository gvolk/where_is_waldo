#include "gui/guicontroller.h"
#include "src/controller.h"
#include "../src/where_is_waldo.h"

int main(int argc, char *argv[])
{

    GuiController gc(argc, argv);
    gc.run();

    string a = "";
    string b = "";
    run(a,b);

}
