#ifndef DEFINES_H
#define DEFINES_H

/*
 *  DATA DEFINES
*/

//#define REF_IMG "training_image.jpg"
#define REF_IMG "training_image.ppm"
//#define REF_AREA "training_areas.jpg"
#define REF_AREA "training_areas.ppm"

#define GAUSS_AREA "gauss_areas.ppm"



/*
 *  CLASSIFIER DEFINES
*/

//#define EPOCHS 1000
//#define LEARN_CONST .0001

#define CPU_MODE true
#define GPU_MODE false


#define EPOCHS 1
#define LEARN_CONST .001


/*
 *  FEATURE DEFINES
*/

#define FEAT_LEN 9


/*
 * TEST DEFINSE
*/

#define TEST_WALDO "../images/testimages/waldo.ppm"
#define TEST_WALDO2 "../images/testimages/waldo2.ppm"
#define TEST_NO_WALDO "../images/testimages/no_waldo.ppm"



#endif // DEFINES_H
