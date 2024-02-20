#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/core/ocl.hpp"
#include <fstream>
#include <limits>

using namespace std;
using namespace cv;
using namespace optflow;

const String keys = "{help h usage ? |      | print this message   }"
        "{@input        |      | image1.exr}"
        "{@output       |      | image2.flo}";
        
static Mat flowToDisplay(const Mat flow)
{
    Mat flow_split[2];
    Mat magnitude, angle;
    Mat hsv_split[3], hsv, rgb;
    split(flow, flow_split);
    cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    merge(hsv_split, 3, hsv);
    cvtColor(hsv, rgb, COLOR_HSV2BGR);
    return rgb;
}

static Mat flowTo3Channels(const Mat flow)
{
    Mat flow_split[2],rgb;
    Mat blue_chan = Mat::zeros(flow.size(), CV_32FC1);
    split(flow, flow_split);
    vector<Mat> channels;
    channels.push_back(flow_split[0]);
    channels.push_back(flow_split[1]);
    channels.push_back(blue_chan);
    merge(channels,rgb);
    return rgb;
}

static Mat exrTo2Channels(const Mat ima)
{
    Mat ima_split[4];
    Mat flow = Mat::zeros(ima.size(), CV_32FC2);
    split(ima, ima_split);
    vector<Mat> channels;
    channels.push_back(ima_split[2]);
    channels.push_back(ima_split[1]);
    merge(channels,flow);
    return flow;
}


int main( int argc, char** argv )
{
    CommandLineParser parser(argc, argv, keys);
    parser.about("convert .exr file to .flo");
    if ( parser.has("help") || argc < 2 )
    {
        parser.printMessage();
        printf("EXAMPLES:\n");
        printf("./exr2flo in.exr out.flo\n");
        return 0;
    }
    
    String inputfile = parser.get<String>(0);
    String outputfile = parser.get<String>(1);
    
    if ( !parser.check() )
    {
        parser.printErrors();
        return 0;
    }
    
    Mat i1;
    Mat_<Point2f> flow;
    i1 = imread(inputfile, -1); //unchanged
    
    if ( !i1.data )
    {
        printf("No exr image data \n");
        return -1;
    }
    
//    flow = Mat(i1.size[0], i1.size[1], CV_32FC2);

    //write result as .flow
    flow = exrTo2Channels(i1);
    cv::writeOpticalFlow(outputfile,flow);
    printf("writing flo file : %s\n",outputfile.c_str());
    
}
