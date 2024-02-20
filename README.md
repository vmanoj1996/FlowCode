**What's Changed?**

Updated to opencv-4


# FlowCode

utilities to convert from .flo format (optical flow) to .exr format

* flo2exr : convert .flo to .exr

    `usage: ./flo2exr [-quiet] in.flo out.exr`

* exr2flo : convert .exr to .flo

    `usage: ./exr2flo [-quiet] in.exr out.flo`
    
* flo2png : visualize .flo file into a png

    `usage: ./flo2png [-quiet] in.flo out.png [maxmotion]` 

* deepflow_opencv : compute flow field between im1 and im2 with opencv deepflow method

    `usage: ./deepflow_opencv im1.png im2.png out.flo` 
    
     ` ./deepflow_opencv --gpu im1.png im2.png out.flo` 
     
     ` ./deepflow_opencv --exr --gpu im1.png im2.png out.exr` 
     
     ` use -d to downsample the flow computation (-d=.5 will compute at half resolution)` 
        
Compile :

    you need to have opencv installed
    
    git clone --recursive https://github.com/lulu1315/FlowCode.git
    cd FlowCode
    mkdir build;cd build
    cmake ..
    make

* tinyexr : https://github.com/syoyo/tinyexr
* imageLib https://github.com/dscharstein/imageLib
* original Code : http://vision.middlebury.edu/flow/data/
* twixtor format : http://revisionfx.com/faq/motion_vector/

