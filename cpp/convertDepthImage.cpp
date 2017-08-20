std::string a_name = "depth228.dpt";
    std::ifstream l_file(a_name.c_str(),std::ofstream::in|std::ofstream::binary );
    
    if( l_file.fail() == true )
    {
        printf("cv_load_depth: could not open file for writing!\n");
        return NULL;
    }
    int l_row;
    int l_col;
    
    l_file.read((char*)&l_row,sizeof(l_row));
    l_file.read((char*)&l_col,sizeof(l_col));
    
    IplImage * lp_image = cvCreateImage(cvSize(l_col,l_row),IPL_DEPTH_16U,1);
    
    for(int l_r=0;l_r<l_row;++l_r)
    {
        for(int l_c=0;l_c<l_col;++l_c)
        {
            l_file.read((char*)&CV_IMAGE_ELEM(lp_image,unsigned short,l_r,l_c),sizeof(unsigned short));
        }
    }
    
    cv::Mat myMat = cv::cvarrToMat(lp_image);
    cv::imshow("Depth Image",myMat/(255.0*255.0));
    cv::imwrite("depth.jpg", myMat);
    cv::waitKey(0);
    
    l_file.close();