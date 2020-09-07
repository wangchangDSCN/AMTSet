# AMTSet

*platform: Windows.

The notes for the folders:
* All the tracking results used in AMTSet are stored in the folders '.\results\results_SRE_CVPR13' and '.\results\results_TRE_CVPR13'.
* The annotation files (bounding box and attributes) are in the folder '.\anno'.
* The folder '.\initOmit' contains the annotation of frames that are omitted for tracking initialization due to occlusion or out of view of targets.
* The tracking results will be stored in the folder '.\results'.
* The folder '.\rstEval' contains some scripts used to compute the tracking performance or draw the results.
* The folder '.\trackers' contains the code for trackers
* The folder '.\tmp' is used to store some temporary results or log files.
* The folder '.\util' ontains some scripts used in the main functions.

1.Dataset<br>
  We proposed a dataset specififically for abrupt motion tracking (AMTSet), which mainly comes from three special scenes of camera switching, sudden dynamic change, and low frame rate video, so as to facilitate the evaluation of abrupt motion tracking.<br>
  All sequences of the dataset are provided in BaiduNetdisk:<br>
  Link：https://pan.baidu.com/s/1AQuMcy-CrH18haP8SP92KA 
  Passward：w984
  
2.Trackers<br>
  We tested 30 representative tracking methods and sorted them according to the final results.<br>
  All trackers:<br>
  Link：https://pan.baidu.com/s/1wJ0zKYjXiAGv6nam2JXRYg 
  Passward：ju3r
  The information for all trackers is listed in the file trackers.txt.<br>
  Full results of 30 trackers:<br>
  Link：https://pan.baidu.com/s/1WRJCEyxzsub37mbD_Itbyw 
  Passward：i5bu<br>
  
3.Main Functions<br>
  * main_running.m is the main function for the tracking test
		- It has the function to validate the results.
	* perfPlot.m is the main function for drawing precision plots and success plots.
		- It will call 'genPerfMat.m' to generate the values for plots.
  * perfPlotAMCR.m is the main function for drawing AMCR plots.
		- It will call 'genPerfMatAMCR.m' to generate the values for plots.
	* drawResultBB.m is the main function for drawing bounding boxes (BBs) of different trackers on each frame
  
  4.Acknowledgment<br>
  We thank Y. Wu, J. Lim, and M.-H. Yang for the "Object Tracking Benchmark" published on IEEE Transactions on Pattern Analysis & Machine Intelligence. Part of the code for this experiment comes from OTB100.
    
  
