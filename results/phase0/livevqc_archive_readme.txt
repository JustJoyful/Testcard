----------------------------------
LIVE Video Quality Challenge (VQC) Database 
----------------------------------

 Copyright Notice

-----------COPYRIGHT NOTICE STARTS WITH THIS LINE------------
Copyright (c) 2018 The University of Texas at Austin
All rights reserved.

Permission is hereby granted, without written agreement and without license or royalty fees, to use, copy, modify, and distribute this database (the images, the results and the source files) and its documentation for any purpose, provided that the copyright notice in its entirety appear in all copies of this database, and the original source of this database, Laboratory for Image and Video Engineering (LIVE, http://live.ece.utexas.edu ) at the University of Texas at Austin (UT Austin, http://www.utexas.edu ), is acknowledged in any publication that reports research using this database.

The following papers are to be cited in the bibliography whenever the database is used as:

1. Z. Sinno and A.C. Bovik, "Large-Scale Study of Perceptual Video Quality", IEEE Transactions on Image Processing, vol. 28, no. 2, pp. 612-627, February 2019 [paper]
2. Z. Sinno and A.C. Bovik, " Large Scale Subjective Video Quality Study", 2018 IEEE International Conference on Image Processing, Athens, Greece, October 2018. [paper]
3. Z. Sinno and A.C. Bovik, "LIVE Video Quality Challenge Database", Online: http://live.ece.utexas.edu/research/LIVEVQC/index.html, 2018.

IN NO EVENT SHALL THE UNIVERSITY OF TEXAS AT AUSTIN BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF THE USE OF THIS DATABASE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF TEXAS AT AUSTIN HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE UNIVERSITY OF TEXAS AT AUSTIN SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE DATABASE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF TEXAS AT AUSTIN HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

-----------COPYRIGHT NOTICE ENDS WITH THIS LINE------------ 

Xiangxu Yu.  Email: yuxiangxu@utexas.edu

REFERENCES
----------

1. Z. Sinno and A.C. Bovik, "Large-Scale Study of Perceptual Video Quality", IEEE Transactions on Image Processing, vol. 28, no. 2, pp. 612-627, February 2019 [paper]
2. Z. Sinno and A.C. Bovik, "Large Scale Subjective Video Quality Study", 2018 IEEE International Conference on Image Processing, Athens, Greece, October 2018. [paper]
3. Z. Sinno and A.C. Bovik, "LIVE Video Quality Challenge Database", Online: http://live.ece.utexas.edu/research/LIVEVQC/index.html, 2018.

DESCRIPTION
--------------

The great variations of videographic skills in videography, camera designs, compression and processing protocols, communication and bandwidth environments, and displays leads to an enormous variety of video impairments. Current no-reference (NR) video quality models are unable to handle this diversity of distortions. This is true in part because available video quality assessment databases contain very limited content, fixed resolutions, were captured using a small number of camera devices by a few videographers and have been subjected to a modest number of distortions. As such, these databases fail to adequately represent real world videos, which contain very different kinds of content obtained under highly diverse imaging conditions and are subject to authentic, complex and often commingled distortions that are difficult or impossible to simulate. As a result, NR video quality predictors tested on real-world video data often perform poorly. Towards advancing NR video quality prediction, we have constructed a large-scale video quality assessment database containing 585 videos of unique content, captured using 101 different devices (43 device models) by 80 different users with wide ranges of levels of complex, authentic distortions. We collected many subjective video quality scores via crowdsourcing. A total of 4776 unique participants took part in the study, yielding more than 205000 opinion scores, resulting in an average of 240 recorded human opinions per video. This study is the largest video quality assessment study ever conducted along several key dimensions: number of unique contents, capture devices, distortion types and combinations of distortions, study participants, and recorded subjective scores. 


SUBJECTIVE DATA FORMAT
----------------------

'data.mat' file includes subjective data corresponding to all 585 videos present in the 'Videos' folder.


ERRATUM TO THE PAPER
--------------------

These refined values were obtained after accounting for missing video scores. They differ only slightly from those reported in the paper.


SROCC values for VQA Models: Erratum to Table V in the paper. 

VIIDEO		: SROCC = -0.0206
NIQE      	: SROCC = 0.5599
BRISQUE    	: SROCC = 0.6017
V-BLIINDS   	: SROCC = 0.7010


PLCC values for VQA Models: Erratum to Table V in the paper.

VIIDEO		: PLCC = 0.1441
NIQE      	: PLCC = 0.5696
BRISQUE    	: PLCC = 0.6338
V-BLIINDS   	: PLCC = 0.7204


RMSE values for VQA Models: Erratum to Table V in the paper.

VIIDEO		: RMSE = 16.8796
NIQE      	: RMSE = 14.0196
BRISQUE    	: RMSE = 13.0520
V-BLIINDS  	 : RMSE = 11.7957


