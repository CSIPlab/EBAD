# extraction:
# mac use tar xopf
# linux use tar -xvf

# VOC 2007
wget http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar

# VOC 2012
# wget http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
# wget http://pjreddie.com/media/files/VOC2012test.tar
# tar -xf VOCtrainval_11-May-2012.tar
# tar -xf VOC2012test.tar

mv VOCdevkit VOC