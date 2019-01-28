import os
fid = open('UCF101_RGB_1_split_0_new.testlist','w')
fa = open('UCF101_RGB_1_split_0.testlist','r')
#fb = open('/data/wjc/caffe_act1/data/JHMDB/list_2.txt','r')
#fc = open('/data/wjc/caffe_act1/data/JHMDB/list_3.txt','r')

alines = fa.readlines()
#blines = fb.readlines()
#clines = fc.readlines()
i=0
while i< len(alines):
  a=alines[i]
 # b=blines[i]
 # c=clines[i]
  b='/data/wjc/caffe_act1/data/UCF101/Frames/'
  c = b+a
#  print c[:-1]
  if os.path.exists(c[:-1]):
    fid.write(a)
#    if not(os.path.isdir(c[:-12])):
#      os.makedirs(c[:-12])
  i=i+1


