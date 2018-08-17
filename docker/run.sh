#docker build -f Dockerfile -t mrbrains18/hust-lrde .
nvidia-docker load --input ../hust-lrde.tar
CONTAINERID=`nvidia-docker run -dit -v /home/canpi/MRBrainS18/data/training/1/orig:/input/orig:ro -v /home/canpi/MRBrainS18/data/training/1/pre:/input/pre:ro -v /output mrbrains18/hust-lrde`
nvidia-docker exec $CONTAINERID python test.py
nvidia-docker cp $CONTAINERID:/output /home/canpi/MRBrainS18
#docker stop $CONTAINERID
#docker rm -v $CONTAINERID


