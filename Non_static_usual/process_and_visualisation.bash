make -j
./main
echo "processing the results"
gnuplot SLAM.p
echo "creating the video"
ffmpeg -y -f gif -r 30 -i Slam.gif Slam.mp4
rm SLAM.p
rm temp
