# theOriginalProblem
computer vision classification of of bricks, balls, and cylinders

Sample data is given in the data folder.
Jupyternotebooks are included for testing.

Navigate to script/sample_student.py to see the actuall code, script/sample_student_too_slow.py yeilds better accuracy, but does not complete in the aloted time.
Run script/evaluate.py to test the implemantation.

Approached of script/sample_student.py explained:

First the image is passed to the clssify function. From there the image is converted to grayscale and a blur effect is applied to the image with a 3x3 averaging kernel. Next horizontal and vertical edge detection with the Sobel kernels is calculated and combined. A threshold is applied yeilding a 2D boolean array for the image. At this point a shortcut is used. As the number of true values increases the time complexity also increases, so a selection is randomly selected and returned. If there are not too many true values an accumulator is created. The accumulator is filled using the Hough transform with the modification of Peter Hart. Instead of representing a line with y=mx+b or in this case v=xu+y, the line is represented with ρ and θ relative to the origin. So, ρ=ysin(θ)+xcos(θ). At this point the the the values in the accumulator is higher when an edge is longer and straight. So, for the easy data and most of the medium the higher values correspond to the brick images with more straight edges, while the balls have the least, and cylinders fall in the middle.

This is by nomeans a stable solution that should be applied generally.